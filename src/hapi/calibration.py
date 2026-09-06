"""Calibration module for the Hapi hydrological modeling framework.

The calibration module connects the parameter spatial distribution function
with both components of the spatial representation of the hydrological
process (conceptual model and spatial routing) to calculate the performance
of predicted runoff at known locations based on a given performance function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from Oasis.harmonysearch import HSapi
from Oasis.optimization import Optimization

from hapi.catchment import Catchment
from hapi.conceptual import ParameterBounds, ParameterSet
from hapi.runs import DistributedRun, LumpedRun
from hapi.wrapper import Wrapper

ROWS_MISMATCH_ERROR = "all input data should have the same number of rows"
COLUMNS_MISMATCH_ERROR = "all input data should have the same number of columns"
OBJECTIVE_FN_ARGS_ERROR = (
    "the objective function you have entered needs more inputs, "
    "please enter them in a list as *args"
)


def _check_optimization_args(api_obj_args: Any, api_solve_args: Any) -> None:
    """Check the two argument bundles the optimizer is handed are mappings.

    Both are unpacked with `**` inside Oasis, so anything else fails there rather than at the
    call that supplied it. Every calibration entry point makes the same pair of checks.

    Args:
        api_obj_args: Keyword arguments forwarded to the objective function.
        api_solve_args: Keyword arguments forwarded to the solver.

    Raises:
        TypeError: Either bundle is not a dict.
    """
    if not isinstance(api_obj_args, dict):
        raise TypeError(
            f"the objective-function arguments should be a dict, got "
            f"{type(api_obj_args).__name__}"
        )
    if not isinstance(api_solve_args, dict):
        raise TypeError(
            f"the solver arguments should be a dict, got {type(api_solve_args).__name__}"
        )


class Calibration:
    """Calibrates a catchment's parameters against observed discharge.

    Holds the catchment it calibrates rather than being one. It was a subclass, which meant it
    inherited a forty-attribute builder to use a dozen fields of, and inherited
    `plot_hydrograph` -- which reads `Qsim.loc[...]` and so could never work against the bare
    array this class's own `extract_discharge` produces. Composition removes that class of
    problem: nothing is inherited, so nothing can be inherited broken.

    The search space lives here too. `ParameterBounds` is read by nothing else, and it carries
    the `(snow, maxbas)` pair every trial vector is checked against, so it belongs beside the
    optimiser rather than on the model.

    Attributes:
        model: The catchment being calibrated. Build it first, then hand it over.
        bounds: The search space, once `read_parameters_bound` has run.
        objective_function: The metric being optimised.
        OFArgs: Extra arguments forwarded to it.
        OFvalue: The best objective value the optimiser found.
        best_parameters: The optimiser's answer -- the flat vector it searched over. Not a
            runnable parameter set: for a distributed calibration the winning vector still has
            to go through the spatial-distribution function to become the `(rows, cols, n)`
            array a run reads. The runnable set is `model.parameters`.
        Qsim: The simulated hydrograph at the gauge cells, as `extract_discharge` builds it --
            a bare array sized `(time_steps, n_gauges)`, which is what the objective function
            consumes.

    Examples:
        ```python
        >>> from hapi.calibration import Calibration      # doctest: +SKIP
        >>> from hapi.catchment import Catchment          # doctest: +SKIP
        >>> model = Catchment.from_yaml("coello.yaml")    # doctest: +SKIP
        >>> calibration = Calibration(model)              # doctest: +SKIP
        >>> calibration.read_parameters_bound(upper, lower)   # doctest: +SKIP
        >>> calibration.read_objective_function(rmse, [])     # doctest: +SKIP
        ```
    """

    def __init__(self, model: Catchment):
        """Wrap the catchment to be calibrated.

        Args:
            model: The catchment to calibrate, with its inputs read. Its `parameters` are
                replaced once per trial vector, so it comes back carrying the last set tried.

        Raises:
            TypeError: `model` is not a `Catchment`.
        """
        if not isinstance(model, Catchment):
            raise TypeError(
                f"Calibration takes the Catchment it calibrates, got "
                f"{type(model).__name__}; build the model first, then wrap it"
            )
        self.model = model
        self.bounds: ParameterBounds | None = None
        self.objective_function: Callable[..., Any] | None = None
        self.OFArgs: list | None = None
        self.OFvalue: float | None = None
        self.best_parameters: np.ndarray | list | None = None
        self.Qsim: np.ndarray | None = None

    def read_parameters_bound(
        self,
        upper_bound: list | np.ndarray,
        lower_bound: list | np.ndarray,
        snow: bool = False,
        maxbas: bool = False,
    ) -> None:
        """Read the search space the optimiser explores.

        Moved here from `Catchment`: nothing but a calibration reads it, and it carries the
        `(snow, maxbas)` pair that fixes how wide every trial vector must be -- which is the
        rule `_parameter_set` checks each one against.

        Args:
            upper_bound: Upper bound per parameter.
            lower_bound: Lower bound per parameter.
            snow: Whether the snow routine runs.
            maxbas: Whether the vector carries a MAXBAS value instead of Muskingum's two.

        Raises:
            ValueError: The bounds are different lengths, or `snow` is not a bool.
        """
        if not isinstance(snow, bool):
            raise ValueError(
                "snow input defines whether to consider snow subroutine or not it has to "
                "be True or False"
            )
        self.bounds = ParameterBounds(
            lower_bound, upper_bound, snow=snow, maxbas=maxbas
        )
        logger.debug("Parameters' bounds are read successfully")

    def _declare_the_parameter_variables(
        self, opt_prob: Optimization, initial_values: list | None = None
    ) -> None:
        """Add one continuous optimisation variable per parameter, bounded by LB and UB.

        Every calibration entry point declares the same variables the same way; only the
        lumped one can also seed them with a starting point.

        Args:
            opt_prob: The problem being built.
            initial_values: One starting value per parameter, or None to let the optimiser
                choose.

        Raises:
            ValueError: `initial_values` is given and does not hold one value per parameter.
        """
        # One starting value per parameter. A shorter list used to index out of range
        # part-way through building the problem, naming neither argument and leaving
        # `opt_prob` half-populated.
        seeded = initial_values is not None and len(initial_values) > 0
        if seeded and len(initial_values) != len(self.bounds):
            raise ValueError(
                f"initial_values must hold one value per parameter; the bounds define "
                f"{len(self.bounds)} and {len(initial_values)} were given"
            )

        for i in range(len(self.bounds)):
            seed = {"value": initial_values[i]} if seeded else {}
            opt_prob.addVar(
                f"x{i}",
                type="c",
                lower=self.bounds.lower[i],
                upper=self.bounds.upper[i],
                **seed,
            )

    def _parameter_set(self, values) -> ParameterSet:
        """Wrap a trial vector as a checked `ParameterSet`.

        The width rule needs the `(snow, maxbas)` pair, which a calibration supplies through
        `read_parameters_bound` rather than by reading a parameter file. Either source works;
        this picks whichever ran.

        Args:
            values: The trial parameter array or vector.

        Returns:
            ParameterSet: The set, its width checked against the configuration.

        Raises:
            ValueError: The trial set is not the width the configuration requires.
        """
        if self.model.parameters is not None:
            return self.model.parameters.with_values(values)
        bounds = self.bounds
        snow = bounds.snow if bounds is not None else False
        maxbas = bounds.maxbas if bounds is not None else False
        return ParameterSet(values, snow=snow, maxbas=maxbas)

    def read_objective_function(
        self, objective_function: Callable[..., Any], args: list | None
    ):
        """Read and store the objective function and its arguments.

        Takes the objective function and any additional arguments that
        need to be passed to the objective function during calibration.

        Args:
            objective_function (callable): A callable function to calculate
                any kind of metric to be used in the calibration.
            args: Any positional or keyword arguments to pass to the
                objective function. If None, defaults to an empty list.

        Raises:
            TypeError: If objective_function is not callable.
        """
        # check objective_function
        if not callable(objective_function):
            raise TypeError(
                f"The Objective function should be a function, got "
                f"{type(objective_function).__name__}"
            )
        self.objective_function = objective_function

        if args is None:
            args = []

        self.OFArgs = args

        print("Objective function is read successfully")

    def extract_discharge(
        self,
        calculate_metrics: bool = True,
        factor: list | None = None,
    ):
        """Extract the simulated discharge hydrograph at gauge locations.

        Extracts discharge values from the total routed discharge array
        (`self.model.results.q_total`) at each gauge location and stores them in
        `self.Qsim`. Optionally applies a multiplication factor per
        gauge.

        Args:
            calculate_metrics (bool, optional): Whether to calculate
                performance metrics. Not used in this override but
                kept so the signature matches the one it overrides.
                Default is True.
            factor (list, optional): List of multiplication factors for
                the simulated discharge, one per gauge. If None, no
                scaling is applied. Default is None.

        Raises:
            ValueError: The results came from MAXBAS routing, whose per-cell values are
                contributions rather than discharges.
        """
        if not self.model.results.outlet_shortcut_valid:
            raise ValueError(
                "this catchment was run with triangular (MAXBAS) routing, which sends "
                "every cell straight to the outlet: a single cell of q_total is that cell's "
                "contribution, not the discharge at it, so reading the gauge cells would "
                "under-report every hydrograph and the objective function would be "
                "calibrated against the wrong signal."
            )

        self.Qsim = np.zeros((self.model.meteo.time_steps, len(self.model.GaugesTable)))
        # error = 0
        for i in range(len(self.model.GaugesTable)):
            Xind = int(
                self.model.GaugesTable.loc[self.model.GaugesTable.index[i], "cell_row"]
            )
            Yind = int(
                self.model.GaugesTable.loc[self.model.GaugesTable.index[i], "cell_col"]
            )
            # gaugeid = self.model.GaugesTable.loc[self.model.GaugesTable.index[i],"id"]

            # Quz = self.model.results.quz_routed[Xind,Yind,:-1]
            # Qlz = self.model.results.qlz_translated[Xind,Yind,:-1]
            # self.Qsim[:,i] = Quz + Qlz

            Qsim = np.reshape(
                self.model.results.q_total[Xind, Yind, :-1], self.model.meteo.time_steps
            )

            if factor is not None:
                self.Qsim[:, i] = Qsim * factor[i]
            else:
                self.Qsim[:, i] = Qsim

            # Qobs = Coello.QGauges.loc[:,gaugeid]
            # error = error + objective_function(Qobs, Qsim)

        # return error

    def run_calibration(
        self,
        spatial_var_fun: Callable[..., Any],
        optimization_args: list,
        print_error: int | None = None,
    ):
        """Run the calibration algorithm for the distributed hydrological model.

        Executes the Harmony Search optimization algorithm to calibrate
        parameters for the conceptual distributed hydrological model.
        The method distributes parameters spatially using `spatial_var_fun`,
        runs the RRM model via `Wrapper.run_muskingum`, and evaluates
        performance using the stored objective function.

        The following attributes must be set on the instance before calling
        this method:

            - `Prec`, `ET`, `Temp`: Meteorological input arrays.
            - `flow_dir_arr`: Flow direction array.
            - `rows`, `cols`: Grid dimensions.
            - `LB`, `UB`: Lower and upper parameter bounds.
            - `objective_function`: Objective function for evaluation.
            - `QGauges`, `GaugesTable`: Observed discharge data and
              gauge metadata.

        Args:
            spatial_var_fun: Spatial variable function object with a
                `Function` method that distributes parameters and a
                `Par3d` attribute holding the 3D parameter array, plus
                `no_parameters` and `no_elem` attributes.
            optimization_args: A list of three elements:
                - `optimization_args[0]` (dict): Harmony Search API
                  objective arguments (e.g., HMS, HMCR, PAR).
                - `optimization_args[1]`: Parallel type for the
                  optimizer.
                - `optimization_args[2]` (dict): Solver arguments with
                  keys `"store_sol"`, `"display_opts"`,
                  `"store_hst"`, and `"hot_start"`.
            print_error: If not 0, prints the error value and parameters
                at each iteration. Default is None.

        Returns:
            tuple: Optimization result tuple containing:
                - res[0]: The optimal objective function value.
                - res[1]: The optimal parameter set.

        Raises:
            ValueError: If input dimensions are inconsistent.
            TypeError: If either bundle of optimization arguments is not a
                dict.
        """
        # input dimensions
        # [rows,cols] = self.FlowAcc.ReadAsArray().shape
        [fd_rows, fd_cols] = self.model.flow_network.flow_dir_arr.shape
        if (
            fd_rows != self.model.flow_network.rows
            or fd_cols != self.model.flow_network.cols
        ):
            raise ValueError(ROWS_MISMATCH_ERROR)

        # The three cubes already agree with each other (checked when MeteoInputs was
        # built); this is the other half -- that they cover the model's grid.
        self.model.meteo.validate_against(
            self.model.flow_network.rows,
            self.model.flow_network.cols,
            self.model.period.date_index,
        )

        # basic inputs
        # check if all inputs are included
        # assert all(["p2","init_st","UB","LB","snow "][i] in basic_inputs.keys()
        #     for i in range(4)), "basic_inputs should contain ['p2','init_st','UB','LB']"

        ### optimization

        # get arguments
        api_obj_args = optimization_args[0]
        pll_type = optimization_args[1]
        api_solve_args = optimization_args[2]
        # check optimization arguement
        _check_optimization_args(api_obj_args, api_solve_args)

        print("Calibration starts")

        ### calculate the objective function
        def opt_fun(par):
            # Distributing the parameters and narrowing the model both happen *outside* the
            # try. They are checks on the setup, not on this candidate: a wrong-width vector or
            # a grid mismatch is a bug to surface, and scoring it `nan` would let the optimiser
            # search on over a model that never ran -- which is what the bare `except` did.
            spatial_var_fun.Function(par)
            self.model.parameters = self._parameter_set(spatial_var_fun.Par3d)
            run = DistributedRun.from_model(self.model)

            try:
                self.model.results = Wrapper.run_muskingum(run)
                # calculate performance of the model
                try:
                    error = self.objective_function(
                        self.model.QGauges, *[self.model.GaugesTable]
                    )  # self.model.results.qout, self.model.results.quz_routed, self.model.results.qlz_translated,
                    f = list(range(9, len(par), spatial_var_fun.no_parameters))
                    g = list()
                    for i in range(len(f)):
                        k = par[f[i]]
                        x = par[f[i] + 1]
                        g.append(2 * k * x / self.model.period.dt)
                        g.append((2 * k * (1 - x)) / self.model.period.dt)

                except TypeError as e:
                    # the objective function received fewer inputs than it needs
                    raise ValueError(OBJECTIVE_FN_ARGS_ERROR) from e

                # print error
                if print_error != 0:
                    print(round(error, 3))
                    print(par)

                fail = 0
            except Exception as exc:
                # A genuine numerical failure for this candidate. Narrowed from a bare
                # `except`, which also caught KeyboardInterrupt -- so a long calibration
                # could not be stopped -- and reported every defect as a bad parameter set.
                logger.warning(f"trial failed, scoring it infeasible: {exc!r}")
                error = np.nan
                g = []
                fail = 1

            return error, g, fail

        ### define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)
        self._declare_the_parameter_variables(opt_prob)

        opt_prob.addObj("f")

        for i in range(spatial_var_fun.no_elem):
            opt_prob.addCon("g" + str(i) + "-1", "i")
            opt_prob.addCon("g" + str(i) + "-2", "i")

        print(opt_prob)

        opt_engine = HSapi(pll_type=pll_type, options=api_obj_args)

        store_sol = api_solve_args["store_sol"]
        display_opts = api_solve_args["display_opts"]
        store_hst = api_solve_args["store_hst"]
        hot_start = api_solve_args["hot_start"]

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.best_parameters = res[1]
        self.OFvalue = res[0]

        return res

    def calibrate_maxbas(
        self,
        spatial_var_fun: Callable[..., Any],
        optimization_args: list,
        print_error: int | None = None,
    ):
        """Run calibration using the FW1 (Focussed Width-1) routing scheme.

        Executes the Harmony Search optimization algorithm to calibrate
        parameters for the conceptual distributed hydrological model using
        the FW1 routing approach via `Wrapper.run_maxbas`.

        The following attributes must be set on the instance before calling
        this method:

            - `Prec`, `ET`, `Temp`: Meteorological input arrays.
            - `rows`, `cols`: Grid dimensions.
            - `LB`, `UB`: Lower and upper parameter bounds.
            - `objective_function`: Objective function for evaluation.
            - `QGauges`, `GaugesTable`: Observed discharge data and
              gauge metadata.

        Args:
            spatial_var_fun: Spatial variable function object with a
                `Function` method that distributes parameters and a
                `Par3d` attribute holding the 3D parameter array.
            optimization_args: A list of three elements:
                - `optimization_args[0]` (dict): Harmony Search API
                  objective arguments (e.g., HMS, HMCR, PAR).
                - `optimization_args[1]`: Parallel type for the
                  optimizer.
                - `optimization_args[2]` (dict): Solver arguments with
                  keys `"store_sol"`, `"display_opts"`,
                  `"store_hst"`, and `"hot_start"`.
            print_error: If not 0, prints the error value and parameters
                at each iteration. Default is None.

        Returns:
            tuple: Optimization result tuple containing:
                - res[0]: The optimal objective function value.
                - res[1]: The optimal parameter set.

        Raises:
            ValueError: If input dimensions are inconsistent.
            TypeError: If either bundle of optimization arguments is not a
                dict.
        """
        # input dimensions
        # [rows,cols] = self.FlowAcc.ReadAsArray().shape
        # [fd_rows,fd_cols] = self.flow_dir_arr.shape
        # assert fd_rows == self.rows and fd_cols == self.cols, ROWS_MISMATCH_ERROR

        # The three cubes already agree with each other (checked when MeteoInputs was
        # built); this is the other half -- that they cover the model's grid.
        self.model.meteo.validate_against(
            self.model.flow_network.rows,
            self.model.flow_network.cols,
            self.model.period.date_index,
        )

        # basic inputs
        # check if all inputs are included
        # assert all(["p2","init_st","UB","LB","snow "][i] in basic_inputs.keys()
        #     for i in range(4)), "basic_inputs should contain ['p2','init_st','UB','LB']"

        ### optimization

        # get arguments
        api_obj_args = optimization_args[0]
        pll_type = optimization_args[1]
        api_solve_args = optimization_args[2]
        # check optimization arguement
        _check_optimization_args(api_obj_args, api_solve_args)

        print("Calibration starts")

        # calculate the objective function
        def opt_fun(par):
            # See run_calibration: the setup checks belong outside the try, so a wrong-width
            # vector or a grid mismatch surfaces instead of being scored `nan`.
            spatial_var_fun.Function(par)
            self.model.parameters = self._parameter_set(spatial_var_fun.Par3d)
            run = DistributedRun.from_model(self.model, needs_flow_direction=False)

            try:
                self.model.results = Wrapper.run_maxbas(run)
                # calculate performance of the model
                try:
                    error = self.objective_function(
                        self.model.QGauges,
                        self.model.results.qout,
                        *[self.model.GaugesTable],
                    )
                except TypeError as e:
                    # the objective function received fewer inputs than it needs
                    raise ValueError(OBJECTIVE_FN_ARGS_ERROR) from e

                # print error
                if print_error != 0:
                    print(round(error, 3))
                    print(par)

                fail = 0
            except Exception as exc:
                # See run_calibration: narrowed from a bare `except`.
                logger.warning(f"trial failed, scoring it infeasible: {exc!r}")
                error = np.nan
                fail = 1

            return error, [], fail

        # define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)
        self._declare_the_parameter_variables(opt_prob)

        print(opt_prob)

        opt_engine = HSapi(pll_type=pll_type, options=api_obj_args)

        store_sol = api_solve_args["store_sol"]
        display_opts = api_solve_args["display_opts"]
        store_hst = api_solve_args["store_hst"]
        hot_start = api_solve_args["hot_start"]

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.best_parameters = res[1]
        self.OFvalue = res[0]

        return res

    def calibrate_lumped(
        self,
        basic_inputs: dict,
        optimization_args: list,
        print_error: int | None = None,
    ):
        """Run the calibration algorithm for the lumped hydrological model.

        Executes the Harmony Search optimization algorithm to calibrate
        parameters for the lumped conceptual hydrological model. The
        method runs the model via `Wrapper.run_lumped` and evaluates
        performance using the stored objective function. Muskingum
        routing constraints are enforced as inequality constraints.

        The following attributes must be set on the instance before calling
        this method:

            - `LB`, `UB`: Lower and upper parameter bounds.
            - `objective_function`: Objective function for evaluation.
            - `OFArgs`: Arguments for the objective function.
            - `QGauges`: Observed discharge DataFrame.
            - `dt`: Time step duration.

        Args:
            basic_inputs (dict): Dictionary containing:
                - `"Route"` (int): Routing flag (1 to enable routing).
                - `"RoutingFn"` (callable): Routing function to use.
                - `"InitialValues"` (list, optional): Initial parameter
                  values for the optimizer. Defaults to an empty list if
                  not provided.
            optimization_args: A list of three elements:
                - `optimization_args[0]` (dict): Harmony Search API
                  objective arguments (e.g., HMS, HMCR, PAR).
                - `optimization_args[1]`: Parallel type for the
                  optimizer.
                - `optimization_args[2]` (dict): Solver arguments with
                  keys `"store_sol"`, `"display_opts"`,
                  `"store_hst"`, and `"hot_start"`.
            print_error: If not 0, prints the error value and constraint
                values at each iteration. Default is None.

        Returns:
            tuple: Optimization result tuple containing:
                - res[0]: The optimal objective function value.
                - res[1]: The optimal parameter set.

        Raises:
            ValueError: If `basic_inputs` is missing required keys
                `"Route"` or `"RoutingFn"`, or if `"InitialValues"` is
                given and does not hold one value per parameter.
            TypeError: If either bundle of optimization arguments is not a
                dict.
        """
        # basic inputs
        # check if all inputs are included
        missing = [key for key in ("Route", "RoutingFn") if key not in basic_inputs]
        if missing:
            raise ValueError(
                f"basic_inputs should contain 'Route' and 'RoutingFn'; "
                f"{', '.join(missing)} is missing"
            )

        route = basic_inputs["Route"]
        routing_fn = basic_inputs["RoutingFn"]
        if "InitialValues" in basic_inputs:
            initial_values = basic_inputs["InitialValues"]
        else:
            initial_values = []

        ### optimization

        # get arguments
        api_obj_args = optimization_args[0]
        pll_type = optimization_args[1]
        api_solve_args = optimization_args[2]
        # check optimization arguement
        _check_optimization_args(api_obj_args, api_solve_args)

        print("Calibration starts")

        ### calculate the objective function
        def opt_fun(par):
            # See run_calibration: the setup checks belong outside the try.
            self.model.parameters = self._parameter_set(par)
            run = LumpedRun.from_model(self.model)

            try:
                self.model.results = Wrapper.run_lumped(run, route, routing_fn)
                self.Qsim = self.model.results.q_total
                # calculate performance of the model
                try:
                    error = self.objective_function(
                        self.model.QGauges[self.model.QGauges.columns[-1]],
                        self.Qsim,
                        *self.OFArgs,
                    )
                    g = [
                        2 * par[-2] * par[-1] / self.model.period.dt,
                        (2 * par[-2] * (1 - par[-1])) / self.model.period.dt,
                    ]
                except TypeError as e:
                    # the objective function received fewer inputs than it needs
                    raise ValueError(OBJECTIVE_FN_ARGS_ERROR) from e

                if print_error != 0:
                    print(
                        f"Error = {round(error, 3)} Inequality Const = {np.round(g, 2)}"
                    )
                    # print(par)
                fail = 0
            except Exception as exc:
                # A genuine numerical failure for this candidate. Narrowed from a bare
                # `except`, which also caught KeyboardInterrupt -- so a long calibration
                # could not be stopped -- and reported every defect as a bad parameter set.
                logger.warning(f"trial failed, scoring it infeasible: {exc!r}")
                error = np.nan
                g = []
                fail = 1
            return error, g, fail

        ### define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)

        self._declare_the_parameter_variables(opt_prob, initial_values)

        opt_prob.addObj("f")

        opt_prob.addCon("g1", "i")
        opt_prob.addCon("g2", "i")
        # print(opt_prob)
        opt_engine = HSapi(pll_type=pll_type, options=api_obj_args)

        # parse the api_solve_args inputs
        # availablekeys = ['store_sol',"display_opts","store_hst","hot_start"]
        store_sol = api_solve_args["store_sol"]
        display_opts = api_solve_args["display_opts"]
        store_hst = api_solve_args["store_hst"]
        hot_start = api_solve_args["hot_start"]

        # for i in range(len(availablekeys)):
        # if availablekeys[i] in api_solve_args.keys():
        # exec(availablekeys[i] + "=" + str(api_solve_args[availablekeys[i]]))
        # print(availablekeys[i] + " = " + str(api_solve_args[availablekeys[i]]))

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.OFvalue = res[0]
        self.best_parameters = res[1]

        return res
