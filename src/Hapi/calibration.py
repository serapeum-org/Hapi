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
from Oasis.harmonysearch import HSapi
from Oasis.optimization import Optimization

from Hapi.catchment import Catchment
from Hapi.wrapper import Wrapper


class Calibration(Catchment):
    """Calibration class for distributed hydrological model parameter optimization.

    The Calibration class connects the parameter spatial distribution function
    with both components of the spatial representation of the hydrological
    process (conceptual model and spatial routing) to calculate the
    performance of predicted runoff at known locations based on a given
    performance function.

    The Calibration class is a subclass of the Catchment superclass, so you
    need to create the Catchment object first to be able to run the
    calibration.
    """

    def __init__(
        self,
        name: Any,
        start: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        spatial_resolution: str | None = "Lumped",
        temporal_resolution: str | None = "Daily",
        routing_method: str | None = "Muskingum",
    ):
        """Initialize the Calibration object.

        Args:
            name (Any): Name of the Catchment.
            start (str): Starting date as a string.
            end (str): End date as a string.
            fmt (str, optional): Format of the given date.
                Default is "%Y-%m-%d".
            spatial_resolution (str, optional): Spatial resolution mode,
                either "Lumped" or "Distributed". Default is "Lumped".
            temporal_resolution (str, optional): Temporal resolution mode,
                either "Hourly" or "Daily". Default is "Daily".
            routing_method (str, optional): Routing method name.
                Default is "Muskingum".
        """
        super().__init__(
            name,
            start,
            end,
            fmt,
            spatial_resolution,
            temporal_resolution,
            routing_method,
        )

    def read_objective_function(self, objective_function: Callable[..., Any], args):
        """Read and store the objective function and its arguments.

        Takes the objective function and any additional arguments that
        need to be passed to the objective function during calibration.

        Args:
            objective_function (callable): A callable function to calculate
                any kind of metric to be used in the calibration.
            args: Any positional or keyword arguments to pass to the
                objective function. If None, defaults to an empty list.

        Raises:
            AssertionError: If objective_function is not callable.
        """
        # check objective_function
        assert callable(
            objective_function
        ), "The Objective function should be a function"
        self.objective_function = objective_function

        if args is None:
            args = []

        self.OFArgs = args

        print("Objective function is read successfully")

    def extract_discharge(
        self,
        calculate_metrics: bool = True,
        frame_work_1: bool = False,
        factor: list | None = None,
        only_outlet: bool = False,
    ):
        """Extract the simulated discharge hydrograph at gauge locations.

        Extracts discharge values from the total routed discharge array
        (``self.Qtot``) at each gauge location and stores them in
        ``self.Qsim``. Optionally applies a multiplication factor per
        gauge.

        Args:
            calculate_metrics (bool, optional): Whether to calculate
                performance metrics. Not used in this override but
                kept for signature compatibility. Default is True.
            frame_work_1 (bool, optional): True if the routing
                function is Maxbas. Not used in this override but
                kept for signature compatibility. Default is False.
            factor (list, optional): List of multiplication factors for
                the simulated discharge, one per gauge. If None, no
                scaling is applied. Default is None.
            only_outlet (bool, optional): True to extract discharge
                only at the outlet cell. Not used in this override but
                kept for signature compatibility. Default is False.
        """
        self.Qsim = np.zeros((self.TS - 1, len(self.GaugesTable)))
        # error = 0
        for i in range(len(self.GaugesTable)):
            Xind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_row"])
            Yind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_col"])
            # gaugeid = self.GaugesTable.loc[self.GaugesTable.index[i],"id"]

            # Quz = self.quz_routed[Xind,Yind,:-1]
            # Qlz = self.qlz_translated[Xind,Yind,:-1]
            # self.Qsim[:,i] = Quz + Qlz

            Qsim = np.reshape(self.Qtot[Xind, Yind, :-1], self.TS - 1)

            if factor is not None:
                self.Qsim[:, i] = Qsim * factor[i]
            else:
                self.Qsim[:, i] = Qsim

            # Qobs = Coello.QGauges.loc[:,gaugeid]
            # error = error + objective_function(Qobs, Qsim)

        # return error

    def run_calibration(self, SpatialVarFun, OptimizationArgs, printError=None):
        """Run the calibration algorithm for the distributed hydrological model.

        Executes the Harmony Search optimization algorithm to calibrate
        parameters for the conceptual distributed hydrological model.
        The method distributes parameters spatially using ``SpatialVarFun``,
        runs the RRM model via ``Wrapper.RRMModel``, and evaluates
        performance using the stored objective function.

        The following attributes must be set on the instance before calling
        this method:

            - ``Prec``, ``ET``, ``Temp``: Meteorological input arrays.
            - ``FlowDirArr``: Flow direction array.
            - ``rows``, ``cols``: Grid dimensions.
            - ``LB``, ``UB``: Lower and upper parameter bounds.
            - ``objective_function``: Objective function for evaluation.
            - ``QGauges``, ``GaugesTable``: Observed discharge data and
              gauge metadata.

        Args:
            SpatialVarFun: Spatial variable function object with a
                ``Function`` method that distributes parameters and a
                ``Par3d`` attribute holding the 3D parameter array, plus
                ``no_parameters`` and ``no_elem`` attributes.
            OptimizationArgs: A list of three elements:
                - ``OptimizationArgs[0]`` (dict): Harmony Search API
                  objective arguments (e.g., HMS, HMCR, PAR).
                - ``OptimizationArgs[1]``: Parallel type for the
                  optimizer.
                - ``OptimizationArgs[2]`` (dict): Solver arguments with
                  keys ``"store_sol"``, ``"display_opts"``,
                  ``"store_hst"``, and ``"hot_start"``.
            printError: If not 0, prints the error value and parameters
                at each iteration. Default is None.

        Returns:
            tuple: Optimization result tuple containing:
                - res[0]: The optimal objective function value.
                - res[1]: The optimal parameter set.

        Raises:
            AssertionError: If input dimensions are inconsistent or if
                optimization arguments are not dictionaries.
        """
        # input dimensions
        # [rows,cols] = self.FlowAcc.ReadAsArray().shape
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # basic inputs
        # check if all inputs are included
        # assert all(["p2","init_st","UB","LB","snow "][i] in Basic_inputs.keys() for i in range(4)), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert type(ApiObjArgs) is dict, "store_history should be 0 or 1"
        assert type(ApiSolveArgs) is dict, "history_fname should be of type string "

        print("Calibration starts")

        ### calculate the objective function
        def opt_fun(par):
            try:
                # distribute the parameters
                SpatialVarFun.Function(
                    par
                )  # , kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb
                self.Parameters = SpatialVarFun.Par3d
                # run the model
                Wrapper.RRMModel(self)
                # calculate performance of the model
                try:
                    error = self.objective_function(
                        self.QGauges, *[self.GaugesTable]
                    )  # self.qout, self.quz_routed, self.qlz_translated,
                    f = list(range(9, len(par), SpatialVarFun.no_parameters))
                    g = list()
                    for i in range(len(f)):
                        k = par[f[i]]
                        x = par[f[i] + 1]
                        g.append(2 * k * x / self.dt)
                        g.append((2 * k * (1 - x)) / self.dt)

                except TypeError:  # if no of inputs less than what the function needs
                    assert (
                        False
                    ), "the objective function you have entered needs more inputs please enter then in a list as *args"

                # print error
                if printError != 0:
                    print(round(error, 3))
                    print(par)

                fail = 0
            except:
                error = np.nan
                g = []
                fail = 1

            return error, g, fail

        ### define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)
        for i in range(len(self.LB)):
            opt_prob.addVar(
                "x{0}".format(i), type="c", lower=self.LB[i], upper=self.UB[i]
            )

        opt_prob.addObj("f")

        for i in range(SpatialVarFun.no_elem):
            opt_prob.addCon("g" + str(i) + "-1", "i")
            opt_prob.addCon("g" + str(i) + "-2", "i")

        print(opt_prob)

        opt_engine = HSapi(pll_type=pll_type, options=ApiObjArgs)

        store_sol = ApiSolveArgs["store_sol"]
        display_opts = ApiSolveArgs["display_opts"]
        store_hst = ApiSolveArgs["store_hst"]
        hot_start = ApiSolveArgs["hot_start"]

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.Parameters = res[1]
        self.OFvalue = res[0]

        return res

    def FW1Calibration(self, SpatialVarFun, OptimizationArgs, printError=None):
        """Run calibration using the FW1 (Focussed Width-1) routing scheme.

        Executes the Harmony Search optimization algorithm to calibrate
        parameters for the conceptual distributed hydrological model using
        the FW1 routing approach via ``Wrapper.FW1``.

        The following attributes must be set on the instance before calling
        this method:

            - ``Prec``, ``ET``, ``Temp``: Meteorological input arrays.
            - ``rows``, ``cols``: Grid dimensions.
            - ``LB``, ``UB``: Lower and upper parameter bounds.
            - ``objective_function``: Objective function for evaluation.
            - ``QGauges``, ``GaugesTable``: Observed discharge data and
              gauge metadata.

        Args:
            SpatialVarFun: Spatial variable function object with a
                ``Function`` method that distributes parameters and a
                ``Par3d`` attribute holding the 3D parameter array.
            OptimizationArgs: A list of three elements:
                - ``OptimizationArgs[0]`` (dict): Harmony Search API
                  objective arguments (e.g., HMS, HMCR, PAR).
                - ``OptimizationArgs[1]``: Parallel type for the
                  optimizer.
                - ``OptimizationArgs[2]`` (dict): Solver arguments with
                  keys ``"store_sol"``, ``"display_opts"``,
                  ``"store_hst"``, and ``"hot_start"``.
            printError: If not 0, prints the error value and parameters
                at each iteration. Default is None.

        Returns:
            tuple: Optimization result tuple containing:
                - res[0]: The optimal objective function value.
                - res[1]: The optimal parameter set.

        Raises:
            AssertionError: If input dimensions are inconsistent or if
                optimization arguments are not dictionaries.
        """
        # input dimensions
        # [rows,cols] = self.FlowAcc.ReadAsArray().shape
        # [fd_rows,fd_cols] = self.FlowDirArr.shape
        # assert fd_rows == self.rows and fd_cols == self.cols, "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # basic inputs
        # check if all inputs are included
        # assert all(["p2","init_st","UB","LB","snow "][i] in Basic_inputs.keys() for i in range(4)), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert type(ApiObjArgs) is dict, "store_history should be 0 or 1"
        assert type(ApiSolveArgs) is dict, "history_fname should be of type string "

        print("Calibration starts")

        # calculate the objective function
        def opt_fun(par):
            try:
                # distribute the parameters
                SpatialVarFun.Function(
                    par
                )  # , kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb, Maskingum=SpatialVarFun.Maskingum
                self.Parameters = SpatialVarFun.Par3d
                # run the model
                Wrapper.FW1(self)
                # calculate performance of the model
                try:
                    # error = self.objective_function(self.QGauges, self.qout, self.quz_routed, self.qlz_translated,*[self.GaugesTable])
                    error = self.objective_function(
                        self.QGauges, self.qout, *[self.GaugesTable]
                    )
                except TypeError:  # if no of inputs less than what the function needs
                    assert (
                        False
                    ), "the objective function you have entered needs more inputs please enter then in a list as *args"

                # print error
                if printError != 0:
                    print(round(error, 3))
                    print(par)

                fail = 0
            except:
                error = np.nan
                fail = 1

            return error, [], fail

        # define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)
        for i in range(len(self.LB)):
            opt_prob.addVar(
                "x{0}".format(i), type="c", lower=self.LB[i], upper=self.UB[i]
            )

        print(opt_prob)

        opt_engine = HSapi(pll_type=pll_type, options=ApiObjArgs)

        store_sol = ApiSolveArgs["store_sol"]
        display_opts = ApiSolveArgs["display_opts"]
        store_hst = ApiSolveArgs["store_hst"]
        hot_start = ApiSolveArgs["hot_start"]

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.Parameters = res[1]
        self.OFvalue = res[0]

        return res

    def lumpedCalibration(self, Basic_inputs, OptimizationArgs, printError=None):
        """Run the calibration algorithm for the lumped hydrological model.

        Executes the Harmony Search optimization algorithm to calibrate
        parameters for the lumped conceptual hydrological model. The
        method runs the model via ``Wrapper.Lumped`` and evaluates
        performance using the stored objective function. Muskingum
        routing constraints are enforced as inequality constraints.

        The following attributes must be set on the instance before calling
        this method:

            - ``LB``, ``UB``: Lower and upper parameter bounds.
            - ``objective_function``: Objective function for evaluation.
            - ``OFArgs``: Arguments for the objective function.
            - ``QGauges``: Observed discharge DataFrame.
            - ``dt``: Time step duration.

        Args:
            Basic_inputs (dict): Dictionary containing:
                - ``"Route"`` (int): Routing flag (1 to enable routing).
                - ``"RoutingFn"`` (callable): Routing function to use.
                - ``"InitialValues"`` (list, optional): Initial parameter
                  values for the optimizer. Defaults to an empty list if
                  not provided.
            OptimizationArgs: A list of three elements:
                - ``OptimizationArgs[0]`` (dict): Harmony Search API
                  objective arguments (e.g., HMS, HMCR, PAR).
                - ``OptimizationArgs[1]``: Parallel type for the
                  optimizer.
                - ``OptimizationArgs[2]`` (dict): Solver arguments with
                  keys ``"store_sol"``, ``"display_opts"``,
                  ``"store_hst"``, and ``"hot_start"``.
            printError: If not 0, prints the error value and constraint
                values at each iteration. Default is None.

        Returns:
            tuple: Optimization result tuple containing:
                - res[0]: The optimal objective function value.
                - res[1]: The optimal parameter set.

        Raises:
            AssertionError: If ``Basic_inputs`` is missing required keys
                ``"Route"`` or ``"RoutingFn"``, or if optimization
                arguments are not dictionaries.
        """
        # basic inputs
        # check if all inputs are included
        assert all(
            ["Route", "RoutingFn"][i] in Basic_inputs.keys() for i in range(2)
        ), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        Route = Basic_inputs["Route"]
        RoutingFn = Basic_inputs["RoutingFn"]
        if "InitialValues" in Basic_inputs.keys():
            InitialValues = Basic_inputs["InitialValues"]
        else:
            InitialValues = []

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert isinstance(ApiObjArgs, dict), "store_history should be 0 or 1"
        assert isinstance(ApiSolveArgs, dict), "history_fname should be of type string "

        # assert history_fname[-4:] == ".txt", "history_fname should be txt file please change extension or add .txt ad the end of the history_fname"

        print("Calibration starts")

        ### calculate the objective function
        def opt_fun(par):
            try:
                # parameters
                self.Parameters = par
                # run the model
                Wrapper.Lumped(self, Route, RoutingFn)
                # calculate performance of the model
                try:
                    error = self.objective_function(
                        self.QGauges[self.QGauges.columns[-1]], self.Qsim, *self.OFArgs
                    )
                    g = [
                        2 * par[-2] * par[-1] / self.dt,
                        (2 * par[-2] * (1 - par[-1])) / self.dt,
                    ]
                except TypeError:  # if no of inputs less than what the function needs
                    assert (
                        False
                    ), "the objective function you have entered needs more inputs please enter then in a list as *args"

                if printError != 0:
                    print(
                        f"Error = {round(error, 3)} Inequality Const = {np.round(g, 2)}"
                    )
                    # print(par)
                fail = 0
            except:
                error = np.nan
                g = []
                fail = 1
            return error, g, fail

        ### define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)

        if InitialValues != []:
            for i in range(len(self.LB)):
                opt_prob.addVar(
                    "x{0}".format(i),
                    type="c",
                    lower=self.LB[i],
                    upper=self.UB[i],
                    value=InitialValues[i],
                )
        else:
            for i in range(len(self.LB)):
                opt_prob.addVar(
                    "x{0}".format(i), type="c", lower=self.LB[i], upper=self.UB[i]
                )

        opt_prob.addObj("f")

        opt_prob.addCon("g1", "i")
        opt_prob.addCon("g2", "i")
        # print(opt_prob)
        opt_engine = HSapi(pll_type=pll_type, options=ApiObjArgs)

        # parse the ApiSolveArgs inputs
        # availablekeys = ['store_sol',"display_opts","store_hst","hot_start"]
        store_sol = ApiSolveArgs["store_sol"]
        display_opts = ApiSolveArgs["display_opts"]
        store_hst = ApiSolveArgs["store_hst"]
        hot_start = ApiSolveArgs["hot_start"]

        # for i in range(len(availablekeys)):
        # if availablekeys[i] in ApiSolveArgs.keys():
        # exec(availablekeys[i] + "=" + str(ApiSolveArgs[availablekeys[i]]))
        # print(availablekeys[i] + " = " + str(ApiSolveArgs[availablekeys[i]]))

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.OFvalue = res[0]
        self.Parameters = res[1]

        return res
