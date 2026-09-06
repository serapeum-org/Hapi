"""Wrapper module for connecting rainfall-runoff model components.

This module provides the Wrapper class that connects the distributed
rainfall-runoff model execution with spatial routing schemes. It
supports multiple configurations including Muskingum routing,
triangular routing, and lake integration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from hapi.results import RoutingKind, SimulationResults
from hapi.routing import Routing as routing
from hapi.rrm.distrrm import DistributedRRM as distrrm
from hapi.rrm.hbv_lake import HBVLake
from hapi.runs import DistributedRun, LumpedRun

if TYPE_CHECKING:
    from hapi.catchment import Lake


def _lake_inputs(lake: Lake) -> tuple[np.ndarray, list, list]:
    """Fetch the two lake inputs the wrapper indexes, or say which reader supplies them.

    `Lake` is a builder like `Catchment`, so both are `X | None` until read. The wrapper used
    to index them straight, so a lake missing either failed on `None` several frames in.

    Args:
        lake: The lake about to be simulated.

    Returns:
        tuple[np.ndarray, list, list]: The meteorological record, the parameter vector and the
        outflow cell.

    Raises:
        ValueError: Any of the three is unset.
    """
    if lake.MeteoData is None:
        raise ValueError(
            "the lake has no meteorological data; call lake.read_meteo_data first"
        )
    if lake.Parameters is None:
        raise ValueError("the lake has no parameters; call lake.read_parameters first")
    if lake.OutflowCell is None:
        raise ValueError(
            "the lake has no outflow cell; pass outflow_cell to lake.read_lumped_model"
        )
    return lake.MeteoData, lake.Parameters, lake.OutflowCell


class Wrapper:
    """Connects rainfall-runoff model components with spatial routing.

    The Wrapper class connects different components together including
    the lumped run of the distributed model with the spatial routing
    for Hapi and for FW1 (triangular routing).

    Methods:
        run_muskingum: Run distributed RRM with Muskingum spatial routing.
        run_muskingum_with_lake: Run distributed RRM with lake and Muskingum
            spatial routing.
        FW1: Run distributed RRM with triangular routing.
        run_maxbas_with_lake: Run distributed RRM with lake and triangular
            routing.
        Lumped: Run a lumped conceptual model with optional routing.
    """

    @staticmethod
    def run_muskingum(run: DistributedRun) -> SimulationResults:
        """Run the distributed rainfall-runoff model with spatial routing.

        Connects two modules:

        1. The distributed rainfall-runoff model that runs separately
           for each cell.
        2. The spatial routing scheme that routes flow following the
           river network.

        The method stores results directly on the Model object,
        including `quz`, `qlz`, `qout`, `quz_routed`, and
        `qlz_translated` arrays.

        Args:
            Model: Catchment model object containing:

                - meteo (:class:`~hapi.inputs.MeteoInputs`): The three driver cubes,
                  each `(rows, cols, time)`, plus the calendar they cover.
                - flow_network (:class:`~hapi.inputs.FlowNetwork`): The flow accumulation
                  and direction arrays, the direction table, and the grid they define.
                - parameters (numpy.ndarray): 3D array of spatially distributed catchment
                  parameters, `(rows, cols, n_parameters)`.
                - conversion_factor (float): Depth-to-discharge factor for the temporal
                  resolution; 24 for daily, 1 for hourly.
                - area (float): Catchment area in km2.
                - initial_cond (list): Initial state variable values
                  [sp, sm, uz, lz, wc].
                - snow (int): 1 to run the snow routine, 0 otherwise.

            ll_temp (numpy.ndarray, optional): 3D array of long-term
                average temperature data. Defaults to None.
            q_0 (float, optional): Initial discharge in m3/s.
                Defaults to None.
            skip_hydraulic_cells (bool, optional): Leave cells with a positive
                `bankfull_depth` unrouted, because a 1D hydraulic model routes them.
                The flood model's path. Defaults to False.

        Returns:
            SimulationResults: The run's output. Nothing is written to the caller's model;
            the entry point in :mod:`hapi.run` is what puts it on `model.results`.
        """
        # run the rainfall runoff model separately
        results = distrrm.run_lumped_model(run)

        # run the GIS part to rout from cell to another. It records
        # `RoutingKind.MUSKINGUM` on the results, which is what makes the outlet-cell
        # shortcut in `extract_discharge` valid for them.
        distrrm.route_muskingum(run, results)
        return results

    @staticmethod
    def run_muskingum_with_lake(run: DistributedRun, Lake: Lake) -> SimulationResults:
        """Run the distributed RRM with lake simulation and routing.

        Connects three modules: the lake module, the distributed
        rainfall-runoff module, and the spatial routing module. The
        lake discharge is simulated using HBVLake, routed via
        Muskingum, and added to the downstream cell before spatial
        routing.

        Args:
            Model: Catchment model object containing the distributed
                model configuration, parameters, and spatial data.
            Lake: Lake object containing:

                - MeteoData (numpy.ndarray): 2D array with columns
                  for precipitation, evapotranspiration, temperature,
                  and long-term average temperature.
                - Parameters (numpy.ndarray): Lake model parameters.
                - CatArea (float): Lake catchment area in km2.
                - LakeArea (float): Lake surface area in km2.
                - StageDischargeCurve (numpy.ndarray): Stage-discharge
                  relationship.
                - InitialCond (list): Initial condition values.
                - OutflowCell (tuple): Row and column indices of the
                  lake outflow cell.

            ll_temp (numpy.ndarray, optional): 3D array of long-term
                average temperature data. Defaults to None.
            q_0 (float, optional): Initial discharge in m3/s.
                Defaults to None.
        """
        meteo_data, lake_parameters, outflow_cell = _lake_inputs(Lake)
        plake = meteo_data[:, 0]
        et = meteo_data[:, 1]
        t = meteo_data[:, 2]
        tm = meteo_data[:, 3]

        # lake simulation
        Lake.Qlake, _ = HBVLake().simulate(
            plake,
            t,
            et,
            lake_parameters,
            [run.period.conversion_factor, Lake.CatArea, Lake.LakeArea],
            Lake.StageDischargeCurve,
            0,
            init_st=Lake.InitialCond,
            ll_temp=tm,
            lake_sim=True,
        )
        # qlake is in m3/sec
        # lake routing
        Lake.QlakeR = routing.muskingum_v(
            Lake.Qlake,
            Lake.Qlake[0],
            lake_parameters[11],
            lake_parameters[12],
            run.period.conversion_factor,
        )

        # subcatchment
        results = distrrm.run_lumped_model(run)

        # `ParameterSet.values` is a flat sequence for a lumped run and a cube for a
        # distributed one; this path is distributed, so index it as the cube it is.
        parameters = np.asarray(run.parameters.values)
        # routing lake discharge with DS cell k & x and adding to cell Q
        qlake = routing.muskingum_v(
            Lake.QlakeR,
            Lake.QlakeR[0],
            parameters[outflow_cell[0], outflow_cell[1], 10],
            parameters[outflow_cell[0], outflow_cell[1], 11],
            run.period.conversion_factor,
        )

        # No padding: `HBVLake.simulate` already prepends the initial-state slot, exactly as
        # the distributed model does, and `muskingum_v` preserves length -- so `qlake` is
        # already `simulation_steps` long and lines up with `quz` slot for slot. Appending a
        # step here made it one longer than the array it is added to, which raised for every
        # input and left this entry point unrunnable.
        # both lake & Quz are in m3/s
        quz = results.quz
        quz[outflow_cell[0], outflow_cell[1], :] = (
            quz[outflow_cell[0], outflow_cell[1], :] + qlake
        )

        # run the GIS part to rout from cell to another. It records
        # `RoutingKind.MUSKINGUM` on the results.
        distrrm.route_muskingum(run, results)
        return results

    @staticmethod
    def _set_maxbas_output_fields(results: SimulationResults) -> None:
        """Fill the distributed output fields after a triangular (MAXBAS) run.

        `save_results` and `plot_distributed_results` read `q_total`,
        `quz_routed` and `qlz_translated` for their discharge options. Only
        :meth:`DistRRM.route_muskingum` (the Muskingum path) used to set them, so
        after a MAXBAS run they stayed `None` and every discharge option raised
        `TypeError: 'NoneType' object is not subscriptable`.

        MAXBAS routes each cell's upper zone straight to the outlet with that
        cell's own `maxbas`, in place, and applies no cell-to-cell translation
        to the lower zone. So the routed/translated fields *are* the per-cell
        arrays, and their sum is the per-cell contribution to the outlet
        hydrograph — `np.nansum(q_total[:, :, i])` reproduces `qout[i]`. That
        differs from the Muskingum path, where the fields accumulate downstream
        and `q_total` at the outlet cell *is* the outlet discharge.

        `quz_routed` / `qlz_translated` alias `quz` / `qlz` rather than
        copying them: they hold the same data, and a copy would double the memory
        of a `(rows, cols, time_steps)` array for no gain. They are outputs, so
        nothing downstream writes through the alias.

        Args:
            Model: Catchment whose `quz` / `qlz` have been routed by
                :meth:`DistRRM.route_maxbas`.
        """
        results.quz_routed = results.quz
        results.qlz_translated = results.qlz
        results.q_total = results.qlz + results.quz
        # Marks the outlet-cell shortcut in `extract_discharge` as invalid for these
        # results, via `SimulationResults.outlet_shortcut_valid`.
        results.routing = RoutingKind.MAXBAS

    @staticmethod
    def run_maxbas(run: DistributedRun) -> SimulationResults:
        """Run the distributed RRM with triangular function-1 routing.

        Connects two modules:

        1. The distributed rainfall-runoff module.
        2. The triangular function-1 (MAXBAS) routing method.

        The output discharge is computed as the sum of routed upper
        zone and unrouted lower zone discharge across all cells.

        Also fills the per-cell output fields (`q_total`, `quz_routed`,
        `qlz_translated`) via :meth:`_set_maxbas_output_fields`, so the
        discharge options of `save_results` / `plot_distributed_results`
        work on this path; see that method for the MAXBAS semantics.

        Args:
            Model: Catchment model object containing the distributed
                model configuration, parameters, and spatial data.
            ll_temp (numpy.ndarray, optional): 3D array of long-term
                average temperature data. Defaults to None.
            q_0 (float, optional): Initial discharge in m3/s.
                Defaults to None.
        """
        # subcatchment
        results = distrrm.run_lumped_model(run)

        distrrm.route_maxbas(run, results)

        Wrapper._set_maxbas_output_fields(results)

        steps = run.meteo.simulation_steps
        qlz1 = np.array(
            [np.nansum(results.qlz[:, :, i]) for i in range(steps)]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [np.nansum(results.quz[:, :, i]) for i in range(steps)]
        )  # average of all cells (routed mm/timestep)

        results.qout = (qlz1 + quz1)[:-1]
        return results

    @staticmethod
    def run_maxbas_with_lake(run: DistributedRun, Lake: Lake) -> SimulationResults:
        """Run the distributed RRM with lake and triangular routing.

        Connects three modules:

        1. The distributed rainfall-runoff module.
        2. The triangular function-1 (MAXBAS) routing method.
        3. The lake simulation module.

        The lake discharge is simulated using HBVLake, routed via
        Muskingum, and combined with the subcatchment discharge that
        has been routed using the triangular function.

        Args:
            Model: Catchment model object containing the distributed
                model configuration, parameters, and spatial data.
            Lake: Lake object containing:

                - MeteoData (numpy.ndarray): 2D array with columns
                  for precipitation, evapotranspiration, temperature,
                  and long-term average temperature.
                - Parameters (numpy.ndarray): Lake model parameters.
                - CatArea (float): Lake catchment area in km2.
                - LakeArea (float): Lake surface area in km2.
                - StageDischargeCurve (numpy.ndarray): Stage-discharge
                  relationship.
                - InitialCond (list): Initial condition values.

            ll_temp (numpy.ndarray, optional): 3D array of long-term
                average temperature data. Defaults to None.
            q_0 (float, optional): Initial discharge in m3/s.
                Defaults to None.
        """
        meteo_data, lake_parameters, outflow_cell = _lake_inputs(Lake)
        plake = meteo_data[:, 0]
        et = meteo_data[:, 1]
        t = meteo_data[:, 2]
        tm = meteo_data[:, 3]

        # lake simulation
        Lake.Qlake, _ = HBVLake().simulate(
            plake,
            t,
            et,
            lake_parameters,
            [run.period.conversion_factor, Lake.CatArea, Lake.LakeArea],
            Lake.StageDischargeCurve,
            0,
            init_st=Lake.InitialCond,
            ll_temp=tm,
            lake_sim=True,
        )

        # qlake is in m3/sec
        # lake routing
        Lake.QlakeR = routing.muskingum_v(
            Lake.Qlake,
            Lake.Qlake[0],
            lake_parameters[11],
            lake_parameters[12],
            run.period.conversion_factor,
        )

        # subcatchment
        results = distrrm.run_lumped_model(run)

        distrrm.route_maxbas(run, results)

        # Subcatchment fields only: the lake is a lumped inflow with no spatial
        # extent, so it enters `qout` below but never `q_total`.
        Wrapper._set_maxbas_output_fields(results)

        steps = run.meteo.simulation_steps
        qlz1 = np.array(
            [np.nansum(results.qlz[:, :, i]) for i in range(steps)]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [np.nansum(results.quz[:, :, i]) for i in range(steps)]
        )  # average of all cells (routed mm/timestep)

        qout = qlz1 + quz1

        # qout = (qlz1 + quz1) * area / (run.period.conversion_factor * 3.6)

        # Both series run over `simulation_steps`, and the non-lake FW1 path returns
        # `qout[:-1]` -- dropping the trailing slot, not the leading initial-state one. The
        # lake series has to be trimmed the same way or the two cannot be added at all.
        results.qout = qout[:-1] + Lake.QlakeR[:-1]
        return results

    @staticmethod
    def run_lumped(
        run: LumpedRun, Routing: int = 0, RoutingFn: Callable | None = None
    ) -> SimulationResults:
        """Run a lumped conceptual model with optional routing.

        Executes a lumped rainfall-runoff model (e.g., HBV) to
        compute the upper and lower zone discharge, then optionally
        routes the combined discharge using the provided routing
        function.

        The discharge is converted from mm/timestep to m3/s using
        the catchment area and conversion factor. Results are stored
        on the Model object as `quz`, `qlz`, `Qsim`, and
        `state_variables`.

        Args:
            Model: Lumped model object containing:

                - data (numpy.ndarray): 2D meteorological data array
                  with columns for precipitation,
                  evapotranspiration, temperature, and long-term
                  average temperature.
                - Parameters (numpy.ndarray): Conceptual model
                  parameters.
                - LumpedModel: Conceptual model instance with a
                  `simulate` method.
                - InitialCond (list): Initial state variable values
                  [sp, sm, uz, lz, wc].
                - q_init (float): Initial discharge value.
                - Snow (int): Flag to include snow module (0 or 1).
                - CatArea (float): Catchment area in km2.
                - conversion_factor (float): Time step conversion
                  factor (1 for hourly, 0.25 for 15 min, 24 for
                  daily).
                - Maxbas (bool): Whether to use MAXBAS triangular
                  routing.
                - dt (float): Time step duration.

            Routing (int, optional): Flag to enable routing. Set to
                0 to disable, nonzero to enable. Defaults to 0.
            RoutingFn (callable): Routing function to apply to the
                discharge hydrograph. Must be callable.

        Raises:
            TypeError: If `RoutingFn` is not callable when
                routing is enabled.
        """
        ### input data validation
        if Routing != 0:
            if RoutingFn is None or not callable(RoutingFn):
                raise TypeError(
                    "routing function should be of type callable (function that takes "
                    f"arguments), got {type(RoutingFn).__name__}"
                )

        # data
        p = run.data[:, 0]
        et = run.data[:, 1]
        t = run.data[:, 2]
        tm = run.data[:, 3]

        # from the conceptual model calculate the upper and lower response mm/time step
        quz, qlz, state_variables = run.model_setup.model.simulate(
            p,
            t,
            et,
            tm,
            run.parameters.values,
            init_st=run.model_setup.initial_cond,
            q_init=run.model_setup.q_init,
            snow=run.parameters.snow,
        )
        # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
        # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
        factor = run.model_setup.area / run.period.conversion_factor
        # A lumped run has no spatial routing at all, so the routed fields stay None and
        # the routing kind says why -- rather than a MAXBAS flag left over from elsewhere.
        results = SimulationResults(
            routing=RoutingKind.LUMPED,
            quz=quz * factor,
            qlz=qlz * factor,
            state_variables=state_variables,
        )
        # The lumped total discharge is exactly what `q_total` means, so it goes there rather
        # than onto the catchment as `Qsim`. `Run.run_lumped` is what indexes it by the period
        # and puts the frame on the model -- so this engine writes nothing outside `results`.
        q_total = results.quz + results.qlz

        if Routing != 0 and run.parameters.maxbas:
            route = RoutingFn
            assert route is not None  # noqa: S101 - guarded above
            q_total = route(np.array(q_total[:-1]), run.parameters.values[-1])
        elif Routing != 0:
            route = RoutingFn
            assert route is not None  # noqa: S101 - guarded above
            q_total = route(
                np.array(q_total[:-1]),
                q_total[0],
                run.parameters.values[-2],
                run.parameters.values[-1],
                run.period.dt,
            )
        results.q_total = q_total
        return results


if __name__ == "__main__":
    print("Wrapper")
