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

from hapi.routing import Routing as routing
from hapi.rrm.distrrm import DistributedRRM as distrrm
from hapi.rrm.hbv_lake import HBVLake

if TYPE_CHECKING:
    from hapi.catchment import Catchment, Lake


class Wrapper:
    """Connects rainfall-runoff model components with spatial routing.

    The Wrapper class connects different components together including
    the lumped run of the distributed model with the spatial routing
    for Hapi and for FW1 (triangular routing).

    Methods:
        RRMModel: Run distributed RRM with Muskingum spatial routing.
        RRMWithlake: Run distributed RRM with lake and Muskingum
            spatial routing.
        FW1: Run distributed RRM with triangular routing.
        FW1Withlake: Run distributed RRM with lake and triangular
            routing.
        Lumped: Run a lumped conceptual model with optional routing.
    """

    def __init__(self):
        """Initialize the Wrapper class."""
        pass

    @staticmethod
    def RRMModel(Model: Catchment, ll_temp=None, q_0=None):
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
        """
        # run the rainfall runoff model separately
        distrrm.run_lumped_model(Model)

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        # Muskingum accumulates downstream, so a cell of `Qtot` is the discharge at that
        # cell and the outlet-cell shortcut in `extract_discharge` is valid again. Clear
        # the flag a previous MAXBAS run on this same model may have left set.
        Model._maxbas_routed = False

        # Model.qout = Model.qout[:-1]

    @staticmethod
    def RRMWithlake(Model: Catchment, Lake: Lake, ll_temp=None, q_0=None):
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
        plake = Lake.MeteoData[:, 0]
        et = Lake.MeteoData[:, 1]
        t = Lake.MeteoData[:, 2]
        tm = Lake.MeteoData[:, 3]

        # lake simulation
        Lake.Qlake, _ = HBVLake().simulate(
            plake,
            t,
            et,
            Lake.Parameters,
            [Model.conversion_factor, Lake.CatArea, Lake.LakeArea],
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
            Lake.Parameters[11],
            Lake.Parameters[12],
            Model.conversion_factor,
        )

        # subcatchment
        distrrm.run_lumped_model(Model)

        # routing lake discharge with DS cell k & x and adding to cell Q
        qlake = routing.muskingum_v(
            Lake.QlakeR,
            Lake.QlakeR[0],
            Model.parameters[Lake.OutflowCell[0], Lake.OutflowCell[1], 10],
            Model.parameters[Lake.OutflowCell[0], Lake.OutflowCell[1], 11],
            Model.conversion_factor,
        )

        # No padding: `HBVLake.simulate` already prepends the initial-state slot, exactly as
        # the distributed model does, and `muskingum_v` preserves length -- so `qlake` is
        # already `simulation_steps` long and lines up with `quz` slot for slot. Appending a
        # step here made it one longer than the array it is added to, which raised for every
        # input and left this entry point unrunnable.
        # both lake & Quz are in m3/s
        Model.quz[Lake.OutflowCell[0], Lake.OutflowCell[1], :] = (
            Model.quz[Lake.OutflowCell[0], Lake.OutflowCell[1], :] + qlake
        )

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        # Muskingum accumulates downstream, so a cell of `Qtot` is the discharge at that
        # cell and the outlet-cell shortcut in `extract_discharge` is valid again. Clear
        # the flag a previous MAXBAS run on this same model may have left set.
        Model._maxbas_routed = False

        # Model.qout = Model.qout[:-1]

    @staticmethod
    def _set_maxbas_output_fields(Model: Catchment):
        """Fill the distributed output fields after a triangular (MAXBAS) run.

        `save_results` and `plot_distributed_results` read `Qtot`,
        `quz_routed` and `qlz_translated` for their discharge options. Only
        :meth:`DistRRM.SpatialRouting` (the Muskingum path) used to set them, so
        after a MAXBAS run they stayed `None` and every discharge option raised
        `TypeError: 'NoneType' object is not subscriptable`.

        MAXBAS routes each cell's upper zone straight to the outlet with that
        cell's own `maxbas`, in place, and applies no cell-to-cell translation
        to the lower zone. So the routed/translated fields *are* the per-cell
        arrays, and their sum is the per-cell contribution to the outlet
        hydrograph — `np.nansum(Qtot[:, :, i])` reproduces `qout[i]`. That
        differs from the Muskingum path, where the fields accumulate downstream
        and `Qtot` at the outlet cell *is* the outlet discharge.

        `quz_routed` / `qlz_translated` alias `quz` / `qlz` rather than
        copying them: they hold the same data, and a copy would double the memory
        of a `(rows, cols, time_steps)` array for no gain. They are outputs, so
        nothing downstream writes through the alias.

        Args:
            Model: Catchment whose `quz` / `qlz` have been routed by
                :meth:`DistRRM.DistMaxbas1`.
        """
        Model.quz_routed = Model.quz
        Model.qlz_translated = Model.qlz
        Model.Qtot = Model.qlz + Model.quz
        # Flags the outlet-cell shortcut in `extract_discharge` as invalid here.
        Model._maxbas_routed = True

    @staticmethod
    def FW1(Model: Catchment, ll_temp=None, q_0=None):
        """Run the distributed RRM with triangular function-1 routing.

        Connects two modules:

        1. The distributed rainfall-runoff module.
        2. The triangular function-1 (MAXBAS) routing method.

        The output discharge is computed as the sum of routed upper
        zone and unrouted lower zone discharge across all cells.

        Also fills the per-cell output fields (`Qtot`, `quz_routed`,
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
        distrrm.run_lumped_model(Model)

        distrrm.DistMaxbas1(Model)

        Wrapper._set_maxbas_output_fields(Model)

        qlz1 = np.array(
            [np.nansum(Model.qlz[:, :, i]) for i in range(Model.meteo.simulation_steps)]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [np.nansum(Model.quz[:, :, i]) for i in range(Model.meteo.simulation_steps)]
        )  # average of all cells (routed mm/timestep)

        Model.qout = qlz1 + quz1

        Model.qout = Model.qout[:-1]

    @staticmethod
    def FW1Withlake(Model: Catchment, Lake: Lake, ll_temp=None, q_0=None):
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
        plake = Lake.MeteoData[:, 0]
        et = Lake.MeteoData[:, 1]
        t = Lake.MeteoData[:, 2]
        tm = Lake.MeteoData[:, 3]

        # lake simulation
        Lake.Qlake, _ = HBVLake().simulate(
            plake,
            t,
            et,
            Lake.Parameters,
            [Model.conversion_factor, Lake.CatArea, Lake.LakeArea],
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
            Lake.Parameters[11],
            Lake.Parameters[12],
            Model.conversion_factor,
        )

        # subcatchment
        distrrm.run_lumped_model(Model)

        distrrm.DistMaxbas1(Model)

        # Subcatchment fields only: the lake is a lumped inflow with no spatial
        # extent, so it enters `qout` below but never `Qtot`.
        Wrapper._set_maxbas_output_fields(Model)

        qlz1 = np.array(
            [np.nansum(Model.qlz[:, :, i]) for i in range(Model.meteo.simulation_steps)]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [np.nansum(Model.quz[:, :, i]) for i in range(Model.meteo.simulation_steps)]
        )  # average of all cells (routed mm/timestep)

        qout = qlz1 + quz1

        # qout = (qlz1 + quz1) * Model.CatArea / (Model.conversion_factor* 3.6)

        # Both series run over `simulation_steps`, and the non-lake FW1 path returns
        # `qout[:-1]` -- dropping the trailing slot, not the leading initial-state one. The
        # lake series has to be trimmed the same way or the two cannot be added at all.
        Model.qout = qout[:-1] + Lake.QlakeR[:-1]

    @staticmethod
    def Lumped(Model: Catchment, Routing: int = 0, RoutingFn: Callable | None = None):
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
            if not callable(RoutingFn):
                raise TypeError(
                    "routing function should be of type callable (function that takes "
                    f"arguments), got {type(RoutingFn).__name__}"
                )

        # data
        p = Model.data[:, 0]
        et = Model.data[:, 1]
        t = Model.data[:, 2]
        tm = Model.data[:, 3]

        # from the conceptual model calculate the upper and lower response mm/time step
        Model.quz, Model.qlz, Model.state_variables = Model.lumped_model.simulate(
            p,
            t,
            et,
            tm,
            Model.parameters,
            init_st=Model.initial_cond,
            q_init=Model.q_init,
            snow=Model.snow,
        )
        # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
        # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
        Model.quz = Model.quz * Model.area / Model.conversion_factor
        Model.qlz = Model.qlz * Model.area / Model.conversion_factor

        Model.Qsim = Model.quz + Model.qlz

        if Routing != 0 and Model.maxbas:
            Model.Qsim = RoutingFn(np.array(Model.Qsim[:-1]), Model.parameters[-1])
        elif Routing != 0:
            Model.Qsim = RoutingFn(
                np.array(Model.Qsim[:-1]),
                Model.Qsim[0],
                Model.parameters[-2],
                Model.parameters[-1],
                Model.dt,
            )


if __name__ == "__main__":
    print("Wrapper")
