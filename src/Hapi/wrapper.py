"""Wrapper module for connecting rainfall-runoff model components.

This module provides the Wrapper class that connects the distributed
rainfall-runoff model execution with spatial routing schemes. It
supports multiple configurations including Muskingum routing,
triangular routing, and lake integration.
"""

import numpy as np

from Hapi.routing import Routing as routing
from Hapi.rrm.distrrm import DistributedRRM as distrrm
from Hapi.rrm.hbv_lake import HBVLake


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
    def RRMModel(Model, ll_temp=None, q_0=None):
        """Run the distributed rainfall-runoff model with spatial routing.

        Connects two modules:

        1. The distributed rainfall-runoff model that runs separately
           for each cell.
        2. The spatial routing scheme that routes flow following the
           river network.

        The method stores results directly on the Model object,
        including ``quz``, ``qlz``, ``qout``, ``quz_routed``, and
        ``qlz_translated`` arrays.

        Args:
            Model: Catchment model object containing:

                - DEM (gdal.dataset): DEM raster clipped to the
                  catchment.
                - flow_acc (gdal.dataset): Flow accumulation raster
                  clipped to the catchment.
                - flow_direct (gdal.dataset): Flow direction raster
                  clipped to the catchment.
                - sp_prec (numpy.ndarray): 3D precipitation array
                  with the same 2D dimensions as the raster input.
                - sp_et (numpy.ndarray): 3D evapotranspiration array
                  with the same 2D dimensions as the raster input.
                - sp_temp (numpy.ndarray): 3D temperature array with
                  the same 2D dimensions as the raster input.
                - sp_par (numpy.ndarray): 3D array of spatially
                  distributed catchment parameters.
                - p2 (list): Unoptimized parameters where p2[0] is
                  tfac (1 for hourly, 0.25 for 15 min, 24 for daily)
                  and p2[1] is catchment area in km2.
                - kub (float): Upper bound of K value for Muskingum
                  routing.
                - klb (float): Lower bound of K value for Muskingum
                  routing.
                - init_st (list): Initial state variable values
                  [sp, sm, uz, lz, wc].

            ll_temp (numpy.ndarray, optional): 3D array of long-term
                average temperature data. Defaults to None.
            q_0 (float, optional): Initial discharge in m3/s.
                Defaults to None.
        """
        # run the rainfall runoff model separately
        distrrm.run_lumped_model(Model)

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        # Model.qout = Model.qout[:-1]

    @staticmethod
    def RRMWithlake(Model, Lake, ll_temp=None, q_0=None):
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
        Lake.QlakeR = routing.Muskingum_V(
            Lake.Qlake,
            Lake.Qlake[0],
            Lake.Parameters[11],
            Lake.Parameters[12],
            Model.conversion_factor,
        )

        # subcatchment
        distrrm.run_lumped_model(Model)

        # routing lake discharge with DS cell k & x and adding to cell Q
        qlake = routing.Muskingum_V(
            Lake.QlakeR,
            Lake.QlakeR[0],
            Model.Parameters[Lake.OutflowCell[0], Lake.OutflowCell[1], 10],
            Model.Parameters[Lake.OutflowCell[0], Lake.OutflowCell[1], 11],
            Model.conversion_factor,
        )

        qlake = np.append(qlake, qlake[-1])
        # both lake & Quz are in m3/s
        Model.quz[Lake.OutflowCell[0], Lake.OutflowCell[1], :] = (
            Model.quz[Lake.OutflowCell[0], Lake.OutflowCell[1], :] + qlake
        )

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        # Model.qout = Model.qout[:-1]

    @staticmethod
    def FW1(Model, ll_temp=None, q_0=None):
        """Run the distributed RRM with triangular function-1 routing.

        Connects two modules:

        1. The distributed rainfall-runoff module.
        2. The triangular function-1 (MAXBAS) routing method.

        The output discharge is computed as the sum of routed upper
        zone and unrouted lower zone discharge across all cells.

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

        qlz1 = np.array(
            [np.nansum(Model.qlz[:, :, i]) for i in range(Model.TS)]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [np.nansum(Model.quz[:, :, i]) for i in range(Model.TS)]
        )  # average of all cells (routed mm/timestep)

        Model.qout = qlz1 + quz1

        Model.qout = Model.qout[:-1]

    @staticmethod
    def FW1Withlake(Model, Lake, ll_temp=None, q_0=None):
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
        Lake.QlakeR = routing.muskingum(
            Lake.Qlake,
            Lake.Qlake[0],
            Lake.Parameters[11],
            Lake.Parameters[12],
            Model.conversion_factor,
        )

        # subcatchment
        distrrm.run_lumped_model(Model)

        distrrm.DistMAXBAS(Model)

        qlz1 = np.array(
            [
                np.nansum(Model.qlz[:, :, i])
                for i in range(Model.Parameters.shape[2] + 1)
            ]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [
                np.nansum(Model.quz[:, :, i])
                for i in range(Model.Parameters.shape[2] + 1)
            ]
        )  # average of all cells (routed mm/timestep)

        qout = qlz1 + quz1

        # qout = (qlz1 + quz1) * Model.CatArea / (Model.conversion_factor* 3.6)

        Model.qout = qout[:-1] + Lake.QlakeR

    @staticmethod
    def Lumped(Model, Routing=0, RoutingFn=[]):
        """Run a lumped conceptual model with optional routing.

        Executes a lumped rainfall-runoff model (e.g., HBV) to
        compute the upper and lower zone discharge, then optionally
        routes the combined discharge using the provided routing
        function.

        The discharge is converted from mm/timestep to m3/s using
        the catchment area and conversion factor. Results are stored
        on the Model object as ``quz``, ``qlz``, ``Qsim``, and
        ``state_variables``.

        Args:
            Model: Lumped model object containing:

                - data (numpy.ndarray): 2D meteorological data array
                  with columns for precipitation,
                  evapotranspiration, temperature, and long-term
                  average temperature.
                - Parameters (numpy.ndarray): Conceptual model
                  parameters.
                - LumpedModel: Conceptual model instance with a
                  ``simulate`` method.
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
            AssertionError: If ``RoutingFn`` is not callable when
                routing is enabled.
        """
        ### input data validation
        assert callable(
            RoutingFn
        ), "routing function should be of type callable (function that takes arguments)"

        # data
        p = Model.data[:, 0]
        et = Model.data[:, 1]
        t = Model.data[:, 2]
        tm = Model.data[:, 3]

        # from the conceptual model calculate the upper and lower response mm/time step
        Model.quz, Model.qlz, Model.state_variables = Model.LumpedModel.simulate(
            p,
            t,
            et,
            tm,
            Model.Parameters,
            init_st=Model.InitialCond,
            q_init=Model.q_init,
            snow=Model.Snow,
        )
        # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
        # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
        Model.quz = Model.quz * Model.CatArea / Model.conversion_factor
        Model.qlz = Model.qlz * Model.CatArea / Model.conversion_factor

        Model.Qsim = Model.quz + Model.qlz

        if Routing != 0 and Model.Maxbas:
            Model.Qsim = RoutingFn(np.array(Model.Qsim[:-1]), Model.Parameters[-1])
        elif Routing != 0:
            Model.Qsim = RoutingFn(
                np.array(Model.Qsim[:-1]),
                Model.Qsim[0],
                Model.Parameters[-2],
                Model.Parameters[-1],
                Model.dt,
            )


if __name__ == "__main__":
    print("Wrapper")
