"""Run module for the Hapi hydrological model.

The run module connects the parameter spatial distribution function with
both components of the spatial representation of the hydrological process
(conceptual model and spatial routing) to calculate the predicted runoff
at known locations based on a given performance function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from Hapi.catchment import Catchment

# from Hapi.hm.saintvenant import SaintVenant
from Hapi.wrapper import Wrapper


class Run(Catchment):
    """Run the catchment model.

    The Run sub-class validates the spatial data and hands it to the
    Wrapper class. It is a sub-class of the Catchment class, so you
    need to create the Catchment object first to run the model.

    Methods:
        RunHapi: Run the distributed hydrological model.
        runHAPIwithLake: Run the distributed model with a lake component.
        runFW1: Run the FW1 distributed model.
        RunFW1withLake: Run the FW1 model with a lake component.
        runLumped: Run the lumped conceptual model.
    """

    def __init__(self):
        """Initialize the Run class."""
        self.Qsim: np.ndarray | pd.DataFrame | None = None

    def RunHapi(self):
        """Run the distributed hydrological model.

        Validates that all input arrays (precipitation, evapotranspiration,
        temperature, parameters, and flow direction) have consistent
        dimensions, then executes the rainfall-runoff model via the
        Wrapper.

        The following instance attributes are set after execution:

        - ``state_variables``: 4D array (rows, cols, time, states) where
          states are [sp, wc, sm, uz, lv].
        - ``qlz``: 3D array of the lower zone discharge.
        - ``quz``: 3D array of the upper zone discharge.
        - ``qout``: 1D timeseries of discharge at the catchment outlet
          in m3/sec.
        - ``quz_routed``: 3D array of the upper zone discharge
          accumulated and routed at each time step.
        - ``qlz_translated``: 3D array of the lower zone discharge
          translated at each time step.

        Raises:
            AssertionError: If input data arrays have inconsistent
                row counts, column counts, or temporal lengths.
        """
        # input dimensions
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # run the model
        Wrapper.RRMModel(self)

        print("Model Run has finished")

    def RunFloodModel(self):
        """Run the flood model.

        Runs the conceptual distributed hydrological model with
        additional validation for river geometry inputs (bankfull depth,
        river width, river roughness, and flood plain roughness).

        Raises:
            AssertionError: If meteorological input arrays, parameter
                arrays, or river geometry arrays have inconsistent
                dimensions.
        """
        # input dimensions
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        assert (
            np.shape(self.BankfullDepth)[0] == self.rows
            and np.shape(self.RiverWidth)[0] == self.rows
            and np.shape(self.RiverRoughness)[0] == self.rows
            and np.shape(self.FloodPlainRoughness)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.BankfullDepth)[1] == self.cols
            and np.shape(self.RiverWidth)[1] == self.cols
            and np.shape(self.RiverRoughness)[1] == self.cols
            and np.shape(self.FloodPlainRoughness)[1] == self.cols
        ), "all input data should have the same number of columns"

        # run the model
        Wrapper.RRMModel(self)
        print("RRM has finished")
        # SV = SaintVenant()
        # SV.KinematicRaster(self)
        # print("1D model Run has finished")

    def runHAPIwithLake(self, Lake):
        """Run the distributed model with a lake component.

        Validates that all input arrays have consistent dimensions and
        that the lake meteorological data matches the simulation period,
        then executes the rainfall-runoff model with lake routing via
        the Wrapper.

        Args:
            Lake: Lake object containing lake configuration and
                meteorological data. Must have a ``MeteoData`` attribute
                with shape ``(time_steps, >= 3)`` where columns are
                rain, ET, and temperature.

        Raises:
            AssertionError: If input data arrays have inconsistent
                dimensions or if the lake meteorological data length
                does not match the distributed raster data length.
        """
        # input dimensions
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows and columns"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        assert (
            np.shape(Lake.MeteoData)[0] == np.shape(self.Prec)[2]
        ), "Lake meteorological data has to have the same length as the distributed raster data"
        assert (
            np.shape(Lake.MeteoData)[1] >= 3
        ), "Lake Meteo data has to have at least three columns of rain, ET, and Temp"

        # run the model
        Wrapper.RRMWithlake(self, Lake)

        print("Model Run has finished")

    def runFW1(self):
        """Run the FW1 distributed hydrological model.

        Validates that all input arrays have consistent dimensions,
        then executes the FW1 model via the Wrapper.

        The following instance attributes are set after execution:

        - ``st``: 4D array of state variables.
        - ``q_out``: 1D array of calculated discharge at the catchment
          outlet.
        - ``q_uz``: 3D array of distributed discharge for each cell.

        Raises:
            AssertionError: If input data arrays have inconsistent
                row counts, column counts, or temporal lengths.
        """
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # run the model
        Wrapper.FW1(self)

        print("Model Run has finished")

    def RunFW1withLake(self, Lake):
        """Run the FW1 distributed model with a lake component.

        Validates that all input arrays have consistent dimensions and
        that the lake meteorological data matches the simulation period,
        then executes the FW1 model with lake routing via the Wrapper.

        Args:
            Lake: Lake object containing lake configuration and
                meteorological data. Must have a ``MeteoData`` attribute
                with shape ``(time_steps, >= 3)`` where columns are
                rain, ET, and temperature.

        Note:
            The following catchment attributes should be set before
            calling this method:

            - ``prec_path``: Path to the folder containing precipitation
              rasters.
            - ``evap_path``: Path to the folder containing
              evapotranspiration rasters.
            - ``temp_path``: Path to the folder containing temperature
              rasters.
            - ``flow_acc_path``: Path to the flow accumulation raster.
            - ``flow_direction_path``: Path to the flow direction raster.
            - ``ParPath``: Path to the folder containing parameter
              rasters.
            - ``p2``: List of unoptimized parameters where ``p2[0]``
              is tfac and ``p2[1]`` is catchment area in km2.

        Raises:
            AssertionError: If input data arrays have inconsistent
                dimensions or if the lake meteorological data length
                does not match the distributed raster data length.
        """
        # input data validation

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        assert (
            np.shape(Lake.MeteoData)[0] == np.shape(self.Prec)[2]
        ), "Lake meteorological data has to have the same length as the distributed raster data"
        assert (
            np.shape(Lake.MeteoData)[1] >= 3
        ), "Lake Meteo data has to have at least three columns rain, ET, and Temp"

        # run the model
        Wrapper.FW1Withlake(self, Lake)

    def runLumped(
        self,
        Route: int = 0,
        RoutingFn=None,
    ):
        """Run the lumped conceptual model.

        Executes a lumped conceptual hydrological model, optionally
        routing the generated discharge hydrograph. The simulated
        discharge is stored in ``self.Qsim`` as a pandas DataFrame
        indexed by the simulation date range.

        Args:
            Route: Flag to decide whether to route the generated
                discharge hydrograph. Use 0 for no routing or 1 to
                enable routing. Defaults to 0.
            RoutingFn: Function to route the discharge hydrograph.
                If None, an empty list is used. Defaults to None.

        Note:
            The following attributes should be defined before calling
            this method:

            - ``LumpedModel``: Conceptual model containing a
              ``simulate`` method.
            - ``data``: Numpy array of meteorological data with
              columns for precipitation, evapotranspiration,
              temperature, and long-term average temperature.
            - ``Parameters``: Numpy array of conceptual model
              parameters.
            - ``CatArea``: Catchment area in km2.
            - ``conversion_factor``: Time conversion factor
              (e.g., 24 for daily).
            - ``InitialCond``: List of initial state variable
              values [sp, sm, uz, lz, wc].
            - ``Snow``: Whether to use the snow subroutine (0 or 1).
            - ``q_init``: Initial discharge value.
        """
        if RoutingFn is None and Route != 0:
            raise ValueError("RoutingFn must be a callable when Route != 0")
        if self.temporal_resolution.lower() == "daily":
            ind = pd.date_range(self.start, self.end, freq="D")
        else:
            ind = pd.date_range(self.start, self.end, freq="h")

        Qsim = pd.DataFrame(index=ind)

        Wrapper.Lumped(self, Route, RoutingFn)
        Qsim["q"] = self.Qsim
        self.Qsim = Qsim[:]
        logger.info("Lumped model run has finished successfully")


if __name__ == "__main__":
    print("Run")
