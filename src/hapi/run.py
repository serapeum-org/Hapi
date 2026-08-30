"""Run module for the Hapi hydrological model.

The run module connects the parameter spatial distribution function with
both components of the spatial representation of the hydrological process
(conceptual model and spatial routing) to calculate the predicted runoff
at known locations based on a given performance function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from hapi.catchment import Catchment
from hapi.catchment import Lake as LakeType

# from hapi.hm.saintvenant import SaintVenant
from hapi.wrapper import Wrapper

ROWS_MISMATCH_ERROR = "the parameters must have as many rows as the catchment grid"
COLS_MISMATCH_ERROR = "the parameters must have as many columns as the catchment grid"
GRID_MISMATCH_ERROR = "all input data should have the same number of rows"


def _check_parameters_cover_grid(model: Catchment) -> None:
    """Check the parameter array spans the catchment grid.

    The same two checks every distributed entry point makes before handing the model to the
    wrapper: a parameter array smaller than the grid is indexed out of range inside the
    per-cell loop, far from the call that supplied it.

    Args:
        model: The model about to run, carrying `parameters` and `flow_network`.

    Raises:
        ValueError: The parameter array has the wrong number of rows or columns.
    """
    if np.shape(model.parameters)[0] != model.flow_network.rows:
        raise ValueError(ROWS_MISMATCH_ERROR)
    if np.shape(model.parameters)[1] != model.flow_network.cols:
        raise ValueError(COLS_MISMATCH_ERROR)


def _check_lake_meteo(model: Catchment, lake: LakeType) -> None:
    """Check the lake's record lines up with the distributed drivers.

    Args:
        model: The model about to run, whose `meteo` sets the expected length.
        lake: The lake whose `MeteoData` is checked.

    Raises:
        ValueError: The lake record is a different length from the distributed drivers, or
            carries fewer than the three columns the lake model reads.
    """
    if np.shape(lake.MeteoData)[0] != model.meteo.time_steps:
        raise ValueError(
            "Lake meteorological data has to have the same length as the distributed "
            "raster data"
        )
    if np.shape(lake.MeteoData)[1] < 3:
        raise ValueError(
            "Lake Meteo data has to have at least three columns of rain, ET, and Temp"
        )


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

        - `state_variables`: 4D array (rows, cols, time, states) where
          states are [sp, wc, sm, uz, lv].
        - `qlz`: 3D array of the lower zone discharge.
        - `quz`: 3D array of the upper zone discharge.
        - `qout`: 1D timeseries of discharge at the catchment outlet
          in m3/sec.
        - `quz_routed`: 3D array of the upper zone discharge
          accumulated and routed at each time step.
        - `qlz_translated`: 3D array of the lower zone discharge
          translated at each time step.

        Raises:
            ValueError: If input data arrays have inconsistent
                row counts, column counts, or temporal lengths.
        """
        # input dimensions
        fd_rows, fd_cols = self.flow_network.flow_dir_arr.shape
        if fd_rows != self.flow_network.rows or fd_cols != self.flow_network.cols:
            raise ValueError(GRID_MISMATCH_ERROR)

        # input dimensions
        # The three cubes already agree with each other (checked when MeteoInputs was
        # built); this is the other half -- that they cover the model's grid.
        self.meteo.validate_against(
            self.flow_network.rows, self.flow_network.cols, self.date_index
        )
        _check_parameters_cover_grid(self)
        # run the model
        Wrapper.RRMModel(self)

        print("Model Run has finished")

    def RunFloodModel(self):
        """Run the flood model.

        Runs the conceptual distributed hydrological model with
        additional validation for river geometry inputs (bankfull depth,
        river width, river roughness, and flood plain roughness).

        Raises:
            ValueError: If meteorological input arrays, parameter
                arrays, or river geometry arrays have inconsistent
                dimensions.
        """
        # input dimensions
        [fd_rows, fd_cols] = self.flow_network.flow_dir_arr.shape
        if fd_rows != self.flow_network.rows or fd_cols != self.flow_network.cols:
            raise ValueError(GRID_MISMATCH_ERROR)

        # input dimensions
        # The three cubes already agree with each other (checked when MeteoInputs was
        # built); this is the other half -- that they cover the model's grid.
        self.meteo.validate_against(
            self.flow_network.rows, self.flow_network.cols, self.date_index
        )
        _check_parameters_cover_grid(self)
        if any(
            np.shape(arr)[0] != self.flow_network.rows
            for arr in (
                self.bankfull_depth,
                self.river_width,
                self.river_roughness,
                self.flood_plain_roughness,
            )
        ):
            raise ValueError(GRID_MISMATCH_ERROR)
        if any(
            np.shape(arr)[1] != self.flow_network.cols
            for arr in (
                self.bankfull_depth,
                self.river_width,
                self.river_roughness,
                self.flood_plain_roughness,
            )
        ):
            raise ValueError("all input data should have the same number of columns")

        # run the model
        Wrapper.RRMModel(self)
        print("RRM has finished")
        # SV = SaintVenant()
        # SV.KinematicRaster(self)
        # print("1D model Run has finished")

    def runHAPIwithLake(self, lake: LakeType):
        """Run the distributed model with a lake component.

        Validates that all input arrays have consistent dimensions and
        that the lake meteorological data matches the simulation period,
        then executes the rainfall-runoff model with lake routing via
        the Wrapper.

        Args:
            lake: Lake object containing lake configuration and
                meteorological data. Must have a `MeteoData` attribute
                with shape `(time_steps, >= 3)` where columns are
                rain, ET, and temperature.

        Raises:
            ValueError: If input data arrays have inconsistent
                dimensions or if the lake meteorological data length
                does not match the distributed raster data length.
        """
        # input dimensions
        [fd_rows, fd_cols] = self.flow_network.flow_dir_arr.shape
        if fd_rows != self.flow_network.rows or fd_cols != self.flow_network.cols:
            raise ValueError(
                "all input data should have the same number of rows and columns"
            )

        # input dimensions
        # The three cubes already agree with each other (checked when MeteoInputs was
        # built); this is the other half -- that they cover the model's grid.
        self.meteo.validate_against(
            self.flow_network.rows, self.flow_network.cols, self.date_index
        )
        _check_parameters_cover_grid(self)
        _check_lake_meteo(self, lake)
        # run the model
        Wrapper.RRMWithlake(self, lake)

        print("Model Run has finished")

    def runFW1(self):
        """Run the FW1 distributed hydrological model.

        Validates that all input arrays have consistent dimensions,
        then executes the FW1 model via the Wrapper.

        The following instance attributes are set after execution:

        - `st`: 4D array of state variables.
        - `q_out`: 1D array of calculated discharge at the catchment
          outlet, summed over every cell.
        - `q_uz`: 3D array of distributed discharge for each cell.
        - `Qtot`, `quz_routed`, `qlz_translated`: 3D per-cell fields
          read by `save_results` and `plot_distributed_results`. MAXBAS
          routes each cell straight to the outlet, so a cell of `Qtot` is
          that cell's *contribution* to the outlet — `np.nansum` over the
          domain reproduces `q_out`. Use
          `extract_discharge(frame_work_1=True)`; the default outlet-cell
          shortcut is invalid for this path and raises.

        Raises:
            ValueError: If input data arrays have inconsistent
                row counts, column counts, or temporal lengths.
        """
        # The three cubes already agree with each other (checked when MeteoInputs was
        # built); this is the other half -- that they cover the model's grid.
        self.meteo.validate_against(
            self.flow_network.rows, self.flow_network.cols, self.date_index
        )
        _check_parameters_cover_grid(self)
        # run the model
        Wrapper.FW1(self)

        print("Model Run has finished")

    def RunFW1withLake(self, lake: LakeType):
        """Run the FW1 distributed model with a lake component.

        Validates that all input arrays have consistent dimensions and
        that the lake meteorological data matches the simulation period,
        then executes the FW1 model with lake routing via the Wrapper.

        Args:
            lake: Lake object containing lake configuration and
                meteorological data. Must have a `MeteoData` attribute
                with shape `(time_steps, >= 3)` where columns are
                rain, ET, and temperature.

        Note:
            The following catchment attributes should be set before
            calling this method:

            - `prec_path`: Path to the folder containing precipitation
              rasters.
            - `evap_path`: Path to the folder containing
              evapotranspiration rasters.
            - `temp_path`: Path to the folder containing temperature
              rasters.
            - `flow_acc_path`: Path to the flow accumulation raster.
            - `flow_direction_path`: Path to the flow direction raster.
            - `ParPath`: Path to the folder containing parameter
              rasters.
            - `p2`: List of unoptimized parameters where `p2[0]`
              is tfac and `p2[1]` is catchment area in km2.

        Raises:
            ValueError: If input data arrays have inconsistent
                dimensions or if the lake meteorological data length
                does not match the distributed raster data length.
        """
        # input data validation

        # input dimensions
        # The three cubes already agree with each other (checked when MeteoInputs was
        # built); this is the other half -- that they cover the model's grid.
        self.meteo.validate_against(
            self.flow_network.rows, self.flow_network.cols, self.date_index
        )
        _check_parameters_cover_grid(self)
        _check_lake_meteo(self, lake)

        # run the model
        Wrapper.FW1Withlake(self, lake)

    def runLumped(
        self,
        Route: int = 0,
        routing_fn: Callable[..., Any] | None = None,
    ):
        """Run the lumped conceptual model.

        Executes a lumped conceptual hydrological model, optionally
        routing the generated discharge hydrograph. The simulated
        discharge is stored in `self.Qsim` as a pandas DataFrame
        indexed by the simulation date range.

        Args:
            Route: Flag to decide whether to route the generated
                discharge hydrograph. Use 0 for no routing or 1 to
                enable routing. Defaults to 0.
            routing_fn: Function to route the discharge hydrograph.
                If None, an empty list is used. Defaults to None.

        Note:
            The following attributes should be defined before calling
            this method:

            - `LumpedModel`: Conceptual model containing a
              `simulate` method.
            - `data`: Numpy array of meteorological data with
              columns for precipitation, evapotranspiration,
              temperature, and long-term average temperature.
            - `Parameters`: Numpy array of conceptual model
              parameters.
            - `CatArea`: Catchment area in km2.
            - `conversion_factor`: Time conversion factor
              (e.g., 24 for daily).
            - `InitialCond`: List of initial state variable
              values [sp, sm, uz, lz, wc].
            - `Snow`: Whether to use the snow subroutine (0 or 1).
            - `q_init`: Initial discharge value.
        """
        if routing_fn is None and Route != 0:
            raise ValueError("routing_fn must be a callable when Route != 0")
        if self.temporal_resolution.lower() == "daily":
            ind = pd.date_range(self.start, self.end, freq="D")
        else:
            ind = pd.date_range(self.start, self.end, freq="h")

        Qsim = pd.DataFrame(index=ind)

        Wrapper.Lumped(self, Route, routing_fn)
        Qsim["q"] = self.Qsim
        self.Qsim = Qsim[:]
        logger.info("Lumped model run has finished successfully")


if __name__ == "__main__":
    print("Run")
