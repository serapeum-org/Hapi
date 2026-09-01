"""Run module for the Hapi hydrological model.

The run module connects the parameter spatial distribution function with
both components of the spatial representation of the hydrological process
(conceptual model and spatial routing) to calculate the predicted runoff
at known locations based on a given performance function.

`Run` is a namespace of static entry points, not a class to instantiate. Each one takes the
model it should run, validates it, and hands it to :class:`~hapi.wrapper.Wrapper`. What each
entry point requires is stated by the protocols below rather than by naming a concrete class:
:class:`~hapi.catchment.Catchment` satisfies them structurally, so this module does not import
it at runtime and anything else carrying the same attributes runs too.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from hapi.protocols import (
    DistributedModel,
    FloodModel,
    LumpedModelInputs,
)
from hapi.results import SimulationResults

# from hapi.hm.saintvenant import SaintVenant
from hapi.wrapper import Wrapper

if TYPE_CHECKING:
    from hapi.catchment import Lake as LakeType

ROWS_MISMATCH_ERROR = "the parameters must have as many rows as the catchment grid"
COLS_MISMATCH_ERROR = "the parameters must have as many columns as the catchment grid"
#: The flow-direction check tests both axes, so it says both. The entry points used to
#: carry three different wordings for this one check; the widest is the accurate one.
GRID_MISMATCH_ERROR = "all input data should have the same number of rows and columns"


def _check_parameters_cover_grid(model: DistributedModel) -> None:
    """Check the parameter array spans the catchment grid.

    The same two checks every distributed entry point makes before handing the model to the
    wrapper: a parameter array smaller than the grid is indexed out of range inside the
    per-cell loop, far from the call that supplied it.

    Args:
        model: The model about to run, carrying `parameters` and `flow_network`.

    Raises:
        ValueError: The parameter array has the wrong number of rows or columns.
    """
    shape = np.asarray(model.parameters.values).shape
    if shape[0] != model.flow_network.rows:
        raise ValueError(ROWS_MISMATCH_ERROR)
    if shape[1] != model.flow_network.cols:
        raise ValueError(COLS_MISMATCH_ERROR)


def _check_lake_meteo(model: DistributedModel, lake: LakeType) -> None:
    """Check the lake's record lines up with the distributed drivers.

    Args:
        model: The model about to run, whose `meteo` sets the expected length.
        lake: The lake whose `MeteoData` is checked.

    Raises:
        ValueError: The lake has no meteorological record, the record is a different length
            from the distributed drivers, or it carries fewer than the three columns the
            lake model reads.
    """
    meteo_data = lake.MeteoData
    if meteo_data is None:
        raise ValueError(
            "the lake has no meteorological data; call lake.read_meteo_data before "
            "running a lake-aware entry point"
        )
    if np.shape(meteo_data)[0] != model.meteo.time_steps:
        raise ValueError(
            "Lake meteorological data has to have the same length as the distributed "
            "raster data"
        )
    if np.shape(meteo_data)[1] < 3:
        raise ValueError(
            "Lake Meteo data has to have at least three columns of rain, ET, and Temp"
        )


def _warn_about_the_unrouted_river_cells(model: FloodModel) -> None:
    """Warn that the river cells will be left unrouted, and by how much.

    Skipping them is the handoff the flood model was designed around: a 1D hydraulic model
    (`SaintVenant.KinematicRaster`) routes them instead. That half left this package in
    commit `733957be` and now lives in Serapis, so nothing here picks those cells up -- on
    the Coello example that is 21 of the 89 catchment cells, carrying 92% of the discharge
    because the river cells are the high-accumulation ones. A caller who has
    Serapis downstream wants exactly this; a caller who set `routing_method="Kinematic"`
    without one gets a number that is not a hydrograph, and nothing used to say so.

    Only the *derived* case warns. Passing `skip_hydraulic_cells=True` is an explicit
    statement that something downstream takes the river cells, and is left quiet.

    Args:
        model: The model about to run, whose `bankfull_depth` marks the river cells.
    """
    # Only cells inside the catchment are ever routed, so only those can be skipped. The
    # bankfull-depth raster carries values outside the domain too, and counting those made
    # the message claim more river cells than the catchment has.
    inside = ~np.isnan(model.flow_network.flow_acc_arr)
    river_cells = int(
        np.count_nonzero((np.nan_to_num(model.bankfull_depth) > 0) & inside)
    )
    domain = int(np.count_nonzero(inside))
    warnings.warn(
        f"routing_method='Kinematic' leaves the {river_cells} river cells of {domain} "
        "unrouted, for a 1D hydraulic model to route instead -- but that model is not part "
        "of Hapi, so their discharge is simply absent from the results. Pass "
        "skip_hydraulic_cells=True to say a downstream model takes them and silence this, "
        "or use routing_method='Muskingum' to route every cell here.",
        UserWarning,
        stacklevel=3,
    )


def _validate_distributed(model: DistributedModel, check_flow_direction: bool) -> None:
    """Run the checks every distributed entry point makes before the wrapper.

    Args:
        model: The model about to run.
        check_flow_direction: Whether to compare the flow-direction raster against the grid.
            The MAXBAS paths never read that raster, so they do not require it.

    Raises:
        ValueError: The grid, the drivers and the parameters do not agree.
    """
    if check_flow_direction:
        flow_dir_arr = model.flow_network.flow_dir_arr
        # `FlowNetwork` takes the direction raster as optional because MAXBAS never reads
        # it. The paths that route cell to cell do, so say which raster is missing rather
        # than failing on None inside the routing loop.
        if flow_dir_arr is None:
            raise ValueError(
                "this run routes cell to cell and needs a flow-direction raster, but the "
                "flow network was built without one; pass it to FlowNetwork.from_rasters"
            )
        fd_rows, fd_cols = flow_dir_arr.shape
        if fd_rows != model.flow_network.rows or fd_cols != model.flow_network.cols:
            raise ValueError(GRID_MISMATCH_ERROR)

    # The three cubes already agree with each other (checked when MeteoInputs was
    # built); this is the other half -- that they cover the model's grid.
    model.meteo.validate_against(
        model.flow_network.rows, model.flow_network.cols, model.period.date_index
    )
    _check_parameters_cover_grid(model)


class Run:
    """Run the catchment model.

    A namespace of static entry points, not a class to instantiate. Each one validates the
    model it is given and hands it to :class:`~hapi.wrapper.Wrapper`, returning the
    :class:`~hapi.results.SimulationResults` the run produced. The same object is also
    assigned to the model's `results`, so the result arrays stay readable off the model
    afterwards.

    Methods:
        run_distributed: Run the distributed hydrological model.
        run_distributed_with_lake: Run the distributed model with a lake component.
        run_maxbas: Run the FW1 distributed model.
        run_maxbas_with_lake: Run the FW1 model with a lake component.
        run_lumped: Run the lumped conceptual model.
        run_flood: Run the flood model.

    Examples:
        - Build a model and run it; the results come back and stay on the model:
            ```python
            >>> from hapi.catchment import Catchment
            >>> from hapi.routing import Routing
            >>> from hapi.run import Run
            >>> model = Catchment.from_yaml(
            ...     "examples/hydrological-model/coello/run/coello-lumped-model-run.yaml"
            ... )
            >>> results = Run.run_lumped(model, 1, Routing.muskingum_v)
            >>> results.routing.value
            'lumped'
            >>> results is model.results
            True

            ```

    See Also:
        hapi.catchment.Catchment.from_yaml: Builds a model from a run configuration.
    """

    @staticmethod
    def run_distributed(model: DistributedModel) -> SimulationResults:
        """Run the distributed hydrological model.

        Validates that all input arrays (precipitation, evapotranspiration,
        temperature, parameters, and flow direction) have consistent
        dimensions, then executes the rainfall-runoff model via the
        Wrapper.

        Args:
            model: The model to run. See :class:`DistributedModel` for what it must carry.

        Returns:
            SimulationResults: The run's output, also assigned to `model.results`:

            - `state_variables`: 4D array (rows, cols, time, states) where
              states are [sp, wc, sm, uz, lv].
            - `qlz`: 3D array of the lower zone discharge.
            - `quz`: 3D array of the upper zone discharge.
            - `quz_routed`: 3D array of the upper zone discharge
              accumulated and routed at each time step.
            - `qlz_translated`: 3D array of the lower zone discharge
              translated at each time step.
            - `q_total`: `quz_routed + qlz_translated`. Routed by Muskingum, so the outlet
              cell carries the outlet hydrograph; `extract_discharge` fills `qout` from it.

        Raises:
            ValueError: If input data arrays have inconsistent
                row counts, column counts, or temporal lengths.
        """
        _validate_distributed(model, check_flow_direction=True)
        # run the model
        results = Wrapper.run_muskingum(model)

        logger.info("Model Run has finished")
        return results

    @staticmethod
    def run_flood(
        model: FloodModel, skip_hydraulic_cells: bool | None = None
    ) -> SimulationResults:
        """Run the flood model.

        Runs the conceptual distributed hydrological model with
        additional validation for river geometry inputs (bankfull depth,
        river width, river roughness, and flood plain roughness).

        Args:
            model: The model to run. See :class:`FloodModel` for what it must carry.
            skip_hydraulic_cells: Leave river cells (a positive `bankfull_depth`) unrouted
                by the Muskingum pass, because the kinematic-wave model routes them instead.
                `None`, the default, derives it from the catchment's own
                `routing_method` -- `"Kinematic"` means yes, anything else no -- and warns,
                because the hydraulic model that was supposed to take those cells is not part
                of Hapi. Pass `True` to state that something downstream takes them and
                silence the warning, or `False` to route every cell here.

        Warns:
            UserWarning: The skip was derived from `routing_method="Kinematic"`, so the river
                cells are left unrouted and their discharge is absent from the results.

        Raises:
            ValueError: If meteorological input arrays, parameter
                arrays, or river geometry arrays have inconsistent
                dimensions.
        """
        _validate_distributed(model, check_flow_direction=True)

        named_geometry = {
            "bankfull_depth": model.bankfull_depth,
            "river_width": model.river_width,
            "river_roughness": model.river_roughness,
            "flood_plain_roughness": model.flood_plain_roughness,
        }
        # `read_river_geometry` sets all four together, so a missing one means it was never
        # called. Naming them beats `np.shape(None)` raising from inside the comparison.
        missing = [name for name, arr in named_geometry.items() if arr is None]
        if missing:
            raise ValueError(
                f"the flood model needs the river geometry, but {', '.join(missing)} "
                "is not set; call read_river_geometry first"
            )
        # Rebuilt from the non-None values rather than `.values()` directly: the guard above
        # has already ruled None out, but only a comprehension carries that into the type.
        geometry = [arr for arr in named_geometry.values() if arr is not None]
        if any(np.shape(arr)[0] != model.flow_network.rows for arr in geometry):
            raise ValueError(GRID_MISMATCH_ERROR)
        if any(np.shape(arr)[1] != model.flow_network.cols for arr in geometry):
            raise ValueError("all input data should have the same number of columns")

        derived = skip_hydraulic_cells is None
        skip = (
            model.routing_method == "Kinematic" if derived else bool(skip_hydraulic_cells)
        )

        if skip and derived:
            _warn_about_the_unrouted_river_cells(model)

        # run the model
        results = Wrapper.run_muskingum(model, skip_hydraulic_cells=skip)
        logger.info("RRM has finished")
        # SV = SaintVenant()
        # SV.KinematicRaster(model)
        # print("1D model Run has finished")
        return results

    @staticmethod
    def run_distributed_with_lake(
        model: DistributedModel, lake: LakeType
    ) -> SimulationResults:
        """Run the distributed model with a lake component.

        Validates that all input arrays have consistent dimensions and
        that the lake meteorological data matches the simulation period,
        then executes the rainfall-runoff model with lake routing via
        the Wrapper.

        Args:
            model: The model to run. See :class:`DistributedModel` for what it must carry.
            lake: Lake object containing lake configuration and
                meteorological data. Must have a `MeteoData` attribute
                with shape `(time_steps, >= 3)` where columns are
                rain, ET, and temperature.

        Returns:
            SimulationResults: The run's output, also assigned to `model.results`.

        Raises:
            ValueError: If input data arrays have inconsistent
                dimensions or if the lake meteorological data length
                does not match the distributed raster data length.
        """
        _validate_distributed(model, check_flow_direction=True)
        _check_lake_meteo(model, lake)
        # run the model
        results = Wrapper.run_muskingum_with_lake(model, lake)

        logger.info("Model Run has finished")
        return results

    @staticmethod
    def run_maxbas(model: DistributedModel) -> SimulationResults:
        """Run the FW1 distributed hydrological model.

        Validates that all input arrays have consistent dimensions,
        then executes the FW1 model via the Wrapper. The flow-direction
        raster is not checked here because MAXBAS never reads it.

        Args:
            model: The model to run. See :class:`DistributedModel` for what it must carry.

        Returns:
            SimulationResults: The run's output, also assigned to `model.results`:

            - `state_variables`: 4D array of state variables.
            - `qout`: 1D array of calculated discharge at the catchment
              outlet, summed over every cell.
            - `quz`: 3D array of distributed discharge for each cell.
            - `q_total`, `quz_routed`, `qlz_translated`: 3D per-cell fields
              read by `save_results` and `plot_distributed_results`. MAXBAS
              routes each cell straight to the outlet, so a cell of `q_total` is
              that cell's *contribution* to the outlet — `np.nansum` over the
              domain reproduces `qout`. Use
              `extract_discharge` reads the routing off the results and takes the
              basin-wide sum on this path automatically.

        Raises:
            ValueError: If input data arrays have inconsistent
                row counts, column counts, or temporal lengths.
        """
        _validate_distributed(model, check_flow_direction=False)
        # run the model
        results = Wrapper.run_maxbas(model)

        logger.info("Model Run has finished")
        return results

    @staticmethod
    def run_maxbas_with_lake(
        model: DistributedModel, lake: LakeType
    ) -> SimulationResults:
        """Run the FW1 distributed model with a lake component.

        Validates that all input arrays have consistent dimensions and
        that the lake meteorological data matches the simulation period,
        then executes the FW1 model with lake routing via the Wrapper.

        Args:
            model: The model to run. See :class:`DistributedModel` for what it must carry.
            lake: Lake object containing lake configuration and
                meteorological data. Must have a `MeteoData` attribute
                with shape `(time_steps, >= 3)` where columns are
                rain, ET, and temperature.

        Returns:
            SimulationResults: The run's output, also assigned to `model.results`.

        Raises:
            ValueError: If input data arrays have inconsistent
                dimensions or if the lake meteorological data length
                does not match the distributed raster data length.
        """
        _validate_distributed(model, check_flow_direction=False)
        _check_lake_meteo(model, lake)

        # run the model
        return Wrapper.run_maxbas_with_lake(model, lake)

    @staticmethod
    def run_lumped(
        model: LumpedModelInputs,
        Route: int = 0,
        routing_fn: Callable[..., Any] | None = None,
    ) -> SimulationResults:
        """Run the lumped conceptual model.

        Executes a lumped conceptual hydrological model, optionally
        routing the generated discharge hydrograph. The simulated
        discharge is stored in `model.Qsim` as a pandas DataFrame
        indexed by the simulation date range.

        Args:
            model: The model to run. See :class:`LumpedModelInputs` for what it must carry.
            Route: Flag to decide whether to route the generated
                discharge hydrograph. Use 0 for no routing or 1 to
                enable routing. Defaults to 0.
            routing_fn: Function to route the discharge hydrograph.
                Required when `Route` is not 0.

        Returns:
            SimulationResults: The run's output, also assigned to `model.results`. A lumped
            run applies no spatial routing, so the routed fields stay None and
            `routing` is `RoutingKind.LUMPED`.

        Raises:
            ValueError: `Route` is not 0 and no routing function was given.
        """
        if routing_fn is None and Route != 0:
            raise ValueError("routing_fn must be a callable when Route != 0")
        # The calendar belongs to the period, which derives it from the span and the
        # resolution -- this branch used to be written out here for the fourth time.
        ind = model.period.date_index

        Qsim = pd.DataFrame(index=ind)

        results = Wrapper.run_lumped(model, Route, routing_fn)
        Qsim["q"] = model.Qsim
        model.Qsim = Qsim[:]
        logger.info("Lumped model run has finished successfully")
        return results


if __name__ == "__main__":
    print("Run")
