"""The schema of a YAML run configuration.

This module describes data and nothing else: `RunConfig` and the blocks it nests validate a
parsed YAML mapping and hold the result. It imports nothing from `hapi`, which keeps it a leaf
of the import graph and lets `hapi.catchment` import it at module level. Reading the file and
building a model out of it -- the `Catchment` construction, the `MeteoInputs` / `FlowNetwork`
loaders, the `read_*` call order -- belongs to :meth:`hapi.catchment.Catchment.from_yaml`.

The schema covers lumped and distributed runs, which disagree on the shape of two blocks:

- `meteo`: a grid for distributed (raster folders or NetCDF, per `source`), and a single CSV of
  catchment-average drivers for lumped.
- `gauges`: a gauge table plus a folder of per-gauge discharge files for distributed, and one
  discharge file with no table for lumped.

Which fields are required therefore depends on `catchment.spatial_resolution` and, for a
distributed run, on `meteo.source`. Those cross-field rules are enforced here by model
validators, so a `RunConfig` that validates is one the builder can consume without re-checking.

Lake-aware runs (`hapi.catchment.Lake`) and the flood model (`Run.RunFloodModel`) are out of
scope -- both need inputs this schema does not carry.

Examples:
    - Validate a lumped configuration and read back what it holds:
        ```python
        >>> from hapi.config import RunConfig
        >>> config = RunConfig.model_validate(
        ...     {
        ...         "catchment": {
        ...             "name": "Coello",
        ...             "start": "2009-01-01",
        ...             "end": "2011-12-31",
        ...         },
        ...         "meteo": {"path": "meteo_data.csv"},
        ...         "parameters": {"path": "parameters.txt"},
        ...         "conceptual_model": {
        ...             "model_class": "HBVBergestrom92",
        ...             "catchment_area": 1530,
        ...             "initial_condition": [0, 10, 10, 10, 0],
        ...         },
        ...         "gauges": {"discharge": "Qout_c.csv"},
        ...     }
        ... )
        >>> config.catchment.spatial_resolution
        'lumped'
        >>> config.conceptual_model.catchment_area
        1530.0
        >>> config.gauges.fmt
        '%Y-%m-%d'

        ```
    - A distributed run needs a routing network, so one without it is refused:
        ```python
        >>> from pydantic import ValidationError
        >>> from hapi.config import RunConfig
        >>> try:
        ...     RunConfig.model_validate(
        ...         {
        ...             "catchment": {
        ...                 "name": "Coello",
        ...                 "start": "2009-01-01",
        ...                 "end": "2011-12-31",
        ...                 "spatial_resolution": "distributed",
        ...             },
        ...             "meteo": {"path": "meteo_data.csv"},
        ...             "parameters": {"path": "parameters.txt"},
        ...             "conceptual_model": {
        ...                 "model_class": "HBVBergestrom92",
        ...                 "catchment_area": 1530,
        ...                 "initial_condition": [0, 10, 10, 10, 0],
        ...             },
        ...             "gauges": {"discharge": "Qout_c.csv"},
        ...         }
        ...     )
        ... except ValidationError as error:
        ...     print(error.errors()[0]["msg"])
        Value error, catchment.spatial_resolution is 'distributed', which needs a flow_network block

        ```
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Rejects unknown keys, so a misspelled field in the YAML fails loudly at parse time rather
#: than being dropped and surfacing later as a missing input.
_STRICT = ConfigDict(extra="forbid")


class CatchmentConfig(BaseModel):
    """The `Catchment` constructor arguments.

    Attributes:
        name: Catchment name.
        start: Start date, parsed with `fmt`. Kept a string: the constructor does the parsing,
            and an unquoted YAML date would arrive here already a `date`.
        end: End date, parsed with `fmt`.
        fmt: `strptime` format for `start` / `end`.
        spatial_resolution: `"lumped"` or `"distributed"`. Selects the shape of `meteo` and
            `gauges`, and whether `flow_network` is required.
        temporal_resolution: `"daily"` or `"hourly"`.
        routing_method: `"muskingum"` or `"maxbas"`. Assigned onto `model.routing_method`; which
            `Run.*` entry point actually routes with it is the caller's choice.
    """

    model_config = _STRICT

    name: str
    start: str
    end: str
    fmt: str = "%Y-%m-%d"
    spatial_resolution: Literal["lumped", "distributed"] = "lumped"
    temporal_resolution: Literal["daily", "hourly"] = "daily"
    routing_method: Literal["muskingum", "maxbas"] = "muskingum"


class MeteoConfig(BaseModel):
    """The meteorological drivers: a distributed grid or a lumped CSV.

    Attributes:
        source: Which `MeteoInputs` loader builds the grid. Ignored for a lumped run, which
            always reads `path` as a single CSV.
        precipitation: Rainfall folder (`"rasters"`), NetCDF path (`"netcdf_files"`), or the
            variable name holding rainfall inside `path` (`"netcdf"`).
        temperature: As `precipitation`, for temperature.
        evapotranspiration: As `precipitation`, for evapotranspiration.
        path: The combined NetCDF (`source="netcdf"`) or the lumped meteo CSV.
        start: Window start; `None` falls back to `catchment.start`. Distributed only.
        end: Window end; `None` falls back to `catchment.end`. Distributed only.
        fmt: `strptime` format for `start` / `end`.
        glob: Raster glob, `source="rasters"` only.
        regex_string: Date regex within file names, `source="rasters"` only.
        file_name_data_fmt: `strptime` format for the matched date; inferred if `None`.
        per_variable: Per-folder overrides of the reader arguments, `source="rasters"` only.
        gdal_env: GDAL environment overrides for the raster read, `source="rasters"` only.
    """

    model_config = _STRICT

    source: Literal["rasters", "netcdf", "netcdf_files"] = "rasters"
    precipitation: str | None = None
    temperature: str | None = None
    evapotranspiration: str | None = None
    path: str | None = None
    start: str | None = None
    end: str | None = None
    fmt: str = "%Y-%m-%d"
    glob: str = "*.tif"
    regex_string: str = r"\d{4}.\d{2}.\d{2}"
    file_name_data_fmt: str | None = None
    per_variable: dict[str, dict[str, Any]] | None = None
    gdal_env: dict[str, str] | None = None


class FlowNetworkConfig(BaseModel):
    """The routing network. Distributed runs only.

    Attributes:
        flow_accumulation: Path to the flow-accumulation raster.
        flow_direction: Path to the flow-direction raster. Muskingum needs it; MAXBAS sends
            every cell straight to the outlet and never reads one, so it may be omitted.
    """

    model_config = _STRICT

    flow_accumulation: str
    flow_direction: str | None = None


class ParametersConfig(BaseModel):
    """Where the conceptual-model parameters live.

    Attributes:
        path: Folder of parameter rasters (distributed) or a single file (lumped).
        snow: Whether the parameter set includes the snow routine (15 parameters against 10).
        maxbas: Whether the set carries the triangular-routing parameter. Independent of
            `catchment.routing_method` -- this describes the parameter set, not the run.
    """

    model_config = _STRICT

    path: str
    snow: bool = False
    maxbas: bool = False


class ConceptualModelConfig(BaseModel):
    """The lumped conceptual model, run per cell (distributed) or per catchment (lumped).

    Attributes:
        model_class: Name of the conceptual model, e.g. `"HBVBergestrom92"`. Resolved to a class
            by the builder, which owns the registry of available models.
        catchment_area: Catchment area, km2.
        initial_condition: `[sp, sm, uz, lz, wc]`, exactly five values.
        q_init: Initial discharge; `None` derives it from the initial condition.
    """

    # `model_class` would collide with pydantic's protected `model_` namespace, so the namespace
    # is cleared rather than renaming a field the YAML already uses.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_class: str
    catchment_area: float = Field(gt=0)
    initial_condition: list[float] = Field(min_length=5, max_length=5)
    q_init: float | None = None


class GaugesConfig(BaseModel):
    """The observed discharge the run is scored against.

    Attributes:
        discharge: Folder of one CSV per gauge id (distributed) or a single CSV (lumped).
        table: Gauge locations and properties. Distributed only; a lumped run has no grid to
            locate gauges on.
        column: Gauge-table column holding the ids the discharge file names match.
        delimiter: Discharge CSV delimiter.
        fmt: `strptime` format for the discharge CSV's date column.
    """

    model_config = _STRICT

    discharge: str
    table: str | None = None
    column: str = "id"
    delimiter: str = ","
    fmt: str = "%Y-%m-%d"


class OutputsConfig(BaseModel):
    """Where to write results after the run.

    Attributes:
        results_dir: Folder `save_results` writes into.
    """

    model_config = _STRICT

    results_dir: str | None = None


class RunConfig(BaseModel):
    """The full input set for one `Catchment` build.

    Attributes:
        catchment: Constructor arguments.
        meteo: The meteorological drivers.
        conceptual_model: The lumped conceptual model.
        parameters: Where the conceptual-model parameters live. Omit for a calibration, which
            derives them from the bounds handed to `read_parameters_bound` rather than reading
            a fitted set.
        gauges: The observed discharge. Omit for a run that is not scored against gauges.
        flow_network: The routing network. Required for distributed, absent for lumped.
        outputs: Where to write results.
    """

    model_config = _STRICT

    catchment: CatchmentConfig
    meteo: MeteoConfig
    conceptual_model: ConceptualModelConfig
    parameters: ParametersConfig | None = None
    gauges: GaugesConfig | None = None
    flow_network: FlowNetworkConfig | None = None
    outputs: OutputsConfig | None = None

    @model_validator(mode="after")
    def _check_blocks_match_the_spatial_resolution(self) -> RunConfig:
        """Enforce the fields each spatial resolution requires.

        Returns:
            RunConfig: This config, unchanged.

        Raises:
            ValueError: A block the chosen `spatial_resolution` needs is missing, or one of the
                three drivers a distributed `meteo.source` needs is unset.
        """
        if self.catchment.spatial_resolution == "distributed":
            if self.flow_network is None:
                raise ValueError(
                    "catchment.spatial_resolution is 'distributed', which needs a flow_network "
                    "block"
                )
            # `flow_direction` is optional on the block because MAXBAS sends every cell straight
            # to the outlet and never reads one. Muskingum routes along the network, so without
            # it the build succeeds and `Run.RunHapi` dereferences a None array after every
            # raster has been read.
            if (
                self.catchment.routing_method == "muskingum"
                and self.flow_network.flow_direction is None
            ):
                raise ValueError(
                    "catchment.routing_method is 'muskingum', which routes along the network, "
                    "so flow_network.flow_direction is required"
                )
            # Only when gauges are configured at all: a distributed run that is not scored
            # against observations omits the block entirely.
            if self.gauges is not None and self.gauges.table is None:
                raise ValueError(
                    "catchment.spatial_resolution is 'distributed', which needs gauges.table "
                    "to locate the gauges on the grid"
                )
            missing = [
                name
                for name in ("precipitation", "temperature", "evapotranspiration")
                if getattr(self.meteo, name) is None
            ]
            if missing:
                raise ValueError(
                    f"a distributed run needs all three drivers; meteo is missing "
                    f"{', '.join(missing)}"
                )
            if self.meteo.source == "netcdf" and self.meteo.path is None:
                raise ValueError("meteo.source is 'netcdf', which needs meteo.path")
        elif self.meteo.path is None:
            raise ValueError(
                "catchment.spatial_resolution is 'lumped', which needs meteo.path -- the CSV "
                "of catchment-average drivers"
            )
        return self
