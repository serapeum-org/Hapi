"""Load a run configuration from YAML and assemble a `Catchment` from it.

Every example script under `examples/hydrological-model/*/run/` starts with a block of
path/constant assignments before the `Catchment` is built and its `read_*` methods are called
in the exact order the build-then-mutate pattern (see the `hapi.catchment` module docstring)
requires. This module lifts that block into a YAML file plus a loader: `from_yaml` reads it and
assigns every input onto the `Catchment` object the same way the hand-written block did.
`load_config` is the parsing step alone, for callers that want the `RunConfig` without building
a model from it. Running the model is still the caller's job, exactly as in a hand-wired script
-- call `Run.RunHapi(model)`, `Run.runFW1(model)` or `Run.runLumped(model, ...)` yourself,
whichever `model.routing_method` / `model.spatial_resolution` calls for.

Two spatial resolutions are supported -- lumped and distributed -- selected by
`catchment.spatial_resolution`. They disagree on the *shape* of two blocks:

- `meteo`: a grid (`MeteoInputs`, via raster folders or NetCDF) for distributed, a single CSV
  (`Catchment.read_lumped_inputs`) for lumped.
- `gauges.discharge`: a folder of one CSV per gauge id for distributed, a single CSV for
  lumped.

Lake-aware runs (`hapi.catchment.Lake`) and the flood model (`Run.RunFloodModel`) are out of
scope for this schema -- both need inputs it does not carry.

Examples:
    >>> from hapi.config import from_yaml  # doctest: +SKIP
    >>> from hapi.run import Run  # doctest: +SKIP
    >>> model = from_yaml("case-study.yaml")  # doctest: +SKIP
    >>> Run.RunHapi(model)  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.rrm.hbv import HBV
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92

#: Conceptual model classes resolvable by name from `conceptual_model.model_class` in the YAML.
CONCEPTUAL_MODELS: dict[str, type] = {
    "HBVBergestrom92": HBVBergestrom92,
    "HBV": HBV,
}

#: `Catchment.__init__` stores `routing_method` verbatim, with no case-folding of its own (unlike
#: `spatial_resolution` / `temporal_resolution`, which it lowercases). `distrrm.SpatialRouting`
#: -- the Muskingum routing loop `Run.RunHapi` reaches -- then compares it with
#: `Model.routing_method != "Muskingum"`, an exact, case-sensitive match against that one
#: literal. Any other spelling, including a differently-cased "muskingum", makes every cell take
#: the MAXBAS branch instead and read `Model.bankfull_depth`, which is `None` outside the flood
#: model and raises `TypeError`. MAXBAS itself never calls `SpatialRouting`, so its label is
#: cosmetic -- but Muskingum's must be this exact string.
_ROUTING_METHOD_LABELS: dict[str, str] = {"muskingum": "Muskingum", "maxbas": "MAXBAS"}


@dataclasses.dataclass
class CatchmentConfig:
    """The `Catchment` constructor arguments.

    Attributes:
        name: Catchment name.
        start: Start date, parsed with `fmt`.
        end: End date, parsed with `fmt`.
        fmt: `strptime` format for `start` / `end`.
        spatial_resolution: `"lumped"` or `"distributed"`.
        temporal_resolution: `"daily"` or `"hourly"`.
        routing_method: `"muskingum"` or `"maxbas"`. Assigned onto `model.routing_method`;
            which `Run.*` entry point actually routes with it is the caller's choice.
    """

    name: str
    start: str
    end: str
    fmt: str = "%Y-%m-%d"
    spatial_resolution: str = "lumped"
    temporal_resolution: str = "daily"
    routing_method: str = "muskingum"


@dataclasses.dataclass
class MeteoConfig:
    """The meteorological drivers: a distributed grid or a lumped CSV.

    Attributes:
        source: `"rasters"`, `"netcdf"` or `"netcdf_files"` (distributed); ignored for lumped,
            which always reads `path` as a single CSV.
        precipitation: Rainfall folder (`"rasters"`), NetCDF path (`"netcdf_files"`), or the
            variable name holding rainfall inside `path` (`"netcdf"`).
        temperature: As `precipitation`, for temperature.
        evapotranspiration: As `precipitation`, for evapotranspiration.
        path: The combined NetCDF (`source="netcdf"`) or the lumped meteo CSV.
        start: Optional window start; `None` uses `catchment.start`. Distributed only.
        end: Optional window end; `None` uses `catchment.end`. Distributed only.
        fmt: `strptime` format for `start` / `end`.
        glob: Raster glob, `source="rasters"` only.
        regex_string: Date regex within file names, `source="rasters"` only.
        file_name_data_fmt: `strptime` format for the matched date; inferred if `None`.
        per_variable: Per-folder overrides of the reader arguments, `source="rasters"` only.
        gdal_env: GDAL environment overrides for the raster read, `source="rasters"` only.
    """

    source: str = "rasters"
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


@dataclasses.dataclass
class FlowNetworkConfig:
    """The routing network. Distributed modes only.

    Attributes:
        flow_accumulation: Path to the flow-accumulation raster.
        flow_direction: Path to the flow-direction raster. Required for Muskingum, unused (and
            may be omitted) for MAXBAS.
    """

    flow_accumulation: str
    flow_direction: str | None = None


@dataclasses.dataclass
class ParametersConfig:
    """Where the conceptual-model parameters live.

    Attributes:
        path: Folder of parameter rasters (distributed) or a CSV file (lumped).
        snow: Whether the parameter set includes the snow routine (15 parameters vs 10).
        maxbas: Whether the parameter set was built for MAXBAS routing. Independent of
            `catchment.routing_method` -- this describes the parameter *set*, not the run.
    """

    path: str
    snow: bool = False
    maxbas: bool = False


@dataclasses.dataclass
class ConceptualModelConfig:
    """The lumped conceptual model run per cell (distributed) or per catchment (lumped).

    Attributes:
        model_class: Name in `CONCEPTUAL_MODELS`, e.g. `"HBVBergestrom92"`.
        catchment_area: Catchment area, km2.
        initial_condition: `[sp, sm, uz, lz, wc]`, five values.
        q_init: Optional initial discharge; `None` derives it from the initial condition.
    """

    model_class: str
    catchment_area: float
    initial_condition: list[float]
    q_init: float | None = None


@dataclasses.dataclass
class GaugesConfig:
    """The observed discharge used to score the run.

    Attributes:
        discharge: Folder of one CSV per gauge id (distributed) or a single CSV (lumped).
        table: Gauge locations and properties. Distributed only.
        column: Gauge table column holding the ids the discharge folder's file names match.
            Unused for lumped.
        delimiter: Discharge CSV delimiter.
        fmt: `strptime` format for the discharge CSV's date column.
    """

    discharge: str
    table: str | None = None
    column: str = "id"
    delimiter: str = ","
    fmt: str = "%Y-%m-%d"


@dataclasses.dataclass
class OutputsConfig:
    """Where to write results after the run.

    Attributes:
        results_dir: Folder `save_results` writes into.
    """

    results_dir: str | None = None


@dataclasses.dataclass
class RunConfig:
    """The full input set for one `Catchment` build.

    Attributes:
        catchment: Constructor arguments.
        meteo: The meteorological drivers.
        parameters: Where the conceptual-model parameters live.
        conceptual_model: The lumped conceptual model.
        gauges: The observed discharge.
        flow_network: The routing network. `None` for lumped.
        outputs: Where to write results. `None` if the run is only scored in memory.
    """

    catchment: CatchmentConfig
    meteo: MeteoConfig
    parameters: ParametersConfig
    conceptual_model: ConceptualModelConfig
    gauges: GaugesConfig
    flow_network: FlowNetworkConfig | None = None
    outputs: OutputsConfig | None = None


def load_config(path: str | Path) -> RunConfig:
    """Read a run configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        RunConfig: The parsed configuration, not yet built into a `Catchment`.

    Raises:
        ValueError: `catchment.spatial_resolution` is `"distributed"` and the file has no
            `flow_network` block.
    """
    raw = yaml.safe_load(Path(path).read_text())

    catchment = CatchmentConfig(**raw["catchment"])
    is_distributed = catchment.spatial_resolution.lower() == "distributed"

    flow_network = (
        FlowNetworkConfig(**raw["flow_network"]) if "flow_network" in raw else None
    )
    if is_distributed and flow_network is None:
        raise ValueError(
            "catchment.spatial_resolution is 'distributed' but the file has no flow_network "
            "block"
        )

    return RunConfig(
        catchment=catchment,
        meteo=MeteoConfig(**raw["meteo"]),
        parameters=ParametersConfig(**raw["parameters"]),
        conceptual_model=ConceptualModelConfig(**raw["conceptual_model"]),
        gauges=GaugesConfig(**raw["gauges"]),
        flow_network=flow_network,
        outputs=OutputsConfig(**raw["outputs"]) if "outputs" in raw else None,
    )


def _build_meteo(meteo: MeteoConfig, catchment: CatchmentConfig) -> MeteoInputs:
    """Dispatch to the `MeteoInputs` loader `meteo.source` names.

    Args:
        meteo: The meteo block. `source` must be `"rasters"`, `"netcdf"` or `"netcdf_files"`.
        catchment: Supplies the default window when `meteo.start` / `meteo.end` are `None`.

    Returns:
        MeteoInputs: The three cubes, windowed to the model's dates.

    Raises:
        ValueError: `meteo.source` is not one of the three recognised loaders.
        AssertionError: `meteo.source` names a loader whose required fields are `None` --
            `precipitation` / `temperature` / `evapotranspiration` for every source, plus
            `path` for `"netcdf"`.
    """
    start = meteo.start or catchment.start
    end = meteo.end or catchment.end

    # precipitation/temperature/evapotranspiration are Optional on the dataclass because a
    # lumped config never sets them, but every distributed source requires all three.
    assert meteo.precipitation is not None, (
        "meteo.precipitation is required for a distributed run"
    )
    assert meteo.temperature is not None, (
        "meteo.temperature is required for a distributed run"
    )
    assert meteo.evapotranspiration is not None, (
        "meteo.evapotranspiration is required for a distributed run"
    )

    if meteo.source == "rasters":
        kwargs: dict[str, Any] = dict(
            glob=meteo.glob,
            regex_string=meteo.regex_string,
            file_name_data_fmt=meteo.file_name_data_fmt,
            start=start,
            end=end,
            fmt=meteo.fmt,
        )
        if meteo.per_variable is not None:
            kwargs["per_variable"] = meteo.per_variable
        if meteo.gdal_env is not None:
            kwargs["gdal_env"] = meteo.gdal_env
        return MeteoInputs.from_rasters(
            meteo.precipitation, meteo.temperature, meteo.evapotranspiration, **kwargs
        )

    if meteo.source == "netcdf":
        assert meteo.path is not None, "meteo.path is required for meteo.source: netcdf"
        return MeteoInputs.from_netcdf(
            meteo.path,
            precipitation=meteo.precipitation,
            temperature=meteo.temperature,
            evapotranspiration=meteo.evapotranspiration,
            start=start,
            end=end,
            fmt=meteo.fmt,
        )

    if meteo.source == "netcdf_files":
        return MeteoInputs.from_netcdf_files(
            meteo.precipitation,
            meteo.temperature,
            meteo.evapotranspiration,
            start=start,
            end=end,
            fmt=meteo.fmt,
        )

    raise ValueError(
        f"meteo.source must be 'rasters', 'netcdf' or 'netcdf_files' for a distributed run, "
        f"got {meteo.source!r}"
    )


def from_yaml(path: str | Path) -> Catchment:
    """Read a YAML run configuration and assemble a `Catchment` from it in one call.

    Calls `load_config(path)`, then follows the build-then-mutate pattern `hapi.catchment`
    documents: constructs the model, assigns `meteo` and (distributed only) `flow_network`,
    then calls the `read_*` methods in the order they depend on each other -- the same
    sequence a hand-wired script's "Paths" block used to drive by hand. Running the model is
    left to the caller, via whichever `Run.*` entry point (`RunHapi`, `runFW1`, `runLumped`)
    fits `model.routing_method` / `model.spatial_resolution`.

    Args:
        path: Path to the YAML file.

    Returns:
        Catchment: The model, with every read_* call made -- gauges included.

    Raises:
        ValueError: `catchment.spatial_resolution` is `"distributed"` and the file has no
            `flow_network` block, or `conceptual_model.model_class` is not in
            `CONCEPTUAL_MODELS`.
    """
    config = load_config(path)
    c = config.catchment
    routing_label = _ROUTING_METHOD_LABELS.get(
        c.routing_method.lower(), c.routing_method
    )
    model = Catchment(
        c.name,
        c.start,
        c.end,
        fmt=c.fmt,
        spatial_resolution=c.spatial_resolution,
        temporal_resolution=c.temporal_resolution,
        routing_method=routing_label,
    )

    is_distributed = c.spatial_resolution.lower() == "distributed"

    if is_distributed:
        model.meteo = _build_meteo(config.meteo, c)
        fn = config.flow_network
        assert fn is not None, (
            "flow_network is required when spatial_resolution is distributed"
        )
        model.flow_network = FlowNetwork.from_rasters(
            fn.flow_accumulation, fn.flow_direction
        )
    else:
        assert config.meteo.path is not None, (
            "meteo.path is required when spatial_resolution is lumped"
        )
        model.read_lumped_inputs(config.meteo.path)

    p = config.parameters
    model.read_parameters(p.path, p.snow, maxbas=p.maxbas)

    cm = config.conceptual_model
    model_class = CONCEPTUAL_MODELS.get(cm.model_class)
    if model_class is None:
        raise ValueError(
            f"conceptual_model.model_class {cm.model_class!r} is not registered; known models "
            f"are {sorted(CONCEPTUAL_MODELS)}"
        )
    model.read_lumped_model(
        model_class, cm.catchment_area, cm.initial_condition, cm.q_init
    )

    g = config.gauges
    if is_distributed:
        assert fn is not None, (
            "flow_network is required when spatial_resolution is distributed"
        )
        assert g.table is not None, (
            "gauges.table is required when spatial_resolution is distributed"
        )
        model.read_gauge_table(g.table, fn.flow_accumulation, fmt=g.fmt)
    model.read_discharge_gauges(
        g.discharge, delimiter=g.delimiter, column=g.column, fmt=g.fmt
    )

    return model
