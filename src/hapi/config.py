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
validators, so the builder can consume a validated `RunConfig` without re-deriving them. Two
things are still left to it: resolving `conceptual_model.model_class` against its registry, and
every path, which is not touched until a reader opens it.

The same dependency decides which fields are *refused*: a field the chosen shape never reads is
rejected rather than dropped. `extra="forbid"` already does that for a misspelled key, and a
correctly spelled but inapplicable one is the same mistake -- a line in the file with no effect
-- so it fails the same way. Only fields written explicitly count, so defaults are never held
against an author.

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

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Rejects unknown keys, so a misspelled field in the YAML fails loudly at parse time rather
#: than being dropped and surfacing later as a missing input.
_STRICT = ConfigDict(extra="forbid")


#: Which `MeteoConfig` fields each `source` actually reaches, mirroring the three branches of
#: `MeteoInputs.from_config`. A field outside its source's set is one the reader never looks at,
#: so setting it says something the run will not do.
_METEO_FIELDS_BY_SOURCE: dict[str, frozenset[str]] = {
    "rasters": frozenset(
        {
            "source",
            "precipitation",
            "temperature",
            "evapotranspiration",
            "start",
            "end",
            "fmt",
            "glob",
            "regex_string",
            "file_name_data_fmt",
            "per_variable",
            "gdal_env",
        }
    ),
    "netcdf": frozenset(
        {
            "source",
            "path",
            "precipitation",
            "temperature",
            "evapotranspiration",
            "start",
            "end",
            "fmt",
        }
    ),
    "netcdf_files": frozenset(
        {
            "source",
            "precipitation",
            "temperature",
            "evapotranspiration",
            "variable",
            "start",
            "end",
            "fmt",
        }
    ),
}

#: A lumped run reads `meteo.path` as one CSV of catchment-average drivers -- no grid, no
#: window, no reader knobs -- and locates nothing, so the gauge table and the column that names
#: hydrographs from it have nothing to act on.
_LUMPED_METEO_FIELDS = frozenset({"source", "path"})
_LUMPED_GAUGES_FIELDS = frozenset({"discharge", "delimiter", "fmt"})


def _reject_fields_the_run_will_not_read(
    block: BaseModel, applicable: frozenset[str], block_name: str, because: str
) -> None:
    """Refuse fields of a block that the chosen run shape never reads.

    `extra="forbid"` catches a misspelled key; this catches a correctly spelled one that does
    not apply. Both are the same failure from the author's side -- a line in the file that has
    no effect -- and silently dropping the second is what makes a configuration hard to trust.
    Only explicitly set fields count, so a default the author never wrote is not held against
    them.

    Args:
        block: The block to check.
        applicable: Names the run shape actually reads.
        block_name: The block's key in the file, for the message.
        because: Why the rest do not apply, phrased to follow "which".

    Raises:
        ValueError: One or more set fields fall outside `applicable`.
    """
    inapplicable = sorted(block.model_fields_set - applicable)
    if inapplicable:
        named = ", ".join(f"{block_name}.{name}" for name in inapplicable)
        raise ValueError(f"{because}, so {named} would be read by nothing")


def _write_dates_in_the_block_format(values: Any, fields: tuple[str, ...]) -> Any:
    """Render any date YAML already parsed back into a string in the block's own format.

    An unquoted `start: 2009-01-01` is a `datetime.date` by the time pydantic sees it, and the
    date fields are strings because the readers downstream parse them with `fmt`. Rejecting the
    unquoted spelling would be rejecting the one a YAML author writes first, over a difference
    that carries no information: a parsed date has no format ambiguity left to preserve, so it
    is simply written back out in `fmt`.

    Args:
        values: The raw mapping pydantic is about to validate. Anything else is passed through
            for pydantic to reject with its own message.
        fields: Names of the date fields in this block.

    Returns:
        Any: The mapping, with any parsed date in `fields` replaced by its `fmt` rendering.
    """
    if not isinstance(values, dict):
        return values

    fmt = values.get("fmt", "%Y-%m-%d")
    if not isinstance(fmt, str):
        return values

    rendered = dict(values)
    for field in fields:
        value = rendered.get(field)
        # `datetime` subclasses `date`, so this covers `2009-01-01 06:00:00` too.
        if isinstance(value, date):
            rendered[field] = value.strftime(fmt)
    return rendered


class CatchmentConfig(BaseModel):
    """The `Catchment` constructor arguments.

    Attributes:
        name: Catchment name.
        start: Start date, parsed with `fmt`. Held as a string, because the constructor does
            the parsing; an unquoted YAML date arrives here already a `date` and is written
            back out in `fmt`, so both spellings work.
        end: End date, parsed with `fmt`. See `start`.
        fmt: `strptime` format for `start` / `end`.
        spatial_resolution: `"lumped"` or `"distributed"`. Selects the shape of `meteo` and
            `gauges`, and whether `flow_network` is required.
        temporal_resolution: `"daily"` or `"hourly"`.
        routing_method: `"muskingum"` or `"maxbas"`. Assigned onto `model.routing_method`, and
            constrains one other block: Muskingum routes along the network, so a distributed
            run needs `flow_network.flow_direction`. Left unwritten it is derived from
            `parameters.maxbas`, which describes the same choice from the parameter set's
            side; written, it must agree with it. Which `Run.*` entry point actually routes
            with it is still the caller's choice.
    """

    model_config = _STRICT

    name: str
    start: str
    end: str
    fmt: str = "%Y-%m-%d"
    spatial_resolution: Literal["lumped", "distributed"] = "lumped"
    temporal_resolution: Literal["daily", "hourly"] = "daily"
    routing_method: Literal["muskingum", "maxbas"] = "muskingum"

    @model_validator(mode="before")
    @classmethod
    def _accept_a_date_yaml_already_parsed(cls, values: Any) -> Any:
        """Render an unquoted YAML date back into `fmt` before the string fields see it.

        Args:
            values: The raw mapping.

        Returns:
            Any: The mapping, with `start` and `end` as strings.
        """
        return _write_dates_in_the_block_format(values, ("start", "end"))


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
        variable: Which variable to take from each file, `source="netcdf_files"` only. `None`
            takes the single variable a file holds, which is an error if it holds several.
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
    variable: str | None = None
    start: str | None = None
    end: str | None = None
    fmt: str = "%Y-%m-%d"
    glob: str = "*.tif"
    regex_string: str = r"\d{4}.\d{2}.\d{2}"
    file_name_data_fmt: str | None = None
    per_variable: dict[str, dict[str, Any]] | None = None
    gdal_env: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_a_date_yaml_already_parsed(cls, values: Any) -> Any:
        """Render an unquoted YAML date back into `fmt` before the string fields see it.

        Args:
            values: The raw mapping.

        Returns:
            Any: The mapping, with `start` and `end` as strings.
        """
        return _write_dates_in_the_block_format(values, ("start", "end"))


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
        maxbas: Whether the set carries the triangular-routing parameter. It describes the
            parameter set rather than the run, but `RunConfig` requires it to agree with
            `catchment.routing_method`: the two counts differ (11 against 12) and `maxbas`
            is what selects which is expected, so a disagreeing pair still passes the count
            check and then reads the wrong parameter as the routing one.
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
    model_config = ConfigDict(**_STRICT, protected_namespaces=())

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
        column: Gauge-table column naming the columns of the resulting hydrograph frame. It
            does not select the discharge file names -- `read_discharge_gauges` reads
            `<id>.csv` regardless -- so a table can label its hydrographs with human-readable
            names while the files stay named after the ids.
        delimiter: Discharge CSV delimiter.
        fmt: `strptime` format for the discharge CSV's date column.
        table_fmt: `strptime` format for the gauge table's optional `start` / `end` columns,
            which bound each gauge's validity period. A separate field because the table is a
            separate file that a separate hand may have written; `None` falls back to `fmt`,
            which is right whenever the two were written together.
    """

    model_config = _STRICT

    discharge: str
    table: str | None = None
    column: str = "id"
    delimiter: str = ","
    fmt: str = "%Y-%m-%d"
    table_fmt: str | None = None


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
        flow_network: The routing network. Required for a distributed run and refused for a
            lumped one, which has no grid to put it on.
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
    def _check_the_routing_method_matches_the_parameter_set(self) -> RunConfig:
        """Make `routing_method` agree with the parameter set, deriving it where it is unstated.

        The parameter-count check downstream cannot catch a disagreement: a MAXBAS set holds 11
        parameters and a Muskingum set 12, and `parameters.maxbas` is what selects which count
        is expected, so a set that contradicts the routing method still counts correctly. The
        run then completes, reading the Muskingum X as the MAXBAS value (or K and X out of a
        MAXBAS set), and produces a hydrograph that is quietly wrong.

        A lumped run picks its routing function at the call site rather than from this
        attribute, so `routing_method` is not load-bearing there -- but it is public, it is what
        `distrrm.SpatialRouting` keys off, and leaving it saying `Muskingum` on a run using a
        MAXBAS parameter set would mislead the next reader. An unstated one is therefore
        derived from the parameter set rather than left at its default.

        Returns:
            RunConfig: This config, with `catchment.routing_method` filled in where it was
                unstated and the parameter set says which it is.

        Raises:
            ValueError: `catchment.routing_method` and `parameters.maxbas` disagree.
        """
        if self.parameters is None:
            return self

        if "routing_method" not in self.catchment.model_fields_set:
            self.catchment.routing_method = "maxbas" if self.parameters.maxbas else "muskingum"
            return self

        if (self.catchment.routing_method == "maxbas") != self.parameters.maxbas:
            raise ValueError(
                f"catchment.routing_method is {self.catchment.routing_method!r} but "
                f"parameters.maxbas is {self.parameters.maxbas}; the parameter set and the "
                f"routing method must agree, or the run reads the wrong parameter as the "
                f"routing one"
            )
        return self

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
            _reject_fields_the_run_will_not_read(
                self.meteo,
                _METEO_FIELDS_BY_SOURCE[self.meteo.source],
                "meteo",
                f"meteo.source is {self.meteo.source!r}",
            )
        else:
            if self.meteo.path is None:
                raise ValueError(
                    "catchment.spatial_resolution is 'lumped', which needs meteo.path -- the "
                    "CSV of catchment-average drivers"
                )
            # `extra="forbid"` exists so a misspelled key fails rather than being dropped;
            # accepting a correctly spelled but inapplicable block would be the same silence
            # by another route. A lumped run has no grid, so neither block can be honoured.
            if self.flow_network is not None:
                raise ValueError(
                    "catchment.spatial_resolution is 'lumped', which has no grid, so a "
                    "flow_network block cannot be used"
                )
            if self.meteo.source != "rasters":
                raise ValueError(
                    f"catchment.spatial_resolution is 'lumped', which reads meteo.path as a "
                    f"CSV of catchment-average drivers; meteo.source "
                    f"{self.meteo.source!r} does not apply"
                )
            lumped = "catchment.spatial_resolution is 'lumped'"
            _reject_fields_the_run_will_not_read(
                self.meteo,
                _LUMPED_METEO_FIELDS,
                "meteo",
                f"{lumped}, which reads meteo.path as one CSV of catchment-average drivers",
            )
            if self.gauges is not None:
                _reject_fields_the_run_will_not_read(
                    self.gauges,
                    _LUMPED_GAUGES_FIELDS,
                    "gauges",
                    f"{lumped}, which reads one discharge file and locates no gauges",
                )
        return self

    @model_validator(mode="after")
    def _check_the_dates_parse_and_are_ordered(self) -> RunConfig:
        """Check every date against the format it is written in, and that the period runs.

        Returns:
            RunConfig: This config, unchanged.

        Raises:
            ValueError: A date does not match its format, or a period ends before it starts.
        """
        for label, value, fmt in (
            ("catchment.start", self.catchment.start, self.catchment.fmt),
            ("catchment.end", self.catchment.end, self.catchment.fmt),
            ("meteo.start", self.meteo.start, self.meteo.fmt),
            ("meteo.end", self.meteo.end, self.meteo.fmt),
        ):
            if value is None:
                continue
            try:
                datetime.strptime(value, fmt)
            except ValueError as error:
                raise ValueError(
                    f"{label} {value!r} does not match its format {fmt!r}: {error}"
                ) from error

        for first, second, fmt, block in (
            (self.catchment.start, self.catchment.end, self.catchment.fmt, "catchment"),
            (self.meteo.start, self.meteo.end, self.meteo.fmt, "meteo"),
        ):
            if first is None or second is None:
                continue
            if datetime.strptime(first, fmt) > datetime.strptime(second, fmt):
                raise ValueError(
                    f"{block}.start {first!r} is after {block}.end {second!r}"
                )
        return self
