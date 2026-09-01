"""Catchment module for the Hapi hydrological modeling framework.

This module provides the Catchment and Lake classes for reading
meteorological and spatial inputs, running distributed hydrological
models, extracting discharge, and saving results. The Catchment class
is the base class that reads all inputs required by the model
(rainfall, temperature, ET, flow accumulation, flow direction,
parameters, and gauge data). It supports both lumped and distributed
spatial modes with daily or hourly temporal resolutions.

The Lake class provides similar functionality for simulating a lake
as a lumped model using a rating curve, where the lake and its
upstream sub-catchments are treated as one lumped model.
"""

from __future__ import annotations

import datetime as dt
import inspect
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statista.descriptors as metrics
import yaml
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay
from loguru import logger
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube
from pyramids.feature import FeatureCollection

from hapi.conceptual import ConceptualModelSetup, ParameterBounds, ParameterSet
from hapi.config import RunConfig
from hapi.inputs import (
    METEO_VARIABLES,
    FlowNetwork,
    MeteoInputs,
    _warn_if_no_sentinel,
    read_rasters,
)
from hapi.period import SimulationPeriod
from hapi.results import SimulationResults
from hapi.rrm.hbv import HBV
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92

if TYPE_CHECKING:
    import matplotlib.animation

    from hapi.rrm.base_model import BaseConceptualModel

STATE_VARIABLES = ["SP", "SM", "UZ", "LZ", "WC"]
CONVERSION_FACTOR = (1000 * 24 * 60 * 60) / (1000**2)
#: (snow, maxbas) -> how many parameters the conceptual model reads in that configuration.
PARAMETER_COUNTS = {
    (True, True): 16,
    (False, True): 11,
    (True, False): 17,
    (False, False): 12,
}

#: Conceptual models `conceptual_model.model_class` can name in a YAML configuration.
#: `read_lumped_model` still takes any `type[BaseConceptualModel]`, so this only bounds what the
#: YAML shorthand can reach, not what the class accepts.
CONCEPTUAL_MODELS: dict[str, type[BaseConceptualModel]] = {
    "HBVBergestrom92": HBVBergestrom92,
    "HBV": HBV,
}

#: Accepted routing methods, canonicalised to one spelling.
#:
#: This records *which routing the parameter set was calibrated for*. It does not select the
#: routing -- the `Run.*` entry point does that -- but it is not decoration either:
#: `hapi.config` cross-checks it against `parameters.maxbas`, and that check is load-bearing.
#: A MAXBAS set holds 11 parameters and a Muskingum set 12, and `maxbas` is what decides which
#: count is expected, so a set contradicting the routing still passes the count check and the
#: run then reads the Muskingum X as the MAXBAS value -- a quietly wrong hydrograph.
#:
#: `"Kinematic"` is the kinematic-wave routing the flood model applies to river cells (see
#: `SaintVenant.KinematicRaster`, and the roadmap item in README). `Run.run_flood` reads it to
#: decide whether the Muskingum pass should leave those cells alone. It is *read by the entry
#: point*, not compared inside the routing loop -- that comparison used to run on every
#: distributed model, so a catchment declaring Kinematic and calling `run_distributed`
#: dereferenced a `bankfull_depth` of None and crashed partway through routing.
ROUTING_METHODS = {
    "muskingum": "Muskingum",
    "maxbas": "MAXBAS",
    "kinematic": "Kinematic",
}


def _resolve_config_paths(config: RunConfig, base: Path) -> None:
    """Rewrite every relative path in a configuration against the file it came from.

    A configuration names its inputs relative to itself, so it runs from any working directory
    and can be moved with the data it points at. Resolving against the process's working
    directory instead would make a file valid only from the one place it happened to be written
    for. An absolute path is left alone.

    `meteo`'s three driver fields are paths only for the folder and per-file sources; under
    `source="netcdf"` they name variables inside `meteo.path` and must not be touched.

    Args:
        config: The parsed configuration, rewritten in place.
        base: Directory of the configuration file.
    """

    def resolve(value: str | None) -> str | None:
        if value is None:
            return value
        return value if Path(value).is_absolute() else str((base / value).resolve())

    if config.meteo.source != "netcdf":
        for field in METEO_VARIABLES:
            setattr(config.meteo, field, resolve(getattr(config.meteo, field)))
    config.meteo.path = resolve(config.meteo.path)

    if config.flow_network is not None:
        # `flow_accumulation` and the two below are required fields, so `resolve` cannot
        # return None for them; the cast keeps that visible rather than widening the model.
        config.flow_network.flow_accumulation = str(
            resolve(config.flow_network.flow_accumulation)
        )
        config.flow_network.flow_direction = resolve(config.flow_network.flow_direction)
    if config.parameters is not None:
        config.parameters.path = str(resolve(config.parameters.path))
    if config.gauges is not None:
        config.gauges.table = resolve(config.gauges.table)
        config.gauges.discharge = str(resolve(config.gauges.discharge))
    if config.outputs is not None:
        config.outputs.results_dir = resolve(config.outputs.results_dir)


def _check_the_configured_paths_exist(config: RunConfig, distributed: bool) -> None:
    """Check every input path the run will open, before the first reader runs.

    The readers fail one at a time and in the order the build happens to call them, so a typo
    in the gauge table is only reported after the whole meteorological cube and the parameter
    folder have been read -- minutes, on a real grid, to learn about a line the file could have
    been checked for at once. Every missing path is named together instead, so one pass over
    the message fixes the file.

    Only the paths the chosen shape will actually open are checked, which is the same set the
    schema validated the configuration against.

    Args:
        config: The parsed configuration, with its paths already resolved.
        distributed: Whether this is a distributed run.

    Raises:
        FileNotFoundError: One or more configured paths do not exist.
    """
    candidates: list[tuple[str, str | None]] = []
    if config.parameters is not None:
        candidates.append(("parameters.path", config.parameters.path))

    if distributed:
        # Under `source="netcdf"` the three driver fields name variables inside `meteo.path`,
        # not paths, so only the file itself is checked.
        if config.meteo.source == "netcdf":
            candidates.append(("meteo.path", config.meteo.path))
        else:
            candidates += [
                (f"meteo.{name}", getattr(config.meteo, name))
                for name in METEO_VARIABLES
            ]
        if config.flow_network is not None:
            candidates += [
                (
                    "flow_network.flow_accumulation",
                    config.flow_network.flow_accumulation,
                ),
                ("flow_network.flow_direction", config.flow_network.flow_direction),
            ]
    else:
        candidates.append(("meteo.path", config.meteo.path))

    if config.gauges is not None:
        candidates += [
            ("gauges.discharge", config.gauges.discharge),
            ("gauges.table", config.gauges.table),
        ]

    missing = [
        f"{field} -> {value}"
        for field, value in candidates
        if value is not None and not Path(value).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "the run configuration names inputs that do not exist:\n  "
            + "\n  ".join(missing)
        )


@contextmanager
def _name_the_path(path) -> Iterator[None]:
    """Re-raise a pyramids `FileNotFoundError` with the offending path in the message.

    `DatasetCollection.from_files` reports "The path you have provided does
    not exist" / "is empty" without saying which path, where the checks this replaced
    named it. With several directories read per model run, the bare message does not
    identify the culprit.

    Args:
        path: The directory being read, echoed into the re-raised message.

    Yields:
        None: The wrapped read runs inside the context.

    Raises:
        FileNotFoundError: Re-raised from pyramids with `path` appended.
    """
    try:
        yield
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc} (path: {path})") from exc


class Catchment:
    """Catchment for reading meteorological/spatial inputs and running the model.

    The Catchment class includes methods to read the meteorological and
    spatial inputs of the distributed hydrological model. It also reads the
    data of the gauges. Build the catchment, then hand it to whichever
    :class:`hapi.run.Run` entry point suits it -- `Run.run_distributed(model)`. `Run` states what it
    needs as a protocol, which this class satisfies structurally; neither class inherits
    from the other.

    A run assigns its output to :attr:`results`. The result arrays are also readable under
    their historical names (`q_total`, `quz`, ...) as read-only properties forwarding to it.
    """

    def __init__(
        self,
        name: str,
        start_data: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        spatial_resolution: str = "Lumped",
        temporal_resolution: str = "Daily",
        routing_method: str = "Muskingum",
    ):
        """Initialize a Catchment instance.

        Args:
            name (str): Name of the Catchment.
            start_data (str): Starting date.
            end (str): End date.
            fmt (str, optional): Format of the given date.
                Default is "%Y-%m-%d".
            spatial_resolution (str, optional): "Lumped" or
                "Distributed". Default is "Lumped".
            temporal_resolution (str, optional): "Hourly" or "Daily".
                Default is "Daily".
            routing_method (str, optional): "Muskingum", "MAXBAS" or
                "Kinematic", matched case-insensitively and stored
                canonicalised. Default is "Muskingum".

        Raises:
            TypeError: If `spatial_resolution`, `temporal_resolution` or
                `routing_method` is not a string.
            ValueError: If `spatial_resolution` is not "lumped" or
                "distributed".
            ValueError: If `temporal_resolution` is not "daily" or
                "hourly".
            ValueError: If `routing_method` is not "Muskingum", "MAXBAS" or
                "Kinematic".
        """
        self.name = name

        for argument, value in (
            ("spatial_resolution", spatial_resolution),
            ("routing_method", routing_method),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{argument} must be a string, got {type(value).__name__}"
                )

        if spatial_resolution.lower() not in ["lumped", "distributed"]:
            raise ValueError(
                "available spatial resolutions are 'lumped' and 'distributed'"
            )
        self.spatial_resolution = spatial_resolution.lower()

        #: The span this model runs over. One object rather than six attributes: `start`,
        #: `end` and `temporal_resolution` are the inputs, and `date_index`, `dt` and
        #: `conversion_factor` are derived from them on read, so they cannot describe a
        #: different span from the one the model is set to. It validates the resolution and
        #: rejects a backwards span.
        self.period = SimulationPeriod.parse(
            start_data, end, fmt=fmt, temporal_resolution=temporal_resolution
        )

        # Canonicalised so the config cross-check against `parameters.maxbas` compares one
        # spelling. The routing loop no longer compares against it at all. Left
        # verbatim, a lower-case "muskingum" therefore routed every cell down the MAXBAS branch
        # and raised `TypeError: 'NoneType' object is not subscriptable`.
        if routing_method.lower() not in ROUTING_METHODS:
            raise ValueError(
                f"available routing methods are {', '.join(map(repr, ROUTING_METHODS))}, "
                f"got {routing_method!r}"
            )
        self.routing_method = ROUTING_METHODS[routing_method.lower()]
        #: The parameters and the `(snow, maxbas)` pair that fixes their width, as
        #: `read_parameters` produces them. Its constructor enforces the count rule, so every
        #: route to a parameter set is checked -- including the per-trial replacements a
        #: calibration makes. `None` until read.
        self.parameters: ParameterSet | None = None
        #: The conceptual model and the state it starts from, as `read_lumped_model`
        #: produces them. `None` until read.
        self.model_setup: ConceptualModelSetup | None = None
        self.data: np.ndarray | None = None
        #: The three meteorological drivers. Assign a :class:`~hapi.inputs.MeteoInputs`
        #: built by one of its loaders; everything meteorological hangs off it.
        self.meteo: MeteoInputs | None = None
        self.QGauges: pd.DataFrame | None = None
        self.GaugesTable: FeatureCollection | pd.DataFrame | None = None
        #: The search space a calibration explores, once read. `None` otherwise.
        self.bounds: ParameterBounds | None = None
        #: The routing network and the grid it defines. Assign a
        #: :class:`~hapi.inputs.FlowNetwork` built by its loader.
        self.flow_network: FlowNetwork | None = None
        self.flow_path_length_arr: np.ndarray | None = None
        self.DEM: np.ndarray | None = None
        self.bankfull_depth: np.ndarray | None = None
        self.river_width: np.ndarray | None = None
        self.river_roughness: np.ndarray | None = None
        self.flood_plain_roughness: np.ndarray | None = None
        #: Everything one run produced, replaced wholesale by the next run. The seven
        #: result arrays below are read-only properties forwarding to it, so `model.results.q_total`
        #: still reads as it always did while the run layer owns the arrays. `None` until
        #: a `Run.*` entry point has been called.
        self.results: SimulationResults | None = None
        self.anim: matplotlib.animation.FuncAnimation | None = None
        self._animation_glyph: ArrayGlyph | None = None
        self.Qsim: np.ndarray | None = None
        self.metrics: pd.DataFrame | None = None
        #: The configuration this model was built from, when it came from
        #: :meth:`from_yaml`; `None` for a model assembled by hand. Carries the blocks the
        #: build itself does not consume, such as `outputs`, so a caller need not restate a
        #: path the file already gives.
        self.config: RunConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Read a YAML run configuration and assemble a model from it.

        The alternate constructor for the build-then-mutate pattern this class documents: it
        constructs the model, assigns `meteo` and (distributed only) `flow_network`, then makes
        the `read_*` calls in the order they depend on each other -- the sequence a hand-written
        script's block of path assignments used to drive by hand.

        `hapi.config` only parses and validates; every assignment onto the model happens here.
        Running the model stays the caller's job, through whichever `Run.*` entry point suits
        `routing_method` and `spatial_resolution`.

        Builds `cls`, so `Calibration.from_yaml(...)` returns a `Calibration` -- it takes the
        same constructor arguments. `Run` is not a catchment at all and has nothing to build;
        `Run.from_yaml` exists only to say so and point here.

        Args:
            path: Path to the YAML file, as a string or a `Path`. See :mod:`hapi.config` for
                the schema.

        Returns:
            Self: The model, with every input read, parsed and assigned.

        Raises:
            FileNotFoundError: No file at `path`.
            yaml.YAMLError: The file is not valid YAML.
            pydantic.ValidationError: The file is missing a required field, carries an unknown
                one, or breaks one of the cross-field rules in :class:`hapi.config.RunConfig`.
            ValueError: The file is empty, or `conceptual_model.model_class` names a model
                that is not in `CONCEPTUAL_MODELS`.

        Examples:
            The configurations below ship with the Hapi repository, so these run from a
            checkout rather than an installed wheel; point at your own file to try them
            elsewhere.

            - Build a lumped model and inspect what the configuration gave it:
                ```python
                >>> from hapi.catchment import Catchment
                >>> model = Catchment.from_yaml(
                ...     "examples/hydrological-model/coello/run/coello-lumped-model-run.yaml"
                ... )
                >>> model.name
                'Coello'
                >>> model.spatial_resolution
                'lumped'
                >>> len(model.period.date_index)
                1095

                ```
            - Build a distributed model, whose drivers and routing network come from the
              `meteo` and `flow_network` blocks:
                ```python
                >>> from hapi.catchment import Catchment
                >>> model = Catchment.from_yaml(
                ...     "examples/hydrological-model/coello/run/"
                ...     "coello-distributed-model-run-netcdf.yaml"
                ... )
                >>> model.meteo.shape
                (13, 14, 10)
                >>> model.flow_network.rows, model.flow_network.cols
                (13, 14)
                >>> model.routing_method
                'Muskingum'

                ```
        """
        # Explicit encoding: without it the file is decoded with the locale codec, so a
        # non-ASCII catchment name or path mojibakes on a machine whose default is not UTF-8
        # -- and does so silently, since the corrupted text is still valid YAML.
        text = Path(path).read_text(encoding="utf-8")
        mapping = yaml.safe_load(text)
        # An empty file parses to None, which pydantic would report as the opaque
        # "Input should be a valid dictionary" without saying which file was empty.
        if mapping is None:
            raise ValueError(f"the run configuration at {path} is empty")
        config = RunConfig.model_validate(mapping)
        # Relative paths belong to the file, not to whatever directory the process happens to
        # be in, so a configuration runs from anywhere and travels with the data it names.
        _resolve_config_paths(config, Path(path).resolve().parent)
        catchment = config.catchment

        # The first three go positionally on purpose: `Catchment.__init__` calls its second
        # parameter `start_data` and `Calibration.__init__` calls it `start`, so naming them
        # would break `Calibration.from_yaml` -- which this method is documented to support --
        # while still working here. Renaming the parameter is the fix, and is breaking.
        model = cls(
            catchment.name,
            catchment.start,
            catchment.end,
            fmt=catchment.fmt,
            spatial_resolution=catchment.spatial_resolution,
            temporal_resolution=catchment.temporal_resolution,
            routing_method=catchment.routing_method,
        )

        # Resolved before any reader runs: it needs nothing but the config, and a typo here
        # would otherwise cost the whole parameter folder read before failing.
        conceptual_model = config.conceptual_model
        if conceptual_model.model_class not in CONCEPTUAL_MODELS:
            raise ValueError(
                f"conceptual_model.model_class {conceptual_model.model_class!r} is not "
                f"registered; known models are {sorted(CONCEPTUAL_MODELS)}"
            )
        model_class = CONCEPTUAL_MODELS[conceptual_model.model_class]

        distributed = catchment.spatial_resolution == "distributed"
        _check_the_configured_paths_exist(config, distributed)
        if distributed:
            model.meteo = MeteoInputs.from_config(
                config.meteo,
                start=catchment.start,
                end=catchment.end,
                fmt=catchment.fmt,
            )
            model.flow_network = FlowNetwork.from_rasters(
                config.flow_network.flow_accumulation,
                config.flow_network.flow_direction,
            )
        else:
            model.read_lumped_inputs(config.meteo.path)

        # A calibration derives its parameters from the bounds `read_parameters_bound` is
        # given rather than reading a fitted set, so the block is optional.
        if config.parameters is not None:
            model.read_parameters(
                config.parameters.path,
                config.parameters.snow,
                maxbas=config.parameters.maxbas,
            )

        model.read_lumped_model(
            model_class,
            conceptual_model.catchment_area,
            conceptual_model.initial_condition,
            conceptual_model.q_init,
        )

        # Equally optional: a run that is not scored against observations has no gauges.
        gauges = config.gauges
        if gauges is not None:
            if distributed:
                # The table's validity-period columns and the discharge files' index are two
                # different files' date layouts, so they get two fields -- with the table
                # falling back to the discharge format, which is right whenever one hand wrote
                # both.
                model.read_gauge_table(
                    gauges.table,
                    config.flow_network.flow_accumulation,
                    fmt=gauges.table_fmt or gauges.fmt,
                )
            model.read_discharge_gauges(
                gauges.discharge,
                delimiter=gauges.delimiter,
                column=gauges.column,
                fmt=gauges.fmt,
            )

        # Kept so the blocks the build does not itself consume stay reachable -- `outputs`
        # above all, which describes where results go rather than what the model reads.
        model.config = config
        return model

    def read_flow_path_length(self, path: str):
        """Read the flow path length raster.

        Reads the flow path length raster into `flow_path_length_arr`. The grid it sits
        on belongs to :class:`~hapi.inputs.FlowNetwork`, so this reader no longer derives
        rows, columns, the no-data value or the domain count from a second raster.

        No-data handling is delegated to pyramids via `read_array(masked=True)`, so
        cells outside the catchment become `NaN`. The array is promoted to floating point so masked cells can
        hold `NaN`.

        Args:
            path (str | Path): Path to the flow path length raster. Any raster format
                GDAL can open is accepted, not only GeoTIFF.

        Raises:
            FileNotFoundError: The path does not exist.
            TypeError: `path` is neither a string nor a `Path`.
            RuntimeError: GDAL cannot open the file as a raster.

        Examples:
            - Read a small path-length raster; the one no-data cell is excluded from the
              domain count:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "fpl.tif")
                >>> Dataset.create_from_array(
                ...     np.array([[10, 20], [30, -9999]], dtype="int32"),
                ...     top_left_corner=(0.0, 8000.0), cell_size=4000.0, epsg=32618,
                ...     no_data_value=-9999, path=path,
                ... ).close()
                >>> model = Catchment("example", "2000-01-01", "2000-01-02",
                ...                   spatial_resolution="Distributed")
                >>> model.read_flow_path_length(path)
                >>> int(np.count_nonzero(~np.isnan(model.flow_path_length_arr)))
                3
                >>> float(model.flow_path_length_arr[0, 1])
                20.0

                ```
            - A real length within 0.1% of the sentinel is kept, so every cell counts:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "fpl_near.tif")
                >>> Dataset.create_from_array(
                ...     np.array([[10, 20], [30, -9990]], dtype="int32"),
                ...     top_left_corner=(0.0, 8000.0), cell_size=4000.0, epsg=32618,
                ...     no_data_value=-9999, path=path,
                ... ).close()
                >>> model = Catchment("example", "2000-01-01", "2000-01-02",
                ...                   spatial_resolution="Distributed")
                >>> model.read_flow_path_length(path)
                >>> int(np.count_nonzero(~np.isnan(model.flow_path_length_arr)))
                4

                ```

        See Also:
            hapi.inputs.FlowNetwork: Holds the matching flow-accumulation raster and the grid.
        """
        # Path validation is delegated to pyramids: a missing path raises
        # FileNotFoundError, a non-path argument TypeError, and an unreadable file a
        # GDAL RuntimeError. Unlike the asserts these replace, they survive `python -O`.
        fpl = Dataset.read_file(path)
        # No-data masking is delegated to pyramids (see FlowNetwork.from_rasters). The grid
        # itself comes from the flow network, so this reader no longer redefines rows, cols,
        # no_data_value or no_elem from a second raster.
        self.flow_path_length_arr = np.ma.filled(
            fpl.read_array(band=0, masked=True).astype(float), np.nan
        )
        _warn_if_no_sentinel(fpl, "flow path length")

        logger.debug("Flow path length input is read successfully")

    def read_river_geometry(
        self,
        dem_file: str,
        bankfull_depth_file: str,
        river_width_file: str,
        river_roughness_file: str,
        floodplain_roughness_file: str,
    ):
        """Read river geometry rasters for hydraulic routing.

        Reads the DEM, bankfull depth, river width, river roughness,
        and floodplain roughness rasters required for hydraulic
        routing computations.

        Args:
            dem_file (str): Path to the DEM raster file.
            bankfull_depth_file (str): Path to the bankfull depth
                raster file.
            river_width_file (str): Path to the river width raster
                file.
            river_roughness_file (str): Path to the river roughness
                raster file.
            floodplain_roughness_file (str): Path to the floodplain
                roughness raster file.
        """
        for name, fpath in [
            ("DEM", dem_file),
            ("bankfull_depth", bankfull_depth_file),
            ("river_width", river_width_file),
            ("river_roughness", river_roughness_file),
            ("flood_plain_roughness", floodplain_roughness_file),
        ]:
            ds = Dataset.read_file(fpath)
            setattr(self, name, ds.read_array(band=0))

    def read_parameters(self, path: str, snow: bool = False, maxbas: bool = False):
        """Read model parameter rasters or a CSV parameter file.

        For distributed mode, reads parameter rasters from a folder.
        For lumped mode, reads parameters from a CSV file.

        Args:
            path (str): Path to the folder containing parameter
                rasters (distributed mode) or to a CSV file (lumped
                mode).
            snow (bool, optional): Whether to simulate snow
                processes. If True, snow-related parameters must be
                provided. Default is False.
            maxbas (bool, optional): True if the routing method is
                Maxbas. Default is False.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If `snow` is not a boolean or if the number
                of parameters does not match the expected count for
                the given snow/maxbas configuration.
        """
        if self.spatial_resolution.lower() == "distributed":
            # Path validation is delegated to pyramids: from_files raises
            # FileNotFoundError for a missing *or* empty directory. Unlike the asserts
            # these replace, that survives `python -O`. Its message does not name the
            # offending directory, so _name_the_path re-raises with it.
            with _name_the_path(path):
                cube = read_rasters(path, regex_string=r"\d+", date=False)
            parameters = np.moveaxis(cube.values, 0, -1)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "The parameter file you have entered does not exist"
                )

            parameters = pd.read_csv(path, index_col=0, header=None)[1].tolist()

        if not (not snow or snow):
            raise ValueError(
                "snow input defines whether to consider snow subroutine or not it has to be True or False"
            )

        # The count check lives in `ParameterSet.__post_init__`, so it runs on every route
        # to a parameter set rather than only on this one.
        self.parameters = ParameterSet(parameters, snow=snow, maxbas=maxbas)

        logger.debug("Parameters are read successfully")

    def read_lumped_model(
        self,
        lumped_model: type[BaseConceptualModel],
        catchment_area: float | int,
        initial_condition: list,
        q_init=None,
    ):
        """Read and set up a lumped conceptual model.

        Args:
            lumped_model: A `BaseConceptualModel` subclass (the class
                itself, not an instance), e.g. `HBVBergestrom92`. It is
                instantiated and stored on `LumpedModel`.
            catchment_area (float | int): Catchment area in
                km2.
            initial_condition (list): List of 5 initial condition
                values: [SnowPack, SoilMoisture, Upper Zone,
                Lower Zone, Water Content].
            q_init (float, optional): Initial discharge. Default is
                None.

        Raises:
            TypeError: If `initial_condition` is not a list, or if
                `q_init` is given and is not a float.
            ValueError: If `lumped_model` is not a class or if
                `initial_condition` does not contain exactly 5
                values.
        """
        if not inspect.isclass(lumped_model):
            raise ValueError(
                "ConceptualModel should be a module or a python file contains functions "
            )

        # The checks on `initial_condition` and `q_init` live in
        # `ConceptualModelSetup.__post_init__` now.
        self.model_setup = ConceptualModelSetup(
            lumped_model(), catchment_area, initial_condition, q_init
        )

        logger.debug("Lumped model is read successfully")

    def read_lumped_inputs(self, path: str):
        """Read meteorological inputs for lumped mode.

        The lumped counterpart of :class:`~hapi.inputs.MeteoInputs`, which carries the
        distributed drivers: the lumped model works on one column per variable rather than a
        grid, and `Wrapper.run_lumped` reads the long-term average straight out of the fourth
        column.

        A three-column file is completed with a fourth holding the record's mean temperature.
        `Wrapper.run_lumped` reads that column unconditionally, so without it a file this method
        accepts raises `IndexError` in the middle of the run instead.

        Args:
            path (str): Path to the input CSV file. Data columns must
                be in the order [date, precipitation, ET, Temp], optionally
                followed by the long-term average temperature.

        Raises:
            ValueError: If the input data does not have 3 or 4
                columns (excluding the date index).
        """
        self.data = pd.read_csv(path, header=0, delimiter=",", index_col=0)
        self.data = self.data.values

        columns = np.shape(self.data)[1]
        if columns not in (3, 4):
            raise ValueError(
                "meteorological data should be of length at least 3 (prec, ET, temp) or 4(prec, ET, temp, tm) "
            )

        if columns == 3:
            # The long-term average the snow routine compares each step against. Derived from
            # the temperature column, as the reader this replaced did.
            long_term_average = np.full(
                (np.shape(self.data)[0], 1), self.data[:, 2].mean()
            )
            self.data = np.hstack([self.data, long_term_average])

        logger.debug("Lumped Model inputs are read successfully")

    def read_gauge_table(
        self, path: str, flow_acc_file: str = "", fmt: str = "%Y-%m-%d"
    ):
        """Read the gauge table listing gauge locations and properties.

        Reads gauge data including coordinates (x, y), area ratio, and
        weight. The coordinates are mandatory to locate the gauges and
        extract discharge at the corresponding cells.

        The result lands on :attr:`GaugesTable`, and its type follows the input format:

        * `.geojson` is read with
          :meth:`pyramids.feature.FeatureCollection.read_file`, giving a
          :class:`~pyramids.feature.FeatureCollection` — a `GeoDataFrame` subclass, so
          it keeps its geometry column and CRS.
        * anything else is read with :func:`pandas.read_csv`, giving a plain
          :class:`~pandas.DataFrame` with no geometry.

        When `flow_acc_file` is given and the table has no `cell_row` column, each
        gauge is mapped onto the raster grid and `cell_row` / `cell_col` columns are
        appended.

        `start` and `end` columns, if present, are parsed with `fmt` into
        `datetime64` columns. The two are handled independently, so a table carrying
        only one of them is fine.

        Args:
            path (str): Path to the gauge file (CSV or GeoJSON).
            flow_acc_file (str, optional): Path to the flow
                accumulation raster used to map gauge coordinates to
                array indices. Default is "".
            fmt (str, optional): Date format for start/end columns
                in the gauge table. Default is "%Y-%m-%d".

        Raises:
            ValueError: A `start` or `end` value does not match `fmt`.

        Examples:
            - Read a GeoJSON gauge file and inspect the loaded stations:
                ```python
                >>> import os, tempfile
                >>> from pyramids.feature import FeatureCollection
                >>> from shapely.geometry import Point
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "gauges.geojson")
                >>> FeatureCollection(
                ...     {"id": [1, 2], "name": ["Station 1", "Station 2"]},
                ...     geometry=[Point(454795.7, 503143.3), Point(443847.6, 481850.7)],
                ...     crs="EPSG:32618",
                ... ).to_file(path, driver="GeoJSON")
                >>> model = Catchment("coello", "2009-01-01", "2009-01-10",
                ...                   spatial_resolution="Distributed")
                >>> model.read_gauge_table(path)
                >>> model.GaugesTable["name"].tolist()
                ['Station 1', 'Station 2']
                >>> model.GaugesTable.crs.to_epsg()
                32618

                ```
            - A CSV gauge table loads as a plain frame with no geometry:
                ```python
                >>> import os, tempfile
                >>> import pandas as pd
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "gauges.csv")
                >>> pd.DataFrame({"id": [1], "name": ["Station 1"]}).to_csv(path, index=False)
                >>> model = Catchment("coello", "2009-01-01", "2009-01-10",
                ...                   spatial_resolution="Distributed")
                >>> model.read_gauge_table(path)
                >>> model.GaugesTable["id"].tolist()
                [1]
                >>> hasattr(model.GaugesTable, "crs")
                False

                ```
            - A validity period is parsed into datetime columns using `fmt`:
                ```python
                >>> import os, tempfile
                >>> import pandas as pd
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "gauges.csv")
                >>> pd.DataFrame(
                ...     {"id": [1], "start": ["03/04/2009"], "end": ["05/06/2011"]}
                ... ).to_csv(path, index=False)
                >>> model = Catchment("coello", "2009-01-01", "2009-01-10",
                ...                   spatial_resolution="Distributed")
                >>> model.read_gauge_table(path, fmt="%d/%m/%Y")
                >>> model.GaugesTable.loc[0, "start"].strftime("%d %B %Y")
                '03 April 2009'

                ```

        See Also:
            Catchment.read_discharge_gauges: Read the observed discharge series per gauge.
        """
        # read the gauge table
        if path.endswith(".geojson"):
            # FeatureCollection is-a GeoDataFrame, so every downstream consumer
            # (.loc, .columns, map_to_array_coordinates) is unaffected. The old
            # `driver="GeoJSON"` was a write-time option that pyogrio warned about
            # and ignored on read, so it is dropped.
            self.GaugesTable = FeatureCollection.read_file(path)
        else:
            self.GaugesTable = pd.read_csv(path)
        col_list = self.GaugesTable.columns.tolist()

        # Convert whole columns rather than assigning per cell: pandas 3 string columns
        # reject an in-place datetime write, and each column is handled independently so
        # a table carrying only one of the two does not raise KeyError on the other.
        for column in ("start", "end"):
            if column in col_list:
                parsed = pd.to_datetime(self.GaugesTable[column], format=fmt)
                # to_datetime maps a blank or missing cell to NaT rather than raising,
                # where the per-cell strptime this replaced rejected it. A gauge with no
                # validity period is almost always a data-entry slip, and silently
                # carrying NaT into the period comparisons hides it.
                if parsed.isna().any():
                    bad = self.GaugesTable.index[parsed.isna()].tolist()
                    raise ValueError(
                        f"the {column!r} column has no usable date at row(s) {bad}; "
                        f"every gauge needs a {column} parseable with {fmt!r}, or the "
                        "column should be omitted entirely."
                    )
                self.GaugesTable[column] = parsed
        if flow_acc_file != "" and "cell_row" not in col_list:
            # if hasattr(self, 'flow_acc'):
            # calculate the nearest cell to each station
            dataset = Dataset.read_file(flow_acc_file)
            loc_arr = dataset.map_to_array_coordinates(self.GaugesTable)
            self.GaugesTable.loc[:, ["cell_row", "cell_col"]] = loc_arr

        logger.debug("Gauge Table is read successfully")

    def read_discharge_gauges(
        self,
        path: str,
        delimiter: str = ",",
        column: str = "id",
        fmt: str = "%Y-%m-%d",
        split: bool = False,
        start_date: str | dt.datetime = "",
        end_date: str | dt.datetime = "",
        readfrom: str = "",
    ):
        """Read gauge discharge data from CSV files.

        For distributed mode, each gauge's discharge must be stored in a
        separate CSV file. File names must match the "id" column in the
        gauge table (read via `read_gauge_table`). For lumped mode, a
        single CSV file with the discharge data is expected.

        Args:
            path (str): Path to the gauge discharge data directory
                (distributed) or file (lumped).
            delimiter (str, optional): Delimiter between the date and
                the discharge column. Default is ",".
            column (str, optional): Gauge-table column naming the columns of the
                resulting `QGauges` frame. It does not select the file names --
                those always come from the "id" column. Default is "id".
            fmt (str, optional): Date format in the discharge files.
                Default is "%Y-%m-%d".
            split (bool, optional): True to subset the data between
                `start_date` and `end_date`. Default is False.
            start_date (str | dt.datetime, optional): Start date for
                subsetting. A string is parsed with `fmt`; a datetime is
                used as it is.
                Default is "".
            end_date (str | dt.datetime, optional): End date for
                subsetting. See `start_date`.
                Default is "".
            readfrom (str, optional): Number of rows to skip when
                reading the CSV. Default is "".

        Raises:
            FileNotFoundError: If the discharge file does not exist
                (lumped mode).
            ValueError: If the gauge table has not been read yet
                (distributed mode).
        """
        if self.period.temporal_resolution.lower() == "daily":
            ind = pd.date_range(self.period.start, self.period.end, freq="D")
        else:
            ind = pd.date_range(self.period.start, self.period.end, freq="h")

        if self.spatial_resolution.lower() == "distributed":
            self._read_one_discharge_file_per_gauge(
                path, ind, delimiter, column, fmt, readfrom
            )
        else:
            self._read_the_single_discharge_file(path, ind, delimiter, fmt)

        if split:
            if isinstance(start_date, str):
                start_date = dt.datetime.strptime(start_date, fmt)
            if isinstance(end_date, str):
                end_date = dt.datetime.strptime(end_date, fmt)
            self.QGauges = self.QGauges.loc[start_date:end_date]

        logger.debug("Gauges data are read successfully")

    def _read_one_discharge_file_per_gauge(
        self,
        path: str,
        index: pd.DatetimeIndex,
        delimiter: str,
        column: str,
        fmt: str,
        readfrom: str,
    ) -> None:
        """Fill `QGauges` from a folder holding one CSV per gauge id.

        Args:
            path: Folder of per-gauge CSVs, each named after a gauge id.
            index: The model's date index, which the frame is built on.
            delimiter: Discharge CSV delimiter.
            column: Gauge-table column naming the frame's columns.
            fmt: `strptime` format for each file's date column.
            readfrom: Rows to skip, or "" to read from the header.

        Raises:
            ValueError: `read_gauge_table` has not been called yet.
        """
        # `__init__` sets GaugesTable to None, so the `hasattr` this replaced was always
        # true and never guarded anything: a caller who skipped `read_gauge_table` got a
        # `TypeError` on None a few lines down instead.
        if self.GaugesTable is None:
            raise ValueError(
                "the gauge table has not been read yet; call read_gauge_table before "
                "read_discharge_gauges in distributed mode"
            )

        # The frame is labelled from `column` but every file is named after `id`, so the
        # two are tracked separately: filling by `int(name)` instead of by the label the
        # frame was built with left a `column != "id"` table with the requested columns
        # all-NaN and a second set of id-named ones beside them, silently.
        labels = self.GaugesTable[column].tolist()
        self.QGauges = pd.DataFrame(index=index, columns=labels)

        for i in range(len(self.GaugesTable)):
            name = self.GaugesTable.loc[i, "id"]
            if readfrom != "":
                f = pd.read_csv(
                    f"{path}/{name}.csv",
                    index_col=0,
                    delimiter=delimiter,
                    skiprows=readfrom,
                )
            else:
                f = pd.read_csv(
                    f"{path}/{name}.csv",
                    header=0,
                    index_col=0,
                    delimiter=delimiter,
                )

            f.index = [dt.datetime.strptime(i, fmt) for i in f.index.tolist()]
            self.QGauges[labels[i]] = f.loc[
                self.period.start : self.period.end, f.columns[-1]
            ]

    def _read_the_single_discharge_file(
        self, path: str, index: pd.DatetimeIndex, delimiter: str, fmt: str
    ) -> None:
        """Fill `QGauges` from one CSV, the lumped case.

        A lumped run has no grid to locate gauges on, so there is one hydrograph and no
        gauge table: the frame takes the file's own first column as its only column.

        Args:
            path: The discharge CSV.
            index: The model's date index, which the frame is built on.
            delimiter: Discharge CSV delimiter.
            fmt: `strptime` format for the file's date column.

        Raises:
            FileNotFoundError: `path` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file you have entered{path} does not exist")

        self.QGauges = pd.DataFrame(index=index)
        f = pd.read_csv(path, header=0, index_col=0, delimiter=delimiter)
        f.index = [dt.datetime.strptime(i, fmt) for i in f.index.tolist()]
        self.QGauges[f.columns[0]] = f.loc[
            self.period.start : self.period.end, f.columns[0]
        ]

    def read_parameters_bound(
        self,
        upper_bound: list | np.ndarray,
        lower_bound: list | np.ndarray,
        snow: bool = False,
        maxbas: bool = False,
    ):
        """Read the lower and upper parameter bounds for calibration.

        Args:
            upper_bound (list | np.ndarray): Upper bound values
                for each parameter.
            lower_bound (list | np.ndarray): Lower bound values
                for each parameter.
            snow (bool, optional): Whether to simulate snow
                processes. If True, snow-related parameters must be
                bounded. Default is False.
            maxbas (bool, optional): True if the parameters include
                maxbas. Default is False.

        Raises:
            ValueError: If the lengths of `upper_bound` and
                `lower_bound` are not equal.
            ValueError: If `snow` is not a boolean.
        """
        if not isinstance(snow, bool):
            raise ValueError(
                " snow input defines whether to consider snow subroutine or not it has to be True or False"
            )
        # A calibration reads no parameter file, so the bounds are where `(snow, maxbas)`
        # enters -- carried here so every trial vector can be checked against it.
        self.bounds = ParameterBounds(
            lower_bound, upper_bound, snow=snow, maxbas=maxbas
        )

        logger.debug("Parameters' bounds are read successfully")

    def extract_discharge(self, calculate_metrics=True, factor=None):
        """Extract and sum discharge at gauge locations.

        Which hydrograph is the right one depends on how the run was routed, and the results
        say so, so nothing has to be passed in. Under Muskingum the discharge accumulates
        downstream, so each gauge is read from its own cell of `q_total`. Under MAXBAS every
        cell is routed straight to the outlet, making a cell that cell's *contribution*; the
        hydrograph is then the basin-wide sum the run already computed into `qout`.

        This used to be a `frame_work_1` flag the caller had to set to match the entry point
        they had called, with a `ValueError` when they got it wrong. The routing is a
        property of the arrays, so it is read off them instead.

        Optionally computes performance metrics (RMSE, NSE, NSEhf, KGE, WB, Pearson-CC, R2)
        between the simulated and observed hydrographs.

        Args:
            calculate_metrics (bool, optional): Whether to calculate
                performance metrics. Default is True.
            factor (list, optional): List of multiplication factors
                for simulated discharge at each gauge. Must have the
                same length as the number of gauges. Applied only on the
                per-gauge (Muskingum) path. Default is None.

        Raises:
            ValueError: The gauge table has not been read, or the model has not been run.
        """
        if self.GaugesTable is None:
            raise ValueError("please read the gauges' table first.")
        if self.results is None:
            raise ValueError(
                "there are no results to extract; run the model first, e.g. "
                "Run.run_distributed(model)"
            )

        if self.results.outlet_shortcut_valid:
            self.Qsim = pd.DataFrame(
                index=self.period.date_index, columns=self.QGauges.columns
            )
            if calculate_metrics:
                index = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
                self.metrics = pd.DataFrame(index=index, columns=self.QGauges.columns)
            # sum the lower zone and the upper zone discharge
            outlet_x = self.flow_network.outlet[0][0]
            outlet_y = self.flow_network.outlet[1][0]

            # Muskingum accumulates downstream, so the outlet cell of `q_total` is the
            # outlet hydrograph. The engine cannot set this itself: finding the outlet
            # needs the gauge table, which is an analysis input, not a run input.
            self.results.qout = self.results.q_total[outlet_x, outlet_y, :]

            for i in range(len(self.GaugesTable)):
                x_ind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_row"])
                y_ind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_col"])
                gauge_id = self.GaugesTable.loc[self.GaugesTable.index[i], "id"]

                # Quz = np.reshape(self.results.quz_routed[x_ind,y_ind,:-1],self.TS-1)
                # Qlz = np.reshape(self.results.qlz_translated[x_ind,y_ind,:-1],self.TS-1)
                # q_sim = Quz + Qlz

                q_sim = np.reshape(
                    self.results.q_total[x_ind, y_ind, :-1], self.meteo.time_steps
                )
                if factor is not None:
                    self.Qsim.loc[:, gauge_id] = q_sim * factor[i]
                else:
                    self.Qsim.loc[:, gauge_id] = q_sim

                if calculate_metrics:
                    q_obs = self.QGauges.loc[:, gauge_id]
                    self.metrics.loc["RMSE", gauge_id] = round(
                        metrics.rmse(q_obs, q_sim), 3
                    )
                    self.metrics.loc["NSE", gauge_id] = round(
                        metrics.nse(q_obs, q_sim), 3
                    )
                    self.metrics.loc["NSEhf", gauge_id] = round(
                        metrics.nse_hf(q_obs, q_sim), 3
                    )
                    self.metrics.loc["KGE", gauge_id] = round(
                        metrics.kge(q_obs, q_sim), 3
                    )
                    self.metrics.loc["WB", gauge_id] = round(
                        metrics.wb(q_obs, q_sim), 3
                    )
                    self.metrics.loc["Pearson-CC", gauge_id] = round(
                        metrics.pearson_corr_coeff(q_obs, q_sim), 3
                    )
                    self.metrics.loc["R2", gauge_id] = round(
                        metrics.r2(q_obs, q_sim), 3
                    )
        else:
            # MAXBAS: a cell of `q_total` is a contribution, so the hydrograph is the
            # basin-wide sum the run already put in `qout`.
            self.Qsim = pd.DataFrame(index=self.period.date_index)
            gauge_id = self.GaugesTable.loc[self.GaugesTable.index[-1], "id"]
            q_sim = np.reshape(self.results.qout, self.meteo.time_steps)
            self.Qsim.loc[:, gauge_id] = q_sim

            if calculate_metrics:
                index = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
                self.metrics = pd.DataFrame(index=index)

                # if CalculateMetrics:
                q_obs = self.QGauges.loc[:, gauge_id]
                self.metrics.loc["RMSE", gauge_id] = round(
                    metrics.rmse(q_obs, q_sim), 3
                )
                self.metrics.loc["NSE", gauge_id] = round(metrics.nse(q_obs, q_sim), 3)
                self.metrics.loc["NSEhf", gauge_id] = round(
                    metrics.nse_hf(q_obs, q_sim), 3
                )
                self.metrics.loc["KGE", gauge_id] = round(metrics.kge(q_obs, q_sim), 3)
                self.metrics.loc["WB", gauge_id] = round(metrics.wb(q_obs, q_sim), 3)
                self.metrics.loc["Pearson-CC", gauge_id] = round(
                    metrics.pearson_corr_coeff(q_obs, q_sim), 3
                )
                self.metrics.loc["R2", gauge_id] = round(metrics.r2(q_obs, q_sim), 3)

    def plot_hydrograph(
        self,
        start_date: str | dt.datetime,
        end_date: str | dt.datetime,
        gauge: int,
        hapi_color: tuple | str = "#004c99",
        gauge_color: tuple | str = "#DC143C",
        line_width: int = 3,
        hapi_order: int = 1,
        gauge_order: int = 0,
        label_font_size: int = 10,
        x_major_fmt: str | dates.DateFormatter = "%Y-%m-%d",
        n_ticks: int = 5,
        title: str = "",
        x_axis_fmt: str = "%d\n%m",
        label: str = "",
        fmt: str = "%Y-%m-%d",
    ):
        r"""Plot simulated and observed hydrographs for a given gauge.

        Args:
            start_date (str | dt.datetime): Starting date for the plot. A
                string is parsed with `fmt`; a datetime is used as it is.
            end_date (str | dt.datetime): End date for the plot. See
                `start_date`.
            gauge (int): Index of the gauge in the GaugesTable.
            hapi_color (tuple | str, optional): Color of the
                simulated hydrograph. Default is "#004c99".
            gauge_color (tuple | str, optional): Color of the
                observed gauge hydrograph. Default is "#DC143C".
            line_width (int, optional): Line width for the
                hydrographs. Default is 3.
            hapi_order (int, optional): Z-order of the simulated
                hydrograph to control layering. Default is 1.
            gauge_order (int, optional): Z-order of the observed
                hydrograph to control layering. Default is 0.
            label_font_size (int, optional): Font size for axis tick
                labels. Default is 10.
            x_major_fmt (str, optional): Format for x-axis major
                tick labels. Default is "%Y-%m-%d".
            n_ticks (int, optional): Maximum number of x-axis ticks.
                Default is 5.
            title (str, optional): Title of the plot. Default is "".
            x_axis_fmt (str, optional): Format for x-axis minor
                tick labels. Default is "%d\n%m".
            label (str, optional): Label for the simulated
                hydrograph in the legend. Default is "".
            fmt (str, optional): Date format for parsing
                `start_date` and `end_date`. Default is "%Y-%m-%d".

        Returns:
            tuple: A tuple of (fig, ax) where fig is the matplotlib
                Figure and ax is the matplotlib Axes object.
        """
        if isinstance(start_date, str):
            start_date = dt.datetime.strptime(start_date, fmt)
        if isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, fmt)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))

        if self.spatial_resolution == "distributed":
            gauge_id = self.GaugesTable.loc[gauge, "id"]

            if title == "":
                title = "Gauge - " + str(self.GaugesTable.loc[gauge, "name"])

            if label == "":
                label = str(self.GaugesTable.loc[gauge, "name"])

            ax.plot(
                self.Qsim.loc[start_date:end_date, gauge_id],
                "-.",
                label=label,
                linewidth=line_width,
                color=hapi_color,
                zorder=hapi_order,
            )
            ax.set_title(title, fontsize=20)
        else:
            gauge_id = self.QGauges.columns[0]
            if title == "":
                title = "Gauge - " + str(gauge_id)
            if label == "":
                label = str(gauge_id)

            ax.plot(
                self.Qsim.loc[start_date:end_date, gauge_id],
                "-.",
                label=title,
                linewidth=line_width,
                color=hapi_color,
                zorder=hapi_order,
            )
            ax.set_title(title, fontsize=20)

        ax.plot(
            self.QGauges.loc[start_date:end_date, gauge_id],
            label="Gauge",
            linewidth=line_width,
            color=gauge_color,
            zorder=gauge_order,
        )

        ax.tick_params(axis="both", which="major", labelsize=label_font_size)
        # ax.locator_params(axis="x", nbins=4)

        x_major_fmt = dates.DateFormatter(x_major_fmt)
        ax.xaxis.set_major_formatter(x_major_fmt)
        # ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1),
        # interval=1))

        ax.xaxis.set_minor_formatter(dates.DateFormatter(x_axis_fmt))

        ax.xaxis.set_major_locator(plt.MaxNLocator(n_ticks))

        ax.legend(fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Discharge m3/s", fontsize=12)
        plt.tight_layout()

        if self.metrics is not None and not self.metrics.empty:
            logger.debug("----------------------------------")
            logger.debug("Gauge - " + str(gauge_id))
            logger.debug("RMSE= " + str(round(self.metrics.loc["RMSE", gauge_id], 2)))
            logger.debug("NSE= " + str(round(self.metrics.loc["NSE", gauge_id], 2)))
            logger.debug("NSEhf= " + str(round(self.metrics.loc["NSEhf", gauge_id], 2)))
            logger.debug("KGE= " + str(round(self.metrics.loc["KGE", gauge_id], 2)))
            logger.debug("WB= " + str(round(self.metrics.loc["WB", gauge_id], 2)))
            logger.debug(
                "Pearson-CC= " + str(round(self.metrics.loc["Pearson-CC", gauge_id], 2))
            )
            logger.debug("R2= " + str(round(self.metrics.loc["R2", gauge_id], 2)))

        return fig, ax

    def plot_distributed_results(
        self,
        start: str | dt.datetime,
        end: str | dt.datetime,
        fmt: str = "%Y-%m-%d",
        option: int = 1,
        gauges: bool = False,
        **kwargs: Any,
    ):
        """Animate distributed model results or meteorological inputs.

        Creates an animation of the time series of meteorological inputs
        or model results (discharge, state variables) over the spatial
        domain. Cells outside the catchment domain are masked on a copy of
        the data, so the model arrays stored on the instance are never
        modified. The animation title defaults to the selected variable's
        name; an explicit `title=` keyword argument overrides it.

        Args:
            start (str): Starting date for the animation.
            end (str): End date for the animation.
            fmt (str, optional): Format of the given date. Default
                is "%Y-%m-%d".
            option (int, optional): Variable to animate. Options are:
                1 - Total discharge, 2 - Upper zone discharge,
                3 - Ground water, 4 - Snow pack, 5 - Soil moisture,
                6 - Upper zone, 7 - Lower zone, 8 - Water content,
                9 - Precipitation, 10 - ET, 11 - Temperature.
                Default is 1.
            gauges (bool, optional): Whether to plot gauge locations
                on the animation. Default is False.
            **kwargs: Additional keyword arguments passed to
                `ArrayGlyph.animate`. Loose styling keywords still
                accepted: title (str), title_size (int), cmap (str),
                vmin (float), vmax (float), interval (int),
                figsize (tuple), cell_value_text_colors (tuple),
                ticks_spacing (int), cbar_label (str),
                cbar_label_size (int), cbar_length (float),
                cbar_orientation (str).
                Styling that cleopatra 0.30 moved onto typed group
                objects is passed as those objects instead:
                color=`ColorScaling` (was color_scale / gamma /
                bounds / midpoint), cells=`CellValues` (was
                display_cell_value / num_size /
                background_color_threshold),
                contour=`Contour` (was levels),
                data_style=`DataStyle` (was style / hillshade),
                frame_label=`FrameLabel` (was label_location /
                label_color / text_loc). See
                `cleopatra.glyphs.gridded.array_glyph.ArrayGlyph.animate`
                for the full list.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.

        Raises:
            ValueError: If `option` is not between 1 and 11.
        """
        start = dt.datetime.strptime(start, fmt)
        end = dt.datetime.strptime(end, fmt)

        start_i = np.nonzero(self.period.date_index == start)[0][0]
        end_i = np.nonzero(self.period.date_index == end)[0][0]

        if option == 1:
            arr = self.results.q_total[:, :, start_i:end_i]
            title = "Total Discharge"
        elif option == 2:
            arr = self.results.quz_routed[:, :, start_i:end_i]
            title = "Surface Flow"
        elif option == 3:
            arr = self.results.qlz_translated[:, :, start_i:end_i]
            title = "Ground Water Flow"
        elif option == 4:
            arr = self.results.state_variables[:, :, start_i:end_i, 0]
            title = "Snow Pack"
        elif option == 5:
            arr = self.results.state_variables[:, :, start_i:end_i, 1]
            title = "Soil Moisture"
        elif option == 6:
            arr = self.results.state_variables[:, :, start_i:end_i, 2]
            title = "Upper Zone"
        elif option == 7:
            arr = self.results.state_variables[:, :, start_i:end_i, 3]
            title = "Lower Zone"
        elif option == 8:
            arr = self.results.state_variables[:, :, start_i:end_i, 4]
            title = "Water Content"
        elif option == 9:
            arr = self.meteo.precipitation[:, :, start_i:end_i]
            title = "Precipitation"
        elif option == 10:
            arr = self.meteo.evapotranspiration[:, :, start_i:end_i]
            title = "ET"
        elif option == 11:
            arr = self.meteo.temperature[:, :, start_i:end_i]
            title = "Temperature"
        else:
            raise ValueError("Plotting options are from 1 to 11")

        # mask the no-data cells on a copy so plotting never mutates the model
        # result arrays stored on the instance
        arr = arr.copy()
        arr[np.isnan(self.flow_network.flow_acc_arr), :] = np.nan

        time = self.period.date_index[start_i:end_i]

        if gauges:
            # animate expects a 3-column array: [value to display, cell row, cell column].
            # cleopatra 0.30 stopped accepting a bare array; it must be wrapped in a
            # PointOverlay, which also carries the marker/label styling.
            kwargs["points"] = PointOverlay(
                self.GaugesTable[["id", "cell_row", "cell_col"]].to_numpy()
            )

        # animate iterates over the first dimension, so move the time axis to the front
        array = ArrayGlyph(np.moveaxis(arr, -1, 0))
        # the option title is a default; an explicit title= kwarg wins
        kwargs.setdefault("title", title)
        anim = array.animate(time, **kwargs)

        self._animation_glyph = array
        self.anim = anim

        return anim

    def save_animation(self, path: str, fps: int = 2):
        """Save the animation created by `plot_distributed_results`.

        The output format is determined by the file extension. GIF uses
        PillowWriter; mov/avi/mp4 require FFmpeg to be installed.

        Args:
            path (str): Output file path. The extension determines the
                format (gif, mov, avi, or mp4).
            fps (int, optional): Frames per second. Default is 2.

        Raises:
            ValueError: If `plot_distributed_results` has not been called
                yet, or if the file format is not supported.
            FileNotFoundError: If a video format is requested but FFmpeg
                is not installed.
        """
        if self._animation_glyph is None:
            raise ValueError(
                "There is no animation to save, call `plot_distributed_results` first"
            )
        self._animation_glyph.save_animation(path, fps=fps)

    def save_results(
        self,
        flow_acc_path: str = "",
        result: int = 1,
        start: str | dt.datetime = "",
        end: str | dt.datetime = "",
        path: str = "",
        prefix: str = "",
        fmt: str = "%Y-%m-%d",
    ):
        """Save model results to raster files or CSV.

        For distributed mode, saves results as GeoTIFF rasters. For
        lumped mode, saves results as a CSV file.

        Args:
            flow_acc_path (str, optional): Path to the flow
                accumulation raster (required for distributed mode).
                Default is "".
            result (int, optional): Type of result to save:
                1 - Total discharge, 2 - Upper zone discharge,
                3 - Lower zone discharge, 4 - Snow pack,
                5 - Soil moisture, 6 - Upper zone, 7 - Lower zone,
                8 - Water content. For lumped mode, 5 saves all
                variables. Default is 1.
            start (str | dt.datetime, optional): Start date for the
                output period. A string is parsed with `fmt`; a datetime
                is used as it is.
                If empty, uses the first index. Default is "".
            end (str | dt.datetime, optional): End date for the output
                period. See `start`. If
                empty, uses the last index. Default is "".
            path (str, optional): Output directory (distributed, created
                if it does not exist) or the CSV file itself (lumped).
                Default is "", the working directory.
            prefix (str, optional): Prefix for the output file
                names. Default is "".
            fmt (str, optional): Date format for parsing `start` and
                `end`. Default is "%Y-%m-%d".

        Raises:
            Exception: If `flow_acc_path` is not provided in
                distributed mode.
            TypeError: If `path` is not a string. `outputs.results_dir`
                is optional in a run configuration, so a caller
                forwarding it can hold None.
            ValueError: If `result` is not a valid option.
        """
        if not isinstance(path, str):
            raise TypeError(
                f"path must be a string naming a directory (distributed) or a file "
                f"(lumped), got {type(path).__name__}"
            )

        if start == "":
            start = self.period.date_index[0]
        elif isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)

        if end == "":
            end = self.period.date_index[-1]
        elif isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)

        start_i = np.nonzero(self.period.date_index == start)[0][0]
        end_i = np.nonzero(self.period.date_index == end)[0][0] + 1

        if self.spatial_resolution == "distributed":
            if flow_acc_path == "":
                raise Exception(
                    "Please enter the FlowAccPath parameter to the saveResults method"
                )

            src = Dataset.read_file(flow_acc_path)

            if prefix == "":
                prefix = "Result_"

            # `path` names a directory here, unlike the lumped branch below where it is the
            # CSV itself. Joined rather than concatenated: the old `path + prefix` wrote
            # `some/dirResult_2009-01-01.tif` for any directory given without a trailing
            # separator, which is how a directory is normally written.
            if path and not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)
            names = [
                os.path.join(path, f"{prefix}{str(i)[:10]}.tif")
                for i in self.period.date_index[start_i:end_i]
            ]
            if result == 1:
                arr = self.results.q_total[:, :, start_i:end_i]
            elif result == 2:
                arr = self.results.quz_routed[:, :, start_i:end_i]
            elif result == 3:
                arr = self.results.qlz_translated[:, :, start_i:end_i]
            elif result == 4:
                arr = self.results.state_variables[:, :, start_i:end_i, 0]
            elif result == 5:
                arr = self.results.state_variables[:, :, start_i:end_i, 1]
            elif result == 6:
                arr = self.results.state_variables[:, :, start_i:end_i, 2]
            elif result == 7:
                arr = self.results.state_variables[:, :, start_i:end_i, 3]
            elif result == 8:
                arr = self.results.state_variables[:, :, start_i:end_i, 4]
            else:
                raise ValueError(
                    f" The result parameter takes a value between 1 and 8, given: {result}"
                )

            # from_dataset is pyramids' named constructor for an in-memory
            # scaffold off a template raster; the bare Datacube(src, time_length=)
            # form it replaced is kept only as a legacy fallback upstream.
            cube = Datacube.from_dataset(src, arr.shape[2])
            arr = np.moveaxis(arr, -1, 0)
            cube.values = arr
            cube.to_file(names)
        else:
            ind = pd.date_range(start, end, freq="D")
            data = pd.DataFrame(index=ind)

            data["date"] = ["'" + str(i)[:10] + "'" for i in data.index]

            if result == 1:
                data["Qsim"] = self.Qsim[start_i:end_i]
                data.to_csv(path, index=False, float_format="%.3f")
            elif result == 2:
                data["Quz"] = self.results.quz[start_i:end_i]
                data.to_csv(path, index=False, float_format="%.3f")
            elif result == 3:
                data["Qlz"] = self.results.qlz[start_i:end_i]
                data.to_csv(path, index=False, float_format="%.3f")
            elif result == 4:
                data[STATE_VARIABLES] = self.results.state_variables[start_i:end_i, :]
                data.to_csv(path, index=False, float_format="%.3f")
            elif result == 5:
                data["Qsim"] = self.Qsim[start_i:end_i]
                data["Quz"] = self.results.quz[start_i:end_i]
                data["Qlz"] = self.results.qlz[start_i:end_i]
                data[STATE_VARIABLES] = self.results.state_variables[start_i:end_i, :]
                data.to_csv(path, index=False, float_format="%.3f")
            else:
                raise ValueError(
                    f"in lumped mode the result parameter takes a value between 1 and 5, "
                    f"given: {result}"
                )

        logger.debug("Data is saved successfully")


class Lake:
    """Lake simulation using a lumped model with a rating curve.

    The Lake class reads meteorological inputs and a lumped model module to
    simulate a lake. The lake and its upstream sub-catchments are treated as
    one lumped model that produces a discharge input to the lake. The
    discharge input changes the volume of the water in the lake, and the
    outflow is obtained from the volume-outflow (stage-discharge) curve.
    """

    def __init__(
        self,
        start: str = "",
        end: str = "",
        fmt: str = "%Y-%m-%d",
        temporal_resolution: str = "Daily",
        split: bool = False,
    ):
        """Initialize a Lake instance for lake simulation.

        Args:
            start (str, optional): Start date. Default is "".
            end (str, optional): End date. Default is "".
            fmt (str, optional): Date format. Default is "%Y-%m-%d".
            temporal_resolution (str, optional): "Daily" or "Hourly".
                Default is "Daily".
            split (bool, optional): True to subset the data between
                the start and end dates. Default is False.
        """
        self.OutflowCell: list | None = None
        self.Snow: int | None = None
        self.Split = split
        self.start = dt.datetime.strptime(start, fmt)
        self.end = dt.datetime.strptime(end, fmt)

        if temporal_resolution.lower() == "daily":
            self.Index = pd.date_range(start, end, freq="D")
        elif temporal_resolution.lower() == "hourly":
            self.Index = pd.date_range(start, end, freq="h")
        else:
            raise ValueError(
                f"available temporal resolutions are 'daily' and 'hourly', got "
                f"{temporal_resolution!r}"
            )

        self.MeteoData: np.ndarray | None = None
        self.Parameters: list | None = None
        self.LumpedModel: BaseConceptualModel | None = None
        self.CatArea: float | None = None
        self.LakeArea: float | None = None
        self.InitialCond: list | None = None
        self.StageDischargeCurve: np.ndarray | None = None

    def read_meteo_data(self, path: str, fmt: str):
        """Read meteorological data for the lake simulation.

        Reads rainfall, evapotranspiration, and temperature data from a
        CSV file.

        Args:
            path (str): Path to the meteorological data CSV file.
                Columns must be in the order [date, rainfall, ET,
                temperature].
            fmt (str): Date format string used to parse the date
                index.
        """
        df = pd.read_csv(path, index_col=0)
        df.index = [dt.datetime.strptime(date, fmt) for date in df.index]

        if self.Split:
            df = df.loc[self.start : self.end, :]

        self.MeteoData = df.values  # lakeCalibArray = lakeCalibArray[:,0:-1]

        logger.debug("Lake Meteo data are read successfully")

    def read_parameters(self, path):
        """Read lake model parameters from a text file.

        Args:
            path (str): Path to the parameter text file.
        """
        self.Parameters = np.loadtxt(path).tolist()
        logger.debug("Lake Parameters are read successfully")

    def read_lumped_model(
        self,
        lumped_model: type[BaseConceptualModel],
        catchment_area,
        lake_area,
        initial_condition,
        outflow_cell,
        stage_discharge_curve,
        snow,
    ):
        """Read and set up a lumped model for lake simulation.

        Args:
            lumped_model: A class representing the lumped conceptual
                model (e.g., HBV).
            catchment_area (float): Catchment area in km2.
            lake_area (float): Area of the lake in km2.
            initial_condition (list): Initial conditions list
                containing [Snow Pack, Soil Moisture, Upper Zone,
                Lower Zone, Water Content, Lake volume].
            outflow_cell (list): Indices of the cell where the lake
                hydrograph is to be added.
            stage_discharge_curve (np.ndarray): Volume-outflow
                (stage-discharge) curve array.
            snow (int): 0 to skip snow processes, 1 to simulate
                snow. If 1, snow-related parameters must be
                provided.

        Raises:
            ValueError: If `lumped_model` is not a class.
            TypeError: If `initial_condition` is not a list.
        """
        if not inspect.isclass(lumped_model):
            raise ValueError(
                "ConceptualModel should be a module or a python file contains functions "
            )

        self.LumpedModel = lumped_model()

        self.CatArea = catchment_area
        self.LakeArea = lake_area
        self.InitialCond = initial_condition

        if self.InitialCond is not None and not isinstance(self.InitialCond, list):
            raise TypeError(
                f"init_st should be of type list, got {type(self.InitialCond).__name__}"
            )

        self.Snow = snow
        self.OutflowCell = outflow_cell
        self.StageDischargeCurve = stage_discharge_curve
        logger.debug("Lumped model is read successfully")
