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
from typing import TYPE_CHECKING, Any

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statista.descriptors as metrics
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay
from loguru import logger
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube
from pyramids.feature import FeatureCollection

from hapi.inputs import (
    FlowNetwork,
    MeteoInputs,
    _warn_if_no_sentinel,
    read_rasters,
)

if TYPE_CHECKING:
    import matplotlib.animation

    from hapi.rrm.base_model import BaseConceptualModel

STATE_VARIABLES = ["SP", "SM", "UZ", "LZ", "WC"]
CONVERSION_FACTOR = (1000 * 24 * 60 * 60) / (1000**2)
DATE_PATTERN = r"\d{4}.\d{2}.\d{2}"


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
    data of the gauges. It is a superclass that has the Run subclass, so you
    need to build the Catchment object and hand it as an input to the Run
    class to run the model.
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
            routing_method (str, optional): Routing method name.
                Default is "Muskingum".

        Raises:
            ValueError: If `spatial_resolution` is not "lumped" or
                "distributed".
            ValueError: If `temporal_resolution` is not "daily" or
                "hourly".
        """
        self.name = name
        self.start = dt.datetime.strptime(start_data, fmt)
        self.end = dt.datetime.strptime(end, fmt)

        if spatial_resolution.lower() not in ["lumped", "distributed"]:
            raise ValueError(
                "available spatial resolutions are 'lumped' and 'distributed'"
            )
        self.spatial_resolution = spatial_resolution.lower()

        if temporal_resolution.lower() not in ["daily", "hourly"]:
            raise ValueError("available temporal resolutions are 'daily' and 'hourly'")
        self.temporal_resolution = temporal_resolution.lower()
        # assuming the default dt is 1 day
        # Only the two resolutions the check above admits: an `else` here would be
        # unreachable, and the one that used to sit here set a conversion factor but no
        # `date_index`, which reads as support for sub-daily steps that does not exist.
        # Adding one (q mm, area km2: 1/(3.6*f)) means widening the check above too.
        if self.temporal_resolution == "daily":
            self.dt = 1  # 24
            self.conversion_factor = CONVERSION_FACTOR * 1
            self.date_index = pd.date_range(self.start, self.end, freq="D")
        else:
            self.dt = 1  # 24
            self.conversion_factor = CONVERSION_FACTOR * 1 / 24
            self.date_index = pd.date_range(self.start, self.end, freq="h")

        self.routing_method = routing_method
        self.parameters: np.ndarray | list | None = None
        self.data: np.ndarray | None = None
        #: The three meteorological drivers. Assign a :class:`~hapi.inputs.MeteoInputs`
        #: built by one of its loaders; everything meteorological hangs off it.
        self.meteo: MeteoInputs | None = None
        self.QGauges: pd.DataFrame | None = None
        self.snow: int | None = None
        self.maxbas: bool | None = None
        self.lumped_model: BaseConceptualModel | None = None
        self.area: float | int | None = None
        self.initial_cond: list | None = None
        self.q_init: float | None = None
        self.GaugesTable: FeatureCollection | pd.DataFrame | None = None
        self.UB: np.ndarray | None = None
        self.LB: np.ndarray | None = None
        #: The routing network and the grid it defines. Assign a
        #: :class:`~hapi.inputs.FlowNetwork` built by its loader.
        self.flow_network: FlowNetwork | None = None
        self.flow_path_length_arr: np.ndarray | None = None
        self.DEM: np.ndarray | None = None
        self.bankfull_depth: np.ndarray | None = None
        self.river_width: np.ndarray | None = None
        self.river_roughness: np.ndarray | None = None
        self.flood_plain_roughness: np.ndarray | None = None
        self.qout: np.ndarray | None = None
        self.Qtot: np.ndarray | None = None
        self.quz_routed: np.ndarray | None = None
        self.qlz_translated: np.ndarray | None = None
        # True once a triangular (MAXBAS) run has filled the output fields. The
        # MAXBAS routing sends every cell straight to the outlet, so a single cell of
        # `Qtot` is that cell's contribution, not the discharge at it — which makes
        # the outlet-cell shortcut in `extract_discharge` invalid. See its guard.
        self._maxbas_routed: bool = False
        self.state_variables: np.ndarray | None = None
        self.anim: matplotlib.animation.FuncAnimation | None = None
        self._animation_glyph: ArrayGlyph | None = None
        self.quz: np.ndarray | None = None
        self.qlz: np.ndarray | None = None
        self.Qsim: np.ndarray | None = None
        self.metrics: pd.DataFrame | None = None

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
            self.parameters = np.moveaxis(cube.values, 0, -1)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "The parameter file you have entered does not exist"
                )

            self.parameters = pd.read_csv(path, index_col=0, header=None)[1].tolist()

        if not (not snow or snow):
            raise ValueError(
                "snow input defines whether to consider snow subroutine or not it has to be True or False"
            )

        self.snow = snow
        self.maxbas = maxbas

        if self.spatial_resolution == "distributed":
            if snow and maxbas:
                if self.parameters.shape[2] != 16:
                    raise ValueError(
                        "current version of HBV (with snow) takes 16 parameters you have entered "
                        f"{self.parameters.shape[2]}"
                    )
            elif not snow and maxbas:
                if self.parameters.shape[2] != 11:
                    raise ValueError(
                        "current version of HBV (with snow) takes 11 parameters you have entered "
                        f"{self.parameters.shape[2]}"
                    )
            elif snow and not maxbas:
                if self.parameters.shape[2] != 17:
                    raise ValueError(
                        "current version of HBV (with snow) takes 17 parameters you have entered "
                        f"{self.parameters.shape[2]}"
                    )
            elif not snow and not maxbas:
                if self.parameters.shape[2] != 12:
                    raise ValueError(
                        "current version of HBV (with snow) takes 12 parameters you have entered "
                        f"{self.parameters.shape[2]}"
                    )
        else:
            if snow and maxbas:
                if len(self.parameters) != 16:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 16 parameters you have entered"
                        f" {len(self.parameters)}"
                    )

            elif not snow and maxbas:
                if len(self.parameters) != 11:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 11 parameters you have entered"
                        f" {len(self.parameters)}"
                    )

            elif snow and not maxbas:
                if not len(self.parameters) == 17:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 17 parameters you have entered{len(self.parameters)}"
                    )

            elif not snow and not maxbas:
                if not len(self.parameters) == 12:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 12 parameters you have entered"
                        f" {len(self.parameters)}"
                    )

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
            ValueError: If `lumped_model` is not a class or if
                `initial_condition` does not contain exactly 5
                values.
        """
        if not inspect.isclass(lumped_model):
            raise ValueError(
                "ConceptualModel should be a module or a python file contains functions "
            )

        self.lumped_model = lumped_model()
        self.area = catchment_area

        if len(initial_condition) != 5:
            raise ValueError(
                f"state variables are 5 and the given initial values are {len(initial_condition)}"
            )

        self.initial_cond = initial_condition

        if q_init is not None:
            assert not isinstance(q_init, float), "q_init should be of type float"
        self.q_init = q_init

        if self.initial_cond is not None:
            assert isinstance(self.initial_cond, list), "init_st should be of type list"

        logger.debug("Lumped model is read successfully")

    def read_lumped_inputs(self, path: str):
        """Read meteorological inputs for lumped mode.

        The lumped counterpart of :class:`~hapi.inputs.MeteoInputs`, which carries the
        distributed drivers: the lumped model works on one column per variable rather than a
        grid, and `Wrapper.Lumped` reads the long-term average straight out of the fourth
        column.

        A three-column file is completed with a fourth holding the record's mean temperature.
        `Wrapper.Lumped` reads that column unconditionally, so without it a file this method
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
                blank = parsed.isna() & self.GaugesTable[column].notna()
                if blank.any() or parsed.isna().any():
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
            column (str, optional): Name of the column in the gauge
                table containing the file names. Default is "id".
            fmt (str, optional): Date format in the discharge files.
                Default is "%Y-%m-%d".
            split (bool, optional): True to subset the data between
                `start_date` and `end_date`. Default is False.
            start_date (str, optional): Start date for subsetting.
                Default is "".
            end_date (str, optional): End date for subsetting.
                Default is "".
            readfrom (str, optional): Number of rows to skip when
                reading the CSV. Default is "".

        Raises:
            FileNotFoundError: If the discharge file does not exist
                (lumped mode).
            AssertionError: If the gauge table has not been read yet
                (distributed mode).
        """
        if self.temporal_resolution.lower() == "daily":
            ind = pd.date_range(self.start, self.end, freq="D")
        else:
            ind = pd.date_range(self.start, self.end, freq="h")

        if self.spatial_resolution.lower() == "distributed":
            assert hasattr(self, "GaugesTable"), "please read the gauges' table first"

            self.QGauges = pd.DataFrame(
                index=ind, columns=self.GaugesTable[column].tolist()
            )

            for i in range(len(self.GaugesTable)):
                name = self.GaugesTable.loc[i, "id"]
                if readfrom != "":
                    f = pd.read_csv(
                        f"{path}/{name}.csv",
                        index_col=0,
                        delimiter=delimiter,
                        skiprows=readfrom,
                    )  # ,#delimiter="\t"
                else:
                    f = pd.read_csv(
                        f"{path}/{name}.csv",
                        header=0,
                        index_col=0,
                        delimiter=delimiter,
                    )

                f.index = [dt.datetime.strptime(i, fmt) for i in f.index.tolist()]
                self.QGauges[int(name)] = f.loc[self.start : self.end, f.columns[-1]]
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"The file you have entered{path} does not exist"
                )

            self.QGauges = pd.DataFrame(index=ind)
            f = pd.read_csv(path, header=0, index_col=0, delimiter=delimiter)
            f.index = [dt.datetime.strptime(i, fmt) for i in f.index.tolist()]
            self.QGauges[f.columns[0]] = f.loc[self.start : self.end, f.columns[0]]

        if split:
            start_date = dt.datetime.strptime(start_date, fmt)
            end_date = dt.datetime.strptime(end_date, fmt)
            self.QGauges = self.QGauges.loc[start_date:end_date]

        logger.debug("Gauges data are read successfully")

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
            AssertionError: If the lengths of `upper_bound` and
                `lower_bound` are not equal.
            ValueError: If `snow` is not a boolean.
        """
        assert len(upper_bound) == len(lower_bound), (
            "the length of UB should be the same as LB"
        )
        self.UB = np.array(upper_bound)
        self.LB = np.array(lower_bound)

        if not isinstance(snow, bool):
            raise ValueError(
                " snow input defines whether to consider snow subroutine or not it has to be True or False"
            )
        self.snow = snow
        self.maxbas = maxbas

        logger.debug("Parameters' bounds are read successfully")

    def extract_discharge(
        self, calculate_metrics=True, frame_work_1=False, factor=None, only_outlet=False
    ):
        """Extract and sum discharge at gauge locations.

        Extracts and sums the discharge from the routed upper zone and
        translated lower zone arrays at each gauge location. Optionally
        computes performance metrics (RMSE, NSE, NSEhf, KGE, WB,
        Pearson-CC, R2) between simulated and observed hydrographs.

        Args:
            calculate_metrics (bool, optional): Whether to calculate
                performance metrics. Default is True.
            frame_work_1 (bool, optional): True if the routing
                function is Maxbas. Default is False.
            factor (list, optional): List of multiplication factors
                for simulated discharge at each gauge. Must have the
                same length as the number of gauges. Default is None.
            only_outlet (bool, optional): True to extract discharge
                only at the outlet cell. Default is False.

        Raises:
            ValueError: If the gauge table has not been read yet.
        """
        if self.GaugesTable is None:
            raise ValueError("please read the gauges' table first.")

        if not frame_work_1:
            if self._maxbas_routed:
                raise ValueError(
                    "this catchment was run with triangular (MAXBAS) routing, which "
                    "sends every cell straight to the outlet: a single cell of Qtot is "
                    "that cell's contribution, not the discharge at it, so reading the "
                    "outlet cell would under-report the hydrograph. Call "
                    "extract_discharge(frame_work_1=True) to use the basin-wide sum "
                    "that Run.runFW1 computed."
                )
            self.Qsim = pd.DataFrame(
                index=self.date_index, columns=self.QGauges.columns
            )
            if calculate_metrics:
                index = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
                self.metrics = pd.DataFrame(index=index, columns=self.QGauges.columns)
            # sum the lower zone and the upper zone discharge
            outlet_x = self.flow_network.outlet[0][0]
            outlet_y = self.flow_network.outlet[1][0]

            # self.qout = self.qlz_translated[outlet_x,outlet_y,:] + self.quz_routed[outlet_x,outlet_y,:]
            # self.Qtot = self.qlz_translated + self.quz_routed
            self.qout = self.Qtot[outlet_x, outlet_y, :]

            for i in range(len(self.GaugesTable)):
                x_ind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_row"])
                y_ind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_col"])
                gauge_id = self.GaugesTable.loc[self.GaugesTable.index[i], "id"]

                # Quz = np.reshape(self.quz_routed[x_ind,y_ind,:-1],self.TS-1)
                # Qlz = np.reshape(self.qlz_translated[x_ind,y_ind,:-1],self.TS-1)
                # q_sim = Quz + Qlz

                q_sim = np.reshape(self.Qtot[x_ind, y_ind, :-1], self.meteo.time_steps)
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
        elif frame_work_1 or only_outlet:
            self.Qsim = pd.DataFrame(index=self.date_index)
            gauge_id = self.GaugesTable.loc[self.GaugesTable.index[-1], "id"]
            q_sim = np.reshape(self.qout, self.meteo.time_steps)
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
            start_date (str): Starting date for the plot.
            end_date (str): End date for the plot.
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
        start_date = dt.datetime.strptime(start_date, fmt)
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

        start_i = np.where(self.date_index == start)[0][0]
        end_i = np.where(self.date_index == end)[0][0]

        if option == 1:
            arr = self.Qtot[:, :, start_i:end_i]
            title = "Total Discharge"
        elif option == 2:
            arr = self.quz_routed[:, :, start_i:end_i]
            title = "Surface Flow"
        elif option == 3:
            arr = self.qlz_translated[:, :, start_i:end_i]
            title = "Ground Water Flow"
        elif option == 4:
            arr = self.state_variables[:, :, start_i:end_i, 0]
            title = "Snow Pack"
        elif option == 5:
            arr = self.state_variables[:, :, start_i:end_i, 1]
            title = "Soil Moisture"
        elif option == 6:
            arr = self.state_variables[:, :, start_i:end_i, 2]
            title = "Upper Zone"
        elif option == 7:
            arr = self.state_variables[:, :, start_i:end_i, 3]
            title = "Lower Zone"
        elif option == 8:
            arr = self.state_variables[:, :, start_i:end_i, 4]
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

        time = self.date_index[start_i:end_i]

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
            start (str, optional): Start date for the output period.
                If empty, uses the first index. Default is "".
            end (str, optional): End date for the output period. If
                empty, uses the last index. Default is "".
            path (str, optional): Path to the output directory
                (distributed) or file (lumped). Default is "".
            prefix (str, optional): Prefix for the output file
                names. Default is "".
            fmt (str, optional): Date format for parsing `start` and
                `end`. Default is "%Y-%m-%d".

        Raises:
            Exception: If `flow_acc_path` is not provided in
                distributed mode.
            ValueError: If `result` is not a valid option.
        """
        if start == "":
            start = self.date_index[0]
        else:
            start = dt.datetime.strptime(start, fmt)

        if end == "":
            end = self.date_index[-1]
        else:
            end = dt.datetime.strptime(end, fmt)

        start_i = np.where(self.date_index == start)[0][0]
        end_i = np.where(self.date_index == end)[0][0] + 1

        if self.spatial_resolution == "distributed":
            if flow_acc_path == "":
                raise Exception(
                    "Please enter the FlowAccPath parameter to the saveResults method"
                )

            src = Dataset.read_file(flow_acc_path)

            if prefix == "":
                prefix = "Result_"

            # create a list of names
            path = path + prefix
            names = [path + str(i)[:10] for i in self.date_index[start_i:end_i]]
            # names = [i.replace("-", "_") for i in names]
            # names = [i.replace(" ", "_") for i in names]
            names = [i + ".tif" for i in names]
            if result == 1:
                arr = self.Qtot[:, :, start_i:end_i]
            elif result == 2:
                arr = self.quz_routed[:, :, start_i:end_i]
            elif result == 3:
                arr = self.qlz_translated[:, :, start_i:end_i]
            elif result == 4:
                arr = self.state_variables[:, :, start_i:end_i, 0]
            elif result == 5:
                arr = self.state_variables[:, :, start_i:end_i, 1]
            elif result == 6:
                arr = self.state_variables[:, :, start_i:end_i, 2]
            elif result == 7:
                arr = self.state_variables[:, :, start_i:end_i, 3]
            elif result == 8:
                arr = self.state_variables[:, :, start_i:end_i, 4]
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
                data["Quz"] = self.quz[start_i:end_i]
                data.to_csv(path, index=False, float_format="%.3f")
            elif result == 3:
                data["Qlz"] = self.qlz[start_i:end_i]
                data.to_csv(path, index=False, float_format="%.3f")
            elif result == 4:
                data[STATE_VARIABLES] = self.state_variables[start_i:end_i, :]
                data.to_csv(path, index=False, float_format="%.3f")
            elif result == 5:
                data["Qsim"] = self.Qsim[start_i:end_i]
                data["Quz"] = self.quz[start_i:end_i]
                data["Qlz"] = self.qlz[start_i:end_i]
                data[STATE_VARIABLES] = self.state_variables[start_i:end_i, :]
                data.to_csv(path, index=False, float_format="%.3f")
            else:
                assert False, "the possible options are from 1 to 5"

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
            assert False, "Error"

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
            AssertionError: If `initial_condition` is not a list.
        """
        if not inspect.isclass(lumped_model):
            raise ValueError(
                "ConceptualModel should be a module or a python file contains functions "
            )

        self.LumpedModel = lumped_model()

        self.CatArea = catchment_area
        self.LakeArea = lake_area
        self.InitialCond = initial_condition

        if self.InitialCond is not None:
            assert isinstance(self.InitialCond, list), "init_st should be of type list"

        self.Snow = snow
        self.OutflowCell = outflow_cell
        self.StageDischargeCurve = stage_discharge_curve
        logger.debug("Lumped model is read successfully")
