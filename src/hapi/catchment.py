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
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statista.descriptors as metrics
from cleopatra.array_glyph import ArrayGlyph
from loguru import logger
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube
from pyramids.feature import FeatureCollection

from hapi.dem import DEM

if TYPE_CHECKING:
    import matplotlib.animation

    from hapi.rrm.base_model import BaseConceptualModel

STATE_VARIABLES = ["SP", "SM", "UZ", "LZ", "WC"]


@contextmanager
def _name_the_path(path) -> Iterator[None]:
    """Re-raise a pyramids `FileNotFoundError` with the offending path in the message.

    ``DatasetCollection.read_multiple_files`` reports "The path you have provided does
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


def _warn_if_no_sentinel(dataset, label: str) -> None:
    """Warn when a raster declares no no-data value, so the whole grid is the domain.

    Before masking was delegated to pyramids, a raster with no marker raised
    ``TypeError`` from `math.isclose(value, None)` — accidental, but loud. pyramids
    masks nothing instead, which is the correct reading of such a raster but silently
    makes every cell part of the catchment. Warn rather than raise: a raster legitimately
    having no marker is valid input.

    Args:
        dataset: The opened pyramids ``Dataset``.
        label: Human-readable name of the input, used in the message.
    """
    if dataset.no_data_value[0] is None:
        warnings.warn(
            f"the {label} raster declares no no-data value, so every cell is treated as "
            "inside the catchment. If it has a sentinel, set it on the band; otherwise "
            "check that a whole-grid domain is intended.",
            UserWarning,
            stacklevel=3,
        )


def _to_int_codes(array: np.ndarray) -> np.typing.NDArray:
    """Return the finite cells of `array` truncated to 64-bit integers.

    Shared by the flow-accumulation and flow-direction readers, which both need the
    distinct *integer* values of a masked raster.

    Truncation happens before the caller de-duplicates: collapsing to integers first is
    what makes 1.2 and 1.8 a single value, matching the per-cell ``set(int(...))`` this
    replaced. De-duplicating first would leave both and yield a repeated ``1``.

    Args:
        array: A 2-D array whose masked cells are ``NaN``.

    Returns:
        np.ndarray: 1-D ``int64`` array of the finite cells, unsorted and not
            de-duplicated.

    Raises:
        ValueError: A cell is infinite, or is too large for ``int64``. ``astype`` would
            otherwise saturate silently to ``INT64_MIN``/``INT64_MAX`` with only a
            ``RuntimeWarning``.
    """
    finite = array[~np.isnan(array)]
    if not np.isfinite(finite).all():
        raise ValueError(
            "raster contains infinite values, which cannot be converted to integer "
            "cell codes; check the source raster's no-data handling."
        )
    info = np.iinfo(np.int64)
    if finite.size and (finite.min() < info.min or finite.max() > info.max):
        raise ValueError(
            f"raster values fall outside the int64 range [{info.min}, {info.max}]; "
            "converting them would silently saturate."
        )
    return finite.astype(np.int64)


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
        spatial_resolution: str | None = "Lumped",
        temporal_resolution: str | None = "Daily",
        routing_method: str | None = "Muskingum",
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
        conversion_factor = (1000 * 24 * 60 * 60) / (1000**2)
        if temporal_resolution.lower() == "daily":
            self.dt = 1  # 24
            self.conversion_factor = conversion_factor * 1
            self.Index = pd.date_range(self.start, self.end, freq="D")
        elif temporal_resolution.lower() == "hourly":
            self.dt = 1  # 24
            self.conversion_factor = conversion_factor * 1 / 24
            self.Index = pd.date_range(self.start, self.end, freq="h")
        else:
            # TODO calculate the temporal resolution factor
            # q mm , area sq km  (1000**2)/1000/f/24/60/60 = 1/(3.6*f)
            # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
            self.conversion_factor = 24

        self.routing_method = routing_method
        self.Parameters: np.ndarray | list | None = None
        self.data: np.ndarray | None = None
        self.Prec: np.ndarray | None = None
        self.TS: int | None = None
        self.Temp: np.ndarray | None = None
        self.ET: np.ndarray | None = None
        self.ll_temp: np.ndarray | float | None = None
        self.QGauges: pd.DataFrame | None = None
        self.Snow: int | None = None
        self.Maxbas: bool | None = None
        self.LumpedModel: BaseConceptualModel | None = None
        self.CatArea: float | int | None = None
        self.InitialCond: list | None = None
        self.q_init: float | None = None
        self.GaugesTable: FeatureCollection | pd.DataFrame | None = None
        self.UB: np.ndarray | None = None
        self.LB: np.ndarray | None = None
        self.cols: int | None = None
        self.rows: int | None = None
        self.NoDataValue: float | None = None
        self.FlowAccArr: np.ndarray | None = None
        self.no_elem: int | None = None
        self.acc_val: list[int] | None = None
        self.Outlet: tuple | None = None
        self.CellSize: float | None = None
        self.px_area: float | None = None
        self.px_tot_area: float | None = None
        self.flow_dir_arr: np.ndarray | None = None
        self.FDT: dict | None = None
        self.fpl_arr: np.ndarray | None = None
        self.DEM: np.ndarray | None = None
        self.BankfullDepth: np.ndarray | None = None
        self.RiverWidth: np.ndarray | None = None
        self.RiverRoughness: np.ndarray | None = None
        self.FloodPlainRoughness: np.ndarray | None = None
        self.qout: np.ndarray | None = None
        self.Qtot: np.ndarray | None = None
        self.quz_routed: np.ndarray | None = None
        self.qlz_translated: np.ndarray | None = None
        self.state_variables: np.ndarray | None = None
        self.anim: matplotlib.animation.FuncAnimation | None = None
        self._animation_glyph: ArrayGlyph | None = None
        self.quz: np.ndarray | None = None
        self.qlz: np.ndarray | None = None
        self.Qsim: np.ndarray | None = None
        self.Metrics: pd.DataFrame | None = None

    def read_rainfall(
        self,
        path: str,
        start: str | None = None,
        end: str | None = None,
        fmt: str = "%Y-%m-%d",
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        extension: str = ".tif",
    ):
        r"""Read rainfall rasters into a 3D numpy array.

        Args:
            path (str): Path to the folder containing precipitation
                rasters.
            start (str, optional): Start date to read a specific
                period only. If not given, all rasters in the path
                will be read. Default is None.
            end (str, optional): End date to read a specific period
                only. If not given, all rasters in the path will be
                read. Default is None.
            fmt (str, optional): Format of the given date. Default
                is "%Y-%m-%d".
            regex_string (str, optional): A regex string to locate
                the date in the file names. Default is
                r"\d{4}.\d{2}.\d{2}".
            date (bool, optional): True if the number in the file
                name is a date. Default is True.
            file_name_data_fmt (str, optional): Date format in file
                names for ordered reading. Default is None.
            extension (str, optional): The extension of the files to
                read from the given path. Default is ".tif".

        Raises:
            FileNotFoundError: The directory does not exist or holds no matching
                rasters. Raised by ``DatasetCollection.read_multiple_files``.
            TypeError: The resulting precipitation array is not a numpy ndarray.
        """
        if self.Prec is None:
            # Path validation is delegated to pyramids: read_multiple_files raises
            # FileNotFoundError for a missing *or* empty directory. Unlike the asserts
            # these replace, that survives `python -O`. Its message does not name the
            # offending directory, so _name_the_path re-raises with it.
            with _name_the_path(path):
                cube = Datacube.read_multiple_files(
                    path,
                    with_order=True,
                    regex_string=regex_string,
                    date=date,
                    start=start,
                    end=end,
                    fmt=fmt,
                    file_name_data_fmt=file_name_data_fmt,
                    extension=extension,
                )
            self.Prec = np.moveaxis(cube.values, 0, -1)
            self.TS = self.Prec.shape[2] + 1
            # no of time steps =length of time series +1
            if not isinstance(self.Prec, np.ndarray):
                raise TypeError("Prec should be of type numpy array")

            logger.debug("Rainfall data are read successfully")

    def read_temperature(
        self,
        path: str,
        ll_temp: list | np.ndarray | None = None,
        start: str | None = None,
        end: str | None = None,
        fmt: str = "%Y-%m-%d",
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        extension: str = ".tif",
    ):
        r"""Read temperature rasters into a 3D numpy array.

        Args:
            path (str): Path to the folder containing temperature
                rasters.
            ll_temp (list | np.ndarray, optional): Long-term
                average temperature array. If None, it is computed
                from the mean of the temperature data. Default is
                None.
            start (str, optional): Start date to read a specific
                period only. If not given, all rasters in the path
                will be read. Default is None.
            end (str, optional): End date to read a specific period
                only. If not given, all rasters in the path will be
                read. Default is None.
            fmt (str, optional): Format of the given date. Default
                is "%Y-%m-%d".
            regex_string (str, optional): A regex string to locate
                the date in the file names. Default is
                r"\d{4}.\d{2}.\d{2}".
            date (bool, optional): True if the number in the file
                name is a date. Default is True.
            file_name_data_fmt (str, optional): Date format in file
                names for ordered reading. Default is None.
            extension (str, optional): The extension of the files to
                read from the given path. Default is ".tif".

        Raises:
            FileNotFoundError: The directory does not exist or holds no matching
                rasters. Raised by ``DatasetCollection.read_multiple_files``.
        """
        if self.Temp is None:
            # Path validation is delegated to pyramids: read_multiple_files raises
            # FileNotFoundError for a missing *or* empty directory. Unlike the asserts
            # these replace, that survives `python -O`. Its message does not name the
            # offending directory, so _name_the_path re-raises with it.
            with _name_the_path(path):
                cube = Datacube.read_multiple_files(
                    path,
                    with_order=True,
                    regex_string=regex_string,
                    date=date,
                    start=start,
                    end=end,
                    fmt=fmt,
                    file_name_data_fmt=file_name_data_fmt,
                    extension=extension,
                )
            self.Temp = np.moveaxis(cube.values, 0, -1)
            assert isinstance(self.Temp, np.ndarray), (
                "array should be of type numpy array"
            )

            if ll_temp is None:
                self.ll_temp = np.zeros_like(self.Temp, dtype=np.float32)
                avg = self.Temp.mean(axis=2)
                for i in range(self.Temp.shape[0]):
                    for j in range(self.Temp.shape[1]):
                        self.ll_temp[i, j, :] = avg[i, j]

            logger.debug("Temperature data are read successfully")

    def read_et(
        self,
        path: str,
        start: str | None = None,
        end: str | None = None,
        fmt: str = "%Y-%m-%d",
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        extension: str = ".tif",
    ):
        r"""Read evapotranspiration rasters into a 3D numpy array.

        Args:
            path (str): Path to the folder containing
                evapotranspiration rasters.
            start (str, optional): Start date to read a specific
                period only. If not given, all rasters in the path
                will be read. Default is None.
            end (str, optional): End date to read a specific period
                only. If not given, all rasters in the path will be
                read. Default is None.
            fmt (str, optional): Format of the given date. Default
                is "%Y-%m-%d".
            regex_string (str, optional): A regex string to locate
                the date in the file names. Default is
                r"\d{4}.\d{2}.\d{2}".
            date (bool, optional): True if the number in the file
                name is a date. Default is True.
            file_name_data_fmt (str, optional): Date format in file
                names for ordered reading. Default is None.
            extension (str, optional): The extension of the files to
                read from the given path. Default is ".tif".

        Raises:
            FileNotFoundError: The directory does not exist or holds no matching
                rasters. Raised by ``DatasetCollection.read_multiple_files``.
        """
        if self.ET is None:
            # Path validation is delegated to pyramids: read_multiple_files raises
            # FileNotFoundError for a missing *or* empty directory. Unlike the asserts
            # these replace, that survives `python -O`. Its message does not name the
            # offending directory, so _name_the_path re-raises with it.
            with _name_the_path(path):
                cube = Datacube.read_multiple_files(
                    path,
                    with_order=True,
                    regex_string=regex_string,
                    date=date,
                    start=start,
                    end=end,
                    fmt=fmt,
                    file_name_data_fmt=file_name_data_fmt,
                    extension=extension,
                )
            self.ET = np.moveaxis(cube.values, 0, -1)
            assert isinstance(self.ET, np.ndarray), (
                "array should be of type numpy array"
            )
            logger.debug("Potential Evapotranspiration data are read successfully")

    def read_flow_acc(self, path: str):
        """Read flow accumulation raster and compute cell properties.

        Reads the flow accumulation raster, extracts the number of rows,
        columns, NoDataValue, number of domain cells, outlet location,
        cell size, and pixel area.

        No-data handling is delegated to pyramids via ``read_array(masked=True)``,
        which compares integer bands for exact equality with the sentinel and float
        bands with a NaN-aware comparison, and additionally honours the band's GDAL
        mask band.

        Note:
            Two consequences worth knowing. The array is promoted to ``float64``
            regardless of the source dtype, so a ``float32`` raster costs twice its
            on-disk size in memory — the price of a representable ``NaN`` mask. And
            because the GDAL mask band is honoured, a raster carrying an alpha or
            internal mask yields a **smaller** domain than before this was delegated,
            which changes :attr:`no_elem` and, through it, the width of the parameter
            arrays: calibration vectors saved against the old domain will not fit. The array is promoted to floating point so masked cells can hold
        ``NaN``, and every downstream attribute (``no_elem``, ``acc_val``, ``Outlet``)
        is derived from that masked array.

        :attr:`acc_val` holds the distinct accumulation values inside the domain, sorted
        ascending, as built-in ``int``. Its maximum is expected to equal the domain cell
        count (or one less, depending on whether the outlet is counted); a mismatch is
        logged at DEBUG rather than raised, since some upstream tools number cells from
        one.

        Cell geometry is read from the named fields of :attr:`~pyramids.dataset.Dataset.transform`.
        :attr:`CellSize` is the pixel **width** in map units (what
        :attr:`~pyramids.dataset.Dataset.cell_size` means), while :attr:`px_area` multiplies the
        pixel width by the pixel height, so a non-square grid is not silently squared off.
        :attr:`px_area` and :attr:`px_tot_area` are in km^2 and assume the raster CRS is
        metric — a geographic (degree) CRS would produce meaningless areas.

        Args:
            path (str | Path): Path to the flow accumulation raster. Any raster format
                GDAL can open is accepted, not only GeoTIFF.

        Raises:
            FileNotFoundError: The path does not exist.
            TypeError: `path` is neither a string nor a ``Path``.
            RuntimeError: GDAL cannot open the file as a raster.
            ValueError: Every cell is no-data, so no accumulation values remain to
                take a maximum of.

        Examples:
            - Read a small accumulation raster and inspect the derived domain
              properties. The bottom-right cell carries the no-data sentinel, so three
              of the four cells lie inside the catchment:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "acc.tif")
                >>> Dataset.create_from_array(
                ...     np.array([[0, 1], [2, -9999]], dtype="int32"),
                ...     top_left_corner=(0.0, 8000.0), cell_size=4000.0, epsg=32618,
                ...     no_data_value=-9999, path=path,
                ... ).close()
                >>> model = Catchment("example", "2000-01-01", "2000-01-02",
                ...                   spatial_resolution="Distributed")
                >>> model.read_flow_acc(path)
                >>> model.no_elem
                3
                >>> float(model.px_area)
                16.0
                >>> model.CellSize
                4000.0
                >>> bool(np.isnan(model.FlowAccArr[1, 1]))
                True
                >>> model.acc_val
                [0, 1, 2]

                ```
            - A real value close to the sentinel survives. ``-9990`` sits within 0.1% of
              ``-9999``, so the tolerance-based comparison used before delegating to
              pyramids destroyed it; exact integer comparison keeps it and the cell
              counts toward the domain:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "acc_near.tif")
                >>> Dataset.create_from_array(
                ...     np.array([[0, 1], [2, -9990]], dtype="int32"),
                ...     top_left_corner=(0.0, 8000.0), cell_size=4000.0, epsg=32618,
                ...     no_data_value=-9999, path=path,
                ... ).close()
                >>> model = Catchment("example", "2000-01-01", "2000-01-02",
                ...                   spatial_resolution="Distributed")
                >>> model.read_flow_acc(path)
                >>> float(model.FlowAccArr[1, 1])
                -9990.0
                >>> model.no_elem
                4

                ```

        See Also:
            Catchment.read_flow_dir: Read the matching flow-direction raster.
            Catchment.read_flow_path_length: Read the matching flow-path-length raster.
        """
        # Path validation is delegated to pyramids: a missing path raises
        # FileNotFoundError, a non-path argument TypeError, and an unreadable file a
        # GDAL RuntimeError. Unlike the asserts these replace, they survive `python -O`.
        flow_acc = Dataset.read_file(path)
        self.rows = flow_acc.rows
        self.cols = flow_acc.columns
        # check flow accumulation input raster
        self.NoDataValue = flow_acc.no_data_value[0]
        _warn_if_no_sentinel(flow_acc, "flow accumulation")
        # Let pyramids resolve the no-data mask: it is vectorised and dtype-aware
        # (exact equality on integer bands, NaN-aware on float ones) and it also
        # honours the band's GDAL mask band. Filling with NaN keeps the
        # float-array-with-NaN contract the rest of this class relies on.
        #
        # astype(float) is unconditional and promotes a float32 raster to float64,
        # doubling resident size. That is the price of a single representable NaN
        # mask: the alternative -- promoting only integer bands -- leaves float32
        # rasters unable to hold NaN at full precision and reintroduces the dtype
        # branch whose `== "int"` test silently failed for int32.
        self.FlowAccArr = np.ma.filled(
            flow_acc.read_array(band=0, masked=True).astype(float), np.nan
        )

        # Count the cells the pyramids mask left intact. Deliberately not
        # Dataset.count_domain_cells(): that re-reads the raster and compares with
        # is_no_data's default rel. tolerance, which masks values within 0.1% of the
        # sentinel -- the defect this branch removed.
        self.no_elem = int(np.count_nonzero(~np.isnan(self.FlowAccArr)))
        # Truncate BEFORE de-duplicating. np.unique on the float values would keep
        # 1.2 and 1.8 apart and only then collapse them to 1, yielding duplicates; the
        # per-cell `set(int(...))` this replaced truncated first, so distinct *integer*
        # accumulation values is the contract.
        self.acc_val = np.unique(_to_int_codes(self.FlowAccArr)).tolist()
        acc_val_mx = max(self.acc_val)

        if not (acc_val_mx == self.no_elem or acc_val_mx == self.no_elem - 1):
            message = (
                "flow accumulation raster values are not correct max "
                "value should equal number of cells or number of cells -1 "
                f"Max Value in the Flow Acc raster is {acc_val_mx}"
                f" while No of cells are {self.no_elem}"
            )
            logger.debug(message)

        # assert acc_val_mx == self.no_elem or acc_val_mx == self.no_elem -1,

        # location of the outlet
        # outlet is the cell that has the max flow_acc
        self.Outlet = np.where(self.FlowAccArr == np.nanmax(self.FlowAccArr))

        # Cell geometry comes from the named fields of the affine transform rather than
        # positional geotransform indices. This is a legibility change only: the
        # expression it replaced already read the two pixel dimensions separately, so
        # non-square grids were handled correctly before and after. What changed is that
        # `geo_trans[-1]` no longer requires the reader to know the geotransform layout.
        transform = flow_acc.transform
        dx = abs(transform.pixel_width) / 1000.0  # dx in Km
        dy = abs(transform.pixel_height) / 1000.0  # dy in Km
        # abs(): Dataset.cell_size returns the signed geotransform pixel width, so a
        # west-to-east-flipped grid would report a negative cell size. The value this
        # replaced was abs()-ed, and every consumer treats it as a magnitude.
        self.CellSize = abs(flow_acc.cell_size)

        # area of the cell
        self.px_area = dx * dy
        self.px_tot_area = self.no_elem * self.px_area  # total area of pixels

        logger.debug("Flow Accmulation input is read successfully")

    def read_flow_dir(self, path: str):
        """Read the flow direction raster and build the flow direction table.

        Cells outside the catchment are masked to ``NaN`` by pyramids via
        ``read_array(masked=True)`` before the ESRI D8 codes are validated, so only
        genuine no-data cells are excluded from validation. A corrupt value that merely
        sits close to the sentinel is therefore no longer swallowed as no-data — it
        reaches the D8 check and is rejected.

        Validation runs on the *distinct* surviving codes, so a raster in which every
        cell shares one direction is legitimate.

        Warning:
            :attr:`FDT` is **not** derived from the masked array above. It comes from
            :meth:`hapi.dem.DEM.flow_direction_table`, which performs its own second read
            of the raster and applies its own ``np.isclose(rtol=1e-5)`` comparison,
            ignoring the band's GDAL mask. The two therefore disagree on any cell whose
            masking depends on the mask band or on the exact-vs-tolerant comparison: such
            a cell can be ``NaN`` in :attr:`flow_dir_arr` yet still appear as a key in
            :attr:`FDT`. The masks already differed before masking was delegated to
            pyramids (``rel_tol=0.001`` against ``rtol=1e-5``); delegating widened the
            gap rather than creating it. Reconciling them means changing
            :mod:`hapi.dem`, which is slated to move to ``digital-rivers``, so it is
            tracked there rather than papered over here.

        :attr:`FDT` is keyed ``"row,col"`` and maps each cell to the cells draining
        directly into it.

        Args:
            path (str | Path): Path to the flow direction raster. Any raster format GDAL
                can open is accepted, not only GeoTIFF.

        Raises:
            FileNotFoundError: The path does not exist.
            TypeError: `path` is neither a string nor a ``Path``.
            RuntimeError: GDAL cannot open the file as a raster.
            AssertionError: The raster contains values other than
                1, 2, 4, 8, 16, 32, 64, 128.

        Examples:
            - Read a small D8 raster and inspect the upstream lookup table. The
              bottom-right cell is no-data, so it gets no entry:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "fd.tif")
                >>> Dataset.create_from_array(
                ...     np.array([[2, 4], [1, -9999]], dtype="int32"),
                ...     top_left_corner=(0.0, 8000.0), cell_size=4000.0, epsg=32618,
                ...     no_data_value=-9999, path=path,
                ... ).close()
                >>> model = Catchment("example", "2000-01-01", "2000-01-02",
                ...                   spatial_resolution="Distributed")
                >>> model.read_flow_dir(path)
                >>> sorted(model.FDT)
                ['0,0', '0,1', '1,0']
                >>> float(model.flow_dir_arr[0, 0])
                2.0

                ```
            - A value that is not a valid D8 code is rejected rather than modelled:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.catchment import Catchment
                >>> path = os.path.join(tempfile.mkdtemp(), "fd_bad.tif")
                >>> Dataset.create_from_array(
                ...     np.array([[2, 4], [1, 3]], dtype="int32"),
                ...     top_left_corner=(0.0, 8000.0), cell_size=4000.0, epsg=32618,
                ...     no_data_value=-9999, path=path,
                ... ).close()
                >>> model = Catchment("example", "2000-01-01", "2000-01-02",
                ...                   spatial_resolution="Distributed")
                >>> try:
                ...     model.read_flow_dir(path)
                ... except AssertionError as exc:
                ...     print("rejected:", "1,2,4,8,16,32,64,128" in str(exc))
                rejected: True

                ```

        See Also:
            Catchment.read_flow_acc: Read the matching flow-accumulation raster.
            hapi.dem.DEM.flow_direction_table: Builds the upstream lookup table.
        """
        # Path validation is delegated to pyramids: a missing path raises
        # FileNotFoundError, a non-path argument TypeError, and an unreadable file a
        # GDAL RuntimeError. Unlike the asserts these replace, they survive `python -O`.
        flow_dir = DEM.read_file(path)
        _warn_if_no_sentinel(flow_dir, "flow direction")
        # No-data masking is delegated to pyramids (see read_flow_acc).
        self.flow_dir_arr = np.ma.filled(
            flow_dir.read_array(band=0, masked=True).astype(float), np.nan
        )

        fd_val = np.unique(_to_int_codes(self.flow_dir_arr))
        fd_should = {1, 2, 4, 8, 16, 32, 64, 128}
        assert set(fd_val.tolist()) <= fd_should, (
            "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "
        )

        # create the flow direction table
        self.FDT = flow_dir.flow_direction_table()
        logger.debug("Flow Direction input is read successfully")

    def read_flow_path_length(self, path: str):
        """Read the flow path length raster.

        Reads the flow path length raster and extracts rows, columns,
        NoDataValue, and the number of domain cells.

        No-data handling is delegated to pyramids via ``read_array(masked=True)``, so
        cells outside the catchment become ``NaN`` and ``no_elem`` counts only the
        cells that remain. The array is promoted to floating point so masked cells can
        hold ``NaN``.

        Args:
            path (str | Path): Path to the flow path length raster. Any raster format
                GDAL can open is accepted, not only GeoTIFF.

        Raises:
            FileNotFoundError: The path does not exist.
            TypeError: `path` is neither a string nor a ``Path``.
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
                >>> model.no_elem
                3
                >>> float(model.fpl_arr[0, 1])
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
                >>> model.no_elem
                4

                ```

        See Also:
            Catchment.read_flow_acc: Read the matching flow-accumulation raster.
        """
        # Path validation is delegated to pyramids: a missing path raises
        # FileNotFoundError, a non-path argument TypeError, and an unreadable file a
        # GDAL RuntimeError. Unlike the asserts these replace, they survive `python -O`.
        fpl = Dataset.read_file(path)
        self.rows = fpl.rows
        self.cols = fpl.columns
        # No-data masking is delegated to pyramids (see read_flow_acc).
        self.fpl_arr = np.ma.filled(
            fpl.read_array(band=0, masked=True).astype(float), np.nan
        )
        self.NoDataValue = fpl.no_data_value[0]
        _warn_if_no_sentinel(fpl, "flow path length")
        # check flow accumulation input raster
        # Count the cells the pyramids mask left intact (see read_flow_acc).
        self.no_elem = int(np.count_nonzero(~np.isnan(self.fpl_arr)))

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
            ("BankfullDepth", bankfull_depth_file),
            ("RiverWidth", river_width_file),
            ("RiverRoughness", river_roughness_file),
            ("FloodPlainRoughness", floodplain_roughness_file),
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
            # Path validation is delegated to pyramids: read_multiple_files raises
            # FileNotFoundError for a missing *or* empty directory. Unlike the asserts
            # these replace, that survives `python -O`. Its message does not name the
            # offending directory, so _name_the_path re-raises with it.
            with _name_the_path(path):
                cube = Datacube.read_multiple_files(
                    path, with_order=True, regex_string=r"\d+", date=False
                )
            self.Parameters = np.moveaxis(cube.values, 0, -1)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "The parameter file you have entered does not exist"
                )

            self.Parameters = pd.read_csv(path, index_col=0, header=None)[1].tolist()

        if not (not snow or snow):
            raise ValueError(
                "snow input defines whether to consider snow subroutine or not it has to be True or False"
            )

        self.Snow = snow
        self.Maxbas = maxbas

        if self.spatial_resolution == "distributed":
            if snow and maxbas:
                if not self.Parameters.shape[2] == 16:
                    raise ValueError(
                        "current version of HBV (with snow) takes 16 parameters you have entered "
                        f"{self.Parameters.shape[2]}"
                    )
            elif not snow and maxbas:
                if not self.Parameters.shape[2] == 11:
                    raise ValueError(
                        "current version of HBV (with snow) takes 11 parameters you have entered "
                        f"{self.Parameters.shape[2]}"
                    )
            elif snow and not maxbas:
                if not self.Parameters.shape[2] == 17:
                    raise ValueError(
                        "current version of HBV (with snow) takes 17 parameters you have entered "
                        f"{self.Parameters.shape[2]}"
                    )
            elif not snow and not maxbas:
                if not self.Parameters.shape[2] == 12:
                    raise ValueError(
                        "current version of HBV (with snow) takes 12 parameters you have entered "
                        f"{self.Parameters.shape[2]}"
                    )
        else:
            if snow and maxbas:
                if not len(self.Parameters) == 16:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 16 parameters you have entered"
                        f" {len(self.Parameters)}"
                    )

            elif not snow and maxbas:
                if len(self.Parameters) != 11:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 11 parameters you have entered"
                        f" {len(self.Parameters)}"
                    )

            elif snow and not maxbas:
                if not len(self.Parameters) == 17:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 17 parameters you have entered{len(self.Parameters)}"
                    )

            elif not snow and not maxbas:
                if not len(self.Parameters) == 12:
                    raise ValueError(
                        f"current version of HBV (with snow) takes 12 parameters you have entered"
                        f" {len(self.Parameters)}"
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

        self.LumpedModel = lumped_model()
        self.CatArea = catchment_area

        if len(initial_condition) != 5:
            raise ValueError(
                f"state variables are 5 and the given initial values are {len(initial_condition)}"
            )

        self.InitialCond = initial_condition

        if q_init is not None:
            assert not isinstance(q_init, float), "q_init should be of type float"
        self.q_init = q_init

        if self.InitialCond is not None:
            assert isinstance(self.InitialCond, list), "init_st should be of type list"

        logger.debug("Lumped model is read successfully")

    def read_lumped_inputs(self, path: str, ll_temp: list | np.ndarray | None = None):
        """Read meteorological inputs for lumped mode.

        Reads precipitation, evapotranspiration, temperature, and
        optionally long-term average temperature from a CSV file.

        Args:
            path (str): Path to the input CSV file. Data columns must
                be in the order [date, precipitation, ET, Temp].
            ll_temp (list | np.ndarray, optional): Average
                long-term temperature. If None, it is calculated as
                the mean of the temperature column. Default is None.

        Raises:
            ValueError: If the input data does not have 3 or 4
                columns (excluding the date index).
        """
        self.data = pd.read_csv(path, header=0, delimiter=",", index_col=0)
        self.data = self.data.values

        if ll_temp is None:
            # self.ll_temp = np.zeros(shape=(len(self.data)), dtype=np.float32)
            self.ll_temp = self.data[:, 2].mean()

        if not (np.shape(self.data)[1] == 3 or np.shape(self.data)[1] == 4):
            raise ValueError(
                "meteorological data should be of length at least 3 (prec, ET, temp) or 4(prec, ET, temp, tm) "
            )

        logger.debug("Lumped Model inputs are read successfully")

    def read_gauge_table(
        self, path: str, flow_acc_file: str = "", fmt: str = "%Y-%m-%d"
    ):
        """Read the gauge table listing gauge locations and properties.

        Reads gauge data including coordinates (x, y), area ratio, and
        weight. The coordinates are mandatory to locate the gauges and
        extract discharge at the corresponding cells.

        The result lands on :attr:`GaugesTable`, and its type follows the input format:

        * ``.geojson`` is read with
          :meth:`pyramids.feature.FeatureCollection.read_file`, giving a
          :class:`~pyramids.feature.FeatureCollection` — a ``GeoDataFrame`` subclass, so
          it keeps its geometry column and CRS.
        * anything else is read with :func:`pandas.read_csv`, giving a plain
          :class:`~pandas.DataFrame` with no geometry.

        When ``flow_acc_file`` is given and the table has no ``cell_row`` column, each
        gauge is mapped onto the raster grid and ``cell_row`` / ``cell_col`` columns are
        appended.

        ``start`` and ``end`` columns, if present, are parsed with ``fmt`` into
        ``datetime64`` columns. The two are handled independently, so a table carrying
        only one of them is fine.

        Args:
            path (str): Path to the gauge file (CSV or GeoJSON).
            flow_acc_file (str, optional): Path to the flow
                accumulation raster used to map gauge coordinates to
                array indices. Default is "".
            fmt (str, optional): Date format for start/end columns
                in the gauge table. Default is "%Y-%m-%d".

        Raises:
            ValueError: A ``start`` or ``end`` value does not match ``fmt``.

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
            - A validity period is parsed into datetime columns using ``fmt``:
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
        gauge table (read via ``read_gauge_table``). For lumped mode, a
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
        self.Snow = snow
        self.Maxbas = maxbas

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
            self.Qsim = pd.DataFrame(index=self.Index, columns=self.QGauges.columns)
            if calculate_metrics:
                index = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
                self.Metrics = pd.DataFrame(index=index, columns=self.QGauges.columns)
            # sum the lower zone and the upper zone discharge
            outlet_x = self.Outlet[0][0]
            outlet_y = self.Outlet[1][0]

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

                q_sim = np.reshape(self.Qtot[x_ind, y_ind, :-1], self.TS - 1)
                if factor is not None:
                    self.Qsim.loc[:, gauge_id] = q_sim * factor[i]
                else:
                    self.Qsim.loc[:, gauge_id] = q_sim

                if calculate_metrics:
                    q_obs = self.QGauges.loc[:, gauge_id]
                    self.Metrics.loc["RMSE", gauge_id] = round(
                        metrics.rmse(q_obs, q_sim), 3
                    )
                    self.Metrics.loc["NSE", gauge_id] = round(
                        metrics.nse(q_obs, q_sim), 3
                    )
                    self.Metrics.loc["NSEhf", gauge_id] = round(
                        metrics.nse_hf(q_obs, q_sim), 3
                    )
                    self.Metrics.loc["KGE", gauge_id] = round(
                        metrics.kge(q_obs, q_sim), 3
                    )
                    self.Metrics.loc["WB", gauge_id] = round(
                        metrics.wb(q_obs, q_sim), 3
                    )
                    self.Metrics.loc["Pearson-CC", gauge_id] = round(
                        metrics.pearson_corr_coeff(q_obs, q_sim), 3
                    )
                    self.Metrics.loc["R2", gauge_id] = round(
                        metrics.r2(q_obs, q_sim), 3
                    )
        elif frame_work_1 or only_outlet:
            self.Qsim = pd.DataFrame(index=self.Index)
            gauge_id = self.GaugesTable.loc[self.GaugesTable.index[-1], "id"]
            q_sim = np.reshape(self.qout, self.TS - 1)
            self.Qsim.loc[:, gauge_id] = q_sim

            if calculate_metrics:
                index = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
                self.Metrics = pd.DataFrame(index=index)

                # if CalculateMetrics:
                q_obs = self.QGauges.loc[:, gauge_id]
                self.Metrics.loc["RMSE", gauge_id] = round(
                    metrics.rmse(q_obs, q_sim), 3
                )
                self.Metrics.loc["NSE", gauge_id] = round(metrics.nse(q_obs, q_sim), 3)
                self.Metrics.loc["NSEhf", gauge_id] = round(
                    metrics.nse_hf(q_obs, q_sim), 3
                )
                self.Metrics.loc["KGE", gauge_id] = round(metrics.kge(q_obs, q_sim), 3)
                self.Metrics.loc["WB", gauge_id] = round(metrics.wb(q_obs, q_sim), 3)
                self.Metrics.loc["Pearson-CC", gauge_id] = round(
                    metrics.pearson_corr_coeff(q_obs, q_sim), 3
                )
                self.Metrics.loc["R2", gauge_id] = round(metrics.r2(q_obs, q_sim), 3)

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

        if self.Metrics:
            logger.debug("----------------------------------")
            logger.debug("Gauge - " + str(gauge_id))
            logger.debug("RMSE= " + str(round(self.Metrics.loc["RMSE", gauge_id], 2)))
            logger.debug("NSE= " + str(round(self.Metrics.loc["NSE", gauge_id], 2)))
            logger.debug("NSEhf= " + str(round(self.Metrics.loc["NSEhf", gauge_id], 2)))
            logger.debug("KGE= " + str(round(self.Metrics.loc["KGE", gauge_id], 2)))
            logger.debug("WB= " + str(round(self.Metrics.loc["WB", gauge_id], 2)))
            logger.debug(
                "Pearson-CC= " + str(round(self.Metrics.loc["Pearson-CC", gauge_id], 2))
            )
            logger.debug("R2= " + str(round(self.Metrics.loc["R2", gauge_id], 2)))

        return fig, ax

    def plot_distributed_results(
        self,
        start: str | dt.datetime,
        end: str | dt.datetime,
        fmt: str = "%Y-%m-%d",
        option: int = 1,
        gauges: bool = False,
        **kwargs,
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
                `ArrayGlyph.animate`. Common options include:
                title (str), interval (int),
                cell_value_text_colors (tuple),
                frame_label (cleopatra `FrameLabel`),
                title_size (int), cmap (str), vmin (float),
                vmax (float), color_scale (str), ticks_spacing (int),
                cbar_label (str), cbar_label_size (int),
                cbar_length (float), cbar_orientation (str),
                display_cell_value (bool), num_size (int),
                background_color_threshold (float), figsize (tuple).
                See `cleopatra.array_glyph.ArrayGlyph.animate` for
                the full list.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.

        Raises:
            ValueError: If `option` is not between 1 and 11.
        """
        start = dt.datetime.strptime(start, fmt)
        end = dt.datetime.strptime(end, fmt)

        start_i = np.where(self.Index == start)[0][0]
        end_i = np.where(self.Index == end)[0][0]

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
            arr = self.Prec[:, :, start_i:end_i]
            title = "Precipitation"
        elif option == 10:
            arr = self.ET[:, :, start_i:end_i]
            title = "ET"
        elif option == 11:
            arr = self.Temp[:, :, start_i:end_i]
            title = "Temperature"
        else:
            raise ValueError("Plotting options are from 1 to 11")

        # mask the no-data cells on a copy so plotting never mutates the model
        # result arrays stored on the instance
        arr = arr.copy()
        arr[np.isnan(self.FlowAccArr), :] = np.nan

        time = self.Index[start_i:end_i]

        if gauges:
            # animate expects a 3-column array: [value to display, cell row, cell column]
            kwargs["points"] = self.GaugesTable[
                ["id", "cell_row", "cell_col"]
            ].to_numpy()

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
            start = self.Index[0]
        else:
            start = dt.datetime.strptime(start, fmt)

        if end == "":
            end = self.Index[-1]
        else:
            end = dt.datetime.strptime(end, fmt)

        start_i = np.where(self.Index == start)[0][0]
        end_i = np.where(self.Index == end)[0][0] + 1

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
            names = [path + str(i)[:10] for i in self.Index[start_i:end_i]]
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

            cube = Datacube(src, time_length=arr.shape[2])
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
