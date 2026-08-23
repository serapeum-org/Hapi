"""Rainfall-runoff inputs.

Everything the distributed model is driven by, and the loaders that build it:

* `MeteoInputs` -- the three meteorological drivers (precipitation, temperature,
  evapotranspiration) as aligned `(rows, cols, time)` cubes, from folders of dated rasters,
  one NetCDF per variable, or a single NetCDF holding all three.
* `FlowNetwork` -- the routing network (flow accumulation, flow direction, the direction
  table) and the grid it defines, from the accumulation and direction rasters.
* `Inputs` -- preparation utilities: aligning rasters to a source DEM, extracting HBV
  parameters from the global datasets, and building lumped inputs from distributed ones.
* `read_rasters` -- the shared adapter over `DatasetCollection.from_files` that decides the
  order a folder's rasters are handed over in.

Rasters are read in chronological order by `DatasetCollection.from_files`, which parses the
date out of each file name, so the files themselves never need renaming on disk.

The module relies on the `pyramids` library for raster I/O and manipulation, and uses the
`HAPI_DATA_DIR` environment variable to locate pre-downloaded global parameter sets
(Beck et al., 2016).
"""

from __future__ import annotations

import datetime as dt
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube
from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

from hapi.dem import DEM


def _as_datetime(value: str | int | None, fmt: str) -> dt.datetime | None:
    """Parse a `start` / `end` bound into a datetime for the date-ordered read.

    Args:
        value: The bound as given by the caller, or None for "unbounded". Declared
            `str | int` because the numeric ordering takes integer indices; in the date
            branch it is a date string, and anything else fails the parse below.
        fmt: `strptime` format of the bound.

    Returns:
        dt.datetime | None: The parsed bound, or None when `value` is None.

    Raises:
        ValueError: `value` does not match `fmt`.
    """
    return None if value is None else dt.datetime.strptime(str(value), fmt)


def _infer_date_format(sample: str) -> str | None:
    """Derive a `strptime` format from a date already matched out of a file name.

    `from_files` needs an explicit format before it will sort by date, but the caller has
    already said where the date sits via `regex_string`. Rather than leave the read
    unordered when no format is given, rebuild one from the shape of what the regex matched:
    the digit runs give the fields, the characters between them are kept verbatim.

    Only the unambiguous layouts are inferred. `1990.02.03` is `%Y.%m.%d` because a
    four-digit leading run can only be a year; `03.02.1990` is refused because day-first and
    month-first cannot be told apart from the digits alone.

    Args:
        sample: The substring `regex_string` matched, e.g. `"2009.01.01"` or `"20090101"`.

    Returns:
        str | None: The format, or None when the layout is ambiguous or unrecognised.

    Examples:
        >>> _infer_date_format("2009.01.01")
        '%Y.%m.%d'
        >>> _infer_date_format("2009_01_01")
        '%Y_%m_%d'
        >>> _infer_date_format("20090101")
        '%Y%m%d'
        >>> _infer_date_format("01.01.2009") is None
        True
    """
    parts = re.findall(r"\d+|\D+", sample)
    widths = tuple(len(p) for p in parts if p.isdigit())

    if widths == (8,):
        return "%Y%m%d"
    if widths != (4, 2, 2):
        # (2, 2, 4) and friends cannot be resolved: nothing in the digits says whether the
        # leading pair is the day or the month, and guessing wrong reorders the whole cube.
        return None
    if any("%" in p for p in parts if not p.isdigit()):
        return None

    directives = iter(("%Y", "%m", "%d"))
    return "".join(next(directives) if p.isdigit() else p for p in parts)


def _infer_date_format_from_folder(
    path: str | Path,
    glob: str,
    regex_string: str,
    gdal_env: dict[str, str] | None = None,
) -> str | None:
    """Sample a folder's file names and infer the date format they carry.

    Args:
        path: Folder holding the rasters.
        glob: `fnmatch` pattern selecting them.
        regex_string: Where the date sits in each name.
        gdal_env: GDAL configuration options applied while resolving the folder.

    Returns:
        str | None: The inferred `strptime` format, or None when no name matched the regex
            or the layout is ambiguous -- in which case the read falls back to unordered and
            a warning says so.
    """
    for file in Datacube.from_files(path, glob=glob, gdal_env=gdal_env).files:
        match = re.search(regex_string, Path(file).name)
        if match is None:
            continue
        inferred = _infer_date_format(match.group())
        if inferred is not None:
            return inferred
        warnings.warn(
            f"could not tell the date layout of {match.group()!r} in {Path(file).name!r} "
            f"apart (day-first and month-first look alike), so {path} is read in file-name "
            "order rather than by date; pass file_name_data_fmt to say which it is",
            stacklevel=3,
        )
        return None

    warnings.warn(
        f"regex {regex_string!r} matched no file name in {path}, so it is read in file-name "
        "order rather than by date; pass regex_string and file_name_data_fmt to order it",
        stacklevel=3,
    )
    return None


def read_rasters(
    path: str | Path,
    *,
    glob: str = "*.tif",
    regex_string: str = r"\d{4}.\d{2}.\d{2}",
    date: bool = True,
    file_name_data_fmt: str | None = None,
    start: str | int | None = None,
    end: str | int | None = None,
    fmt: str = "%Y-%m-%d",
    gdal_env: dict[str, str] | None = None,
) -> Datacube:
    r"""Read a folder of rasters into a `DatasetCollection` in the right order.

    A thin adapter over :meth:`DatasetCollection.from_files` -- pyramids does every bit of the
    resolving and reading; this only decides the order the files are handed over in, and
    translates Hapi's string/int `start` / `end` into what `from_files` accepts.

    Three orderings are supported, matching Hapi's public reader arguments:

    * **By date** (`date=True`) -- delegated wholesale to
      `from_files(date_format=..., date_regex=...)`, which sorts and builds the time axis.
      When no `file_name_data_fmt` is given it is inferred from the first name `regex_string`
      matches, so the default ordering is chronological rather than lexicographic.
    * **By number** (`date=False`) -- for names carrying a plain index, e.g.
      `01_Par_RFCF.tif` or `1000_Temp_..._1981_9_27.tif`. `from_files` sorts only by date, and
      its default order is lexicographic, which puts `10_` before `2_` whenever the index is
      not zero-padded. So the files are resolved through `from_files`, sorted on the integer in
      each name, and handed back to `from_files` as an explicit sequence -- which it keeps in
      the given order.
    * **Unordered** -- only when `date=True` and the layout cannot be inferred (an ambiguous
      day-first/month-first date, or a regex that matches no name). Both warn.

    Args:
        path: Folder holding the rasters.
        glob: :mod:`fnmatch` pattern selecting them. Defaults to `"*.tif"`.
        regex_string: Where the date (or the index, when `date=False`) sits in each name.
        date: Whether the matched value is a date. `False` selects the numeric ordering.
        file_name_data_fmt: `strptime` format of the date in the names. Inferred from the
            names themselves when omitted; pass it for a layout that cannot be told apart
            from the digits alone, such as a day-first `03.02.1990`.
        start: Inclusive lower bound -- a date string parsed with `fmt`, or an integer index
            when `date=False`.
        end: Inclusive upper bound; see `start`.
        fmt: `strptime` format of `start` / `end` when they are date strings.
        gdal_env: GDAL configuration options applied for the read, e.g.
            `{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}`. GDAL lists the directory on
            every open to look for sidecars, which on network storage is a remote listing
            per raster -- 369 ms against 18 ms in one measurement over 14,823 files. Left
            unset by default because disabling it also stops GDAL finding `.aux.xml`,
            world files and `.ovr`, which some of this repository's fixtures rely on.

    Returns:
        DatasetCollection: The collection, ordered as described above.

    Raises:
        FileNotFoundError: The folder does not exist, matched no file, or `start` / `end`
            excluded every file.
        ValueError: `regex_string` matched no number in a file name (numeric ordering only).
    """
    if date and file_name_data_fmt is None:
        # The caller said where the date is; that is enough to sort on. Reading the folder
        # unordered here used to hand back a lexicographic cube -- `10_precip_2009.01.11`
        # in slot 1 -- with no error and no time axis, so neither the length check nor the
        # calendar check could see it, and the run silently paired each day's rainfall with
        # the wrong date.
        file_name_data_fmt = _infer_date_format_from_folder(
            path, glob, regex_string, gdal_env
        )

    if date and file_name_data_fmt is not None:
        return Datacube.from_files(
            path,
            glob=glob,
            date_format=file_name_data_fmt,
            date_regex=regex_string,
            start=_as_datetime(start, fmt),
            end=_as_datetime(end, fmt),
            gdal_env=gdal_env,
        )

    if not date:
        return _read_by_index(path, glob, regex_string, start, end, gdal_env)

    return Datacube.from_files(path, glob=glob, gdal_env=gdal_env)


def _read_by_index(
    path: str | Path,
    glob: str,
    regex_string: str,
    start: str | int | None,
    end: str | int | None,
    gdal_env: dict[str, str] | None = None,
) -> Datacube:
    """Read a folder whose names carry a plain index, ordered numerically.

    `from_files` sorts only by date and otherwise keeps lexicographic order, which puts
    `10_` before `2_` whenever the index is not zero-padded -- scrambling parameter rasters
    into the wrong HBV slots. So resolve the files, sort on the integer in each name, and
    hand them back as an explicit sequence, which `from_files` preserves.

    Args:
        path: Folder holding the rasters.
        glob: `fnmatch` pattern selecting them.
        regex_string: Where the index sits in each name.
        start: Inclusive lower bound on the index, or None.
        end: Inclusive upper bound on the index, or None.
        gdal_env: GDAL configuration options applied for the read.

    Returns:
        Datacube: The collection in ascending index order.

    Raises:
        ValueError: `regex_string` matched no number in one of the names.
        FileNotFoundError: `start` / `end` excluded every file.
    """
    keyed = []
    for file in Datacube.from_files(path, glob=glob, gdal_env=gdal_env).files:
        match = re.search(regex_string, Path(file).name)
        if match is None:
            raise ValueError(
                f"regex {regex_string!r} matched no number in {Path(file).name!r}"
            )
        keyed.append((int(match.group()), file))
    keyed.sort()

    if start is not None or end is not None:
        low = int(start) if start is not None else None
        high = int(end) if end is not None else None
        keyed = [
            (number, file)
            for number, file in keyed
            if (low is None or number >= low) and (high is None or number <= high)
        ]
        if not keyed:
            raise FileNotFoundError(
                f"no file in {path} carries an index within [{start}, {end}]"
            )

    return Datacube.from_files([file for _, file in keyed], gdal_env=gdal_env)


def _warn_if_no_sentinel(dataset, label: str) -> None:
    """Warn when a raster declares no no-data value, so the whole grid is the domain.

    Before masking was delegated to pyramids, a raster with no marker raised
    `TypeError` from `math.isclose(value, None)` — accidental, but loud. pyramids
    masks nothing instead, which is the correct reading of such a raster but silently
    makes every cell part of the catchment. Warn rather than raise: a raster legitimately
    having no marker is valid input.

    Args:
        dataset: The opened pyramids `Dataset`.
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

    Shared by the flow-accumulation and flow-direction reads, which both need the
    distinct *integer* values of a masked raster.

    Truncation happens before the caller de-duplicates: collapsing to integers first is
    what makes 1.2 and 1.8 a single value, matching the per-cell `set(int(...))` this
    replaced. De-duplicating first would leave both and yield a repeated `1`.

    Args:
        array: A 2-D array whose masked cells are `NaN`.

    Returns:
        np.ndarray: 1-D `int64` array of the finite cells, unsorted and not
            de-duplicated.

    Raises:
        ValueError: A cell is infinite, or is too large for `int64`. `astype` would
            otherwise saturate silently to `INT64_MIN`/`INT64_MAX` with only a
            `RuntimeWarning`.
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


#: The eight-directional ESRI codes a flow-direction raster may hold.
D8_CODES = frozenset({1, 2, 4, 8, 16, 32, 64, 128})


@dataclass(eq=False)
class FlowNetwork:
    """The catchment's routing network and the grid it defines.

    Built from the flow-accumulation and flow-direction rasters, which together fix both
    *where* the catchment is -- its grid, its domain cells, its outlet -- and *how* water
    moves through it. The two were separate readers on
    :class:`~hapi.catchment.Catchment`; holding them together keeps the grid with the
    array it is measured from.

    Only what the rasters carry is stored. Everything the flow-accumulation reader computed
    is derived here instead, so the grid can never disagree with the accumulation array
    it came from.

    Cells outside the domain are `NaN` in both arrays: masking is delegated to pyramids
    via `read_array(masked=True)`, which compares integer bands to the sentinel exactly
    and honours a band's GDAL mask.

    Attributes:
        flow_acc_arr: `(rows, cols)` flow accumulation, `NaN` outside the domain.
        flow_dir_arr: `(rows, cols)` D8 flow direction, `NaN` outside the domain.
        FDT: Flow-direction table -- `"row,col"` mapped to the cells draining into it.
        no_data_value: The accumulation raster's sentinel, as declared on the band.
        cell_size: Pixel width in map units.
        px_area: Pixel area in km2 -- width times height, so a non-square grid is not
            silently squared off. Assumes a metric CRS; a geographic one gives a
            meaningless area.

    Examples:
        >>> from hapi.inputs import FlowNetwork
        >>> network = FlowNetwork.from_rasters(  # doctest: +SKIP
        ...     "gis/acc4000.tif", "gis/fd4000.tif"
        ... )
        >>> network.rows, network.cols  # doctest: +SKIP
        (13, 14)

        - Or straight from arrays, which is what the properties below derive from:

          >>> import numpy as np
          >>> from hapi.inputs import FlowNetwork
          >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
          >>> network = FlowNetwork(
          ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
          ... )
          >>> network.shape, network.no_elem
          ((2, 2), 3)

    Warning:
        `FDT` is **not** derived from the masked `flow_dir_arr`. It comes from
        :meth:`hapi.dem.DEM.flow_direction_table`, which reads the raster a second time and
        applies its own `np.isclose(rtol=1e-5)` comparison, ignoring the band's GDAL mask.
        The two therefore disagree on any cell whose masking depends on the mask band or on
        the exact-vs-tolerant comparison: such a cell can be `NaN` in `flow_dir_arr` and
        still appear as a key in `FDT`. The masks already differed before masking was
        delegated to pyramids (`rel_tol=0.001` against `rtol=1e-5`); delegating widened the
        gap rather than creating it. Reconciling them means changing :mod:`hapi.dem`, which
        is slated to move to `digital-rivers`, so it is left as it is and documented here.

    """

    flow_acc_arr: np.ndarray
    no_data_value: float | int | None
    cell_size: float
    px_area: float
    flow_dir_arr: np.ndarray | None = None
    FDT: dict | None = None

    def __post_init__(self):
        """Check the two rasters describe the same grid.

        Raises:
            ValueError: The accumulation and direction arrays are not the same shape, so
                a cell index would mean a different place in each.
        """
        if (
            self.flow_dir_arr is not None
            and self.flow_acc_arr.shape != self.flow_dir_arr.shape
        ):
            raise ValueError(
                f"the flow accumulation raster is {self.flow_acc_arr.shape} but the flow "
                f"direction raster is {self.flow_dir_arr.shape}; both must share the "
                "catchment's grid"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """tuple[int, int]: The `(rows, cols)` grid both rasters share."""
        return self.flow_acc_arr.shape

    @property
    def rows(self) -> int:
        """int: Number of grid rows."""
        return self.shape[0]

    @property
    def cols(self) -> int:
        """int: Number of grid columns."""
        return self.shape[1]

    @property
    def no_elem(self) -> int:
        """int: Number of cells inside the domain, i.e. not masked.

        Sizes the parameter vectors a calibration produces, so it is derived from the
        masked array rather than recounted from the raster.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import FlowNetwork
            >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
            >>> network = FlowNetwork(
            ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
            ... )
            >>> network.no_elem
            3

        """
        return int(np.count_nonzero(~np.isnan(self.flow_acc_arr)))

    def __setattr__(self, name: str, value: object) -> None:
        """Drop the cached `acc_val` when the array it is derived from is replaced.

        Args:
            name: Attribute being set.
            value: New value.
        """
        if name == "flow_acc_arr":
            self.__dict__.pop("acc_val", None)
        object.__setattr__(self, name, value)

    @cached_property
    def acc_val(self) -> list[int]:
        """list[int]: The distinct accumulation values inside the domain, ascending.

        Cached: `SpatialRouting` reads this once per `(accumulation level, row, column)`, so
        recomputing the `np.unique` on every read costs `(n_acc - 1) x rows x cols` scans of
        the whole grid -- unnoticeable on the 13x14 test catchment and hours on a real one.
        Replacing `flow_acc_arr` clears the cache.

        The maximum is expected to equal :attr:`no_elem`, or one less depending on whether
        the outlet is counted; :meth:`from_rasters` logs a mismatch at DEBUG rather than
        raising, since some upstream tools number cells from one.

        Examples:
            - Values are truncated before de-duplication, so 1.2 and 1.8 are one code:

              >>> import numpy as np
              >>> from hapi.inputs import FlowNetwork
              >>> acc = np.array([[1.2, 1.8], [3.0, np.nan]])
              >>> FlowNetwork(
              ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
              ... ).acc_val
              [1, 3]

        """
        values: list[int] = np.unique(_to_int_codes(self.flow_acc_arr)).tolist()
        return values

    @property
    def outlet(self) -> tuple:
        """tuple: Index of the most-accumulated cell, as `np.where` returns it."""
        return np.nonzero(self.flow_acc_arr == np.nanmax(self.flow_acc_arr))

    @property
    def px_tot_area(self) -> float:
        """float: Total domain area in km2 -- :attr:`no_elem` times :attr:`px_area`."""
        return self.no_elem * self.px_area

    def matches(self, rows: int, cols: int) -> bool:
        """Report whether the network covers a given grid.

        Args:
            rows: Number of rows to compare against.
            cols: Number of columns.

        Returns:
            bool: True when the network's grid is exactly `(rows, cols)`.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import FlowNetwork
            >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
            >>> network = FlowNetwork(
            ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
            ... )
            >>> network.matches(2, 2), network.matches(3, 3)
            (True, False)

        """
        return self.shape == (rows, cols)

    @property
    def has_flow_direction(self) -> bool:
        """bool: Whether a flow-direction raster was loaded.

        The Muskingum path routes cell to cell and needs one; the triangular (MAXBAS) path
        sends every cell straight to the outlet and does not.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import FlowNetwork
            >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
            >>> network = FlowNetwork(
            ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
            ... )
            >>> network.has_flow_direction
            False

        """
        return self.flow_dir_arr is not None

    @classmethod
    def from_rasters(
        cls, flow_acc: str | Path, flow_dir: str | Path | None = None
    ) -> FlowNetwork:
        """Read the routing network from the accumulation and direction rasters.

        Args:
            flow_acc: Path to the flow-accumulation raster. Any format GDAL can open.
            flow_dir: Path to the flow-direction raster, in the eight-directional ESRI
                encoding. Optional: the Muskingum routing needs it, but the triangular
                (MAXBAS) path sends every cell straight to the outlet and never reads it.

        Returns:
            FlowNetwork: The two masked arrays, the direction table, and the cell geometry
                read off the accumulation raster's transform.

        Raises:
            FileNotFoundError: Either path does not exist.
            ValueError: The direction raster holds a code outside
                :data:`D8_CODES`, the two rasters disagree on the grid, or every
                accumulation cell is no-data.

        Warns:
            UserWarning: A raster declares no no-data value, so every cell is treated as
                inside the catchment.
        """
        acc = Dataset.read_file(str(flow_acc))
        _warn_if_no_sentinel(acc, "flow accumulation")
        acc_arr = np.ma.filled(
            acc.read_array(band=0, masked=True).astype(float), np.nan
        )

        dir_arr, table = None, None
        if flow_dir is not None:
            direction = DEM.read_file(str(flow_dir))
            _warn_if_no_sentinel(direction, "flow direction")
            dir_arr = np.ma.filled(
                direction.read_array(band=0, masked=True).astype(float), np.nan
            )
            codes = set(np.unique(_to_int_codes(dir_arr)).tolist())
            if not codes <= set(D8_CODES):
                raise ValueError(
                    "flow direction raster should contain values 1,2,4,8,16,32,64,128 "
                    f"only, found {sorted(codes - set(D8_CODES))}"
                )
            table = direction.flow_direction_table()

        transform = acc.transform
        network = cls(
            flow_acc_arr=acc_arr,
            flow_dir_arr=dir_arr,
            FDT=table,
            no_data_value=acc.no_data_value[0],
            cell_size=abs(acc.cell_size),
            px_area=(abs(transform.pixel_width) / 1000.0)
            * (abs(transform.pixel_height) / 1000.0),
        )

        if not network.acc_val:
            raise ValueError(
                f"every cell of {flow_acc} is no-data, so the catchment has no domain: "
                "check the raster's no-data value and its mask band"
            )
        acc_val_mx = max(network.acc_val)
        if acc_val_mx not in (network.no_elem, network.no_elem - 1):
            logger.debug(
                "flow accumulation raster values are not correct max value should equal "
                "number of cells or number of cells -1 Max Value in the Flow Acc raster "
                f"is {acc_val_mx} while No of cells are {network.no_elem}"
            )
        logger.debug("Flow network is read successfully")
        return network


#: The meteorological drivers the conceptual model consumes, in the order the readers report them.
METEO_VARIABLES = ("precipitation", "temperature", "evapotranspiration")


def _cube_from_netcdf(nc: NetCDF, variable: str) -> np.ndarray:
    """Read one NetCDF variable as a `(rows, cols, time)` cube.

    Args:
        nc: An open :class:`~pyramids.netcdf.NetCDF`.
        variable: Name of the variable to read.

    Returns:
        np.ndarray: The variable with time moved to the last axis, matching the layout the
            raster readers produce.

    Raises:
        KeyError: `variable` is not in the file.
    """
    if variable not in nc.variable_names:
        raise KeyError(
            f"variable {variable!r} is not in the NetCDF. Available: {nc.variable_names}."
        )
    values = np.asarray(nc.get_variable(variable).read_array())
    # NetCDF stores (time, y, x); the model indexes cells then time.
    return np.moveaxis(values, 0, -1)


@dataclass(eq=False)
class MeteoInputs:
    r"""The three meteorological drivers of the rainfall-runoff model, held as aligned cubes.

    Each field is a `(rows, cols, time)` array — cell first, time last — which is the layout
    :class:`~hapi.catchment.Catchment` and the conceptual models index. The three cubes must
    agree on all three axes; that is checked on construction, because a silent mismatch surfaces
    much later as a confusing index error inside the run loop.

    No-data cells are carried through **as stored**, not converted to NaN. The distributed model
    takes its domain from the flow-accumulation raster rather than from the meteorological
    no-data mask, so masking here would change what the run sees.

    Build one with whichever classmethod matches how the data is stored:

    * :meth:`from_rasters` -- three folders of date-stamped rasters (the historical layout).
    * :meth:`from_netcdf_files` -- one NetCDF per variable.
    * :meth:`from_netcdf` -- a single NetCDF holding all three as separate variables.

    Attributes:
        precipitation: `(rows, cols, time)` rainfall cube.
        temperature: `(rows, cols, time)` temperature cube.
        evapotranspiration: `(rows, cols, time)` potential-evapotranspiration cube.
        time: Optional calendar axis, one entry per timestep. Carried for reference and for
            cross-checking against the model's own date index; the run itself is positional.

    Examples:
        - From three folders of rasters:
            ```python
            >>> from hapi.inputs import MeteoInputs
            >>> data = MeteoInputs.from_rasters(  # doctest: +SKIP
            ...     "data/prec", "data/temp", "data/evap", file_name_data_fmt="%Y.%m.%d"
            ... )

            ```
        - From one NetCDF per variable:
            ```python
            >>> data = MeteoInputs.from_netcdf_files(  # doctest: +SKIP
            ...     "data/prec.nc", "data/temp.nc", "data/evap.nc"
            ... )

            ```
    """

    precipitation: np.ndarray
    temperature: np.ndarray
    evapotranspiration: np.ndarray
    time: pd.DatetimeIndex | None = field(default=None)
    #: Cache behind the :attr:`ll_temp` property; not part of the constructor signature.
    _ll_temp: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Check the three cubes are 3D and share a shape.

        Raises:
            ValueError: A cube is not 3-dimensional, the three shapes disagree, or `time` does
                not have one entry per timestep.
        """
        for name in METEO_VARIABLES:
            cube = getattr(self, name)
            if not isinstance(cube, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy array, got {type(cube).__name__}"
                )
            if cube.ndim != 3:
                raise ValueError(
                    f"{name} must be a 3D (rows, cols, time) array, got shape {cube.shape}"
                )

        shapes = {name: getattr(self, name).shape for name in METEO_VARIABLES}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"the three cubes must share one shape, got {shapes}")

        if self.time is not None and len(self.time) != self.time_steps:
            raise ValueError(
                f"time has {len(self.time)} entries but the cubes hold {self.time_steps} steps"
            )

    def __setattr__(self, name: str, value: object) -> None:
        """Set an attribute, keeping the three cubes in agreement.

        The class promises the cubes share a shape, and `__post_init__` alone cannot hold
        that promise: the fields are plain mutable attributes, so replacing one afterwards
        silently breaks it. `shape`, `rows` and `time_steps` all report precipitation's, so
        a replacement of the wrong size passes `validate_against` and the run then indexes
        past the end of whichever cube is short -- or, worse, reads the right index of the
        wrong grid. Re-check on assignment instead.

        Replacing `temperature` also drops the cached `ll_temp`, which is derived from it.

        Args:
            name: Attribute being set.
            value: New value.

        Raises:
            ValueError: `value` is a cube that does not match the other two.
        """
        if name in METEO_VARIABLES and getattr(self, name, None) is not None:
            self._check_replacement(name, value)
        if name == "temperature" and getattr(self, "_ll_temp", None) is not None:
            object.__setattr__(self, "_ll_temp", None)
        object.__setattr__(self, name, value)

    def _check_replacement(self, name: str, value: object) -> None:
        """Reject a cube that would leave the three disagreeing.

        Args:
            name: Which cube is being replaced.
            value: The replacement.

        Raises:
            ValueError: The replacement is not a 3D array of the shape the others share.
        """
        others = [
            getattr(self, other)
            for other in METEO_VARIABLES
            if other != name and getattr(self, other, None) is not None
        ]
        if not others:
            return
        expected = others[0].shape
        if not isinstance(value, np.ndarray) or value.shape != expected:
            got = getattr(value, "shape", type(value).__name__)
            raise ValueError(
                f"{name} must stay {expected} to match the other cubes, got {got}; "
                "build a new MeteoInputs to change the grid or the period"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """tuple[int, int, int]: The shared `(rows, cols, time)` shape."""
        return self.precipitation.shape

    @property
    def rows(self) -> int:
        """int: Number of grid rows."""
        return self.shape[0]

    @property
    def cols(self) -> int:
        """int: Number of grid columns."""
        return self.shape[1]

    @property
    def time_steps(self) -> int:
        """int: Number of timesteps the drivers cover."""
        return self.shape[2]

    @property
    def simulation_steps(self) -> int:
        """int: `time_steps` plus one, the length the run's state arrays need.

        The conceptual model carries an initial state before the first driver step, so the
        per-cell result arrays hold one slot more than there is data.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import MeteoInputs
            >>> cube = np.zeros((2, 3, 4), dtype="float32")
            >>> data = MeteoInputs(cube, cube, cube)
            >>> data.time_steps, data.simulation_steps
            (4, 5)

        """
        return self.time_steps + 1

    @property
    def ll_temp(self) -> np.ndarray:
        """np.ndarray: Long-term average temperature, `(rows, cols, time)`.

        Each cell's mean over the whole record, broadcast back across the time axis -- the
        reference the snow routine compares each step against. Derived on first use and cached,
        since the run reads it per cell and it never changes once the cubes are set.

        Assign to this to override the derived value; the replacement must match
        :attr:`shape`.

        Examples:
            - Each cell's own mean, repeated across time:

                >>> import numpy as np
                >>> from hapi.inputs import MeteoInputs
                >>> temp = np.arange(8, dtype="float32").reshape(1, 2, 4)
                >>> data = MeteoInputs(temp, temp, temp)
                >>> data.ll_temp[0, 0, :]
                array([1.5, 1.5, 1.5, 1.5], dtype=float32)
                >>> data.ll_temp[0, 1, :]
                array([5.5, 5.5, 5.5, 5.5], dtype=float32)

        """
        if self._ll_temp is None:
            avg = self.temperature.mean(axis=2)
            self._ll_temp = np.repeat(
                avg[:, :, np.newaxis], self.time_steps, axis=2
            ).astype(np.float32)
        return self._ll_temp

    @ll_temp.setter
    def ll_temp(self, value: np.ndarray) -> None:
        """Override the derived long-term average temperature.

        Args:
            value: A `(rows, cols, time)` array matching :attr:`shape`.

        Raises:
            ValueError: `value` does not match the cubes' shape.
        """
        value = np.asarray(value)
        if value.shape != self.shape:
            raise ValueError(
                f"ll_temp must match the cubes {self.shape}, got {value.shape}"
            )
        self._ll_temp = value

    def validate_against(
        self, rows: int, cols: int, date_index: pd.DatetimeIndex | None = None
    ) -> None:
        """Check the cubes cover the model's grid, and optionally its calendar.

        The three cubes already agree with each other -- that is settled at construction. This
        is the other half: that they agree with the grid the GIS inputs defined, and with the
        period the model was built for.

        Args:
            rows: Number of grid rows the model expects.
            cols: Number of grid columns.
            date_index: The model's own dates. When given, the drivers must supply one step
                per date. `None` skips the check, which is what a caller with no calendar of
                its own does.

        Raises:
            ValueError: The cubes do not cover that grid, or they do not span `date_index`.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import MeteoInputs
            >>> cube = np.zeros((2, 3, 4), dtype="float32")
            >>> data = MeteoInputs(cube, cube, cube)
            >>> data.validate_against(2, 3)
            >>> data.validate_against(5, 5)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: the meteorological inputs are 2x3 but the model grid is 5x5...

        """
        if (self.rows, self.cols) != (rows, cols):
            raise ValueError(
                f"the meteorological inputs are {self.rows}x{self.cols} but the model grid is "
                f"{rows}x{cols}; every input must share the catchment's grid"
            )

        if date_index is None:
            return

        if self.time_steps != len(date_index):
            raise ValueError(
                f"the meteorological inputs hold {self.time_steps} steps but the model spans "
                f"{len(date_index)} ({date_index[0]:%Y-%m-%d} to {date_index[-1]:%Y-%m-%d}); "
                "the run is positional, so a mismatch silently pairs each step with the wrong "
                "date"
            )
        if self.time is not None and (
            self.time[0] != date_index[0] or self.time[-1] != date_index[-1]
        ):
            raise ValueError(
                f"the meteorological inputs cover {self.time[0]:%Y-%m-%d} to "
                f"{self.time[-1]:%Y-%m-%d} but the model spans {date_index[0]:%Y-%m-%d} to "
                f"{date_index[-1]:%Y-%m-%d}"
            )

    @classmethod
    def from_rasters(
        cls,
        precipitation: str | Path,
        temperature: str | Path,
        evapotranspiration: str | Path,
        per_variable: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> MeteoInputs:
        r"""Read the three drivers from folders of date-stamped rasters.

        Args:
            precipitation: Folder of rainfall rasters.
            temperature: Folder of temperature rasters.
            evapotranspiration: Folder of evapotranspiration rasters.
            per_variable: Per-folder overrides, keyed by driver name, merged over `kwargs`
                for that folder only. Needed when the three folders come from different
                sources, which the documented download workflow produces: CHIRPS names its
                rainfall `..._2009.01.01.tif` while ERA5 names its temperature
                `..._20090101.tif`, and no single `regex_string` finds the date in both.
            **kwargs: Forwarded to :func:`read_rasters` for all three folders -- e.g.
                `regex_string`, `date`, `file_name_data_fmt`, `start`, `end`, `fmt`, `glob`.

        Returns:
            MeteoInputs: The three cubes plus a calendar -- the rainfall folder's when it carries
                one, otherwise the first source that does, and None when none do.

        Raises:
            FileNotFoundError: A folder does not exist or holds no matching raster.
            KeyError: `per_variable` names something that is not one of the three drivers.
            ValueError: The three folders do not yield the same shape.

        Examples:
            Three folders from one source share every argument:

            >>> MeteoInputs.from_rasters(  # doctest: +SKIP
            ...     prec_dir, temp_dir, evap_dir, start="2009-01-01", end="2009-12-31"
            ... )

            CHIRPS rainfall alongside ERA5 temperature and evapotranspiration:

            >>> MeteoInputs.from_rasters(  # doctest: +SKIP
            ...     chirps_dir,
            ...     era5_temp_dir,
            ...     era5_evap_dir,
            ...     per_variable={
            ...         "temperature": {"regex_string": r"\d{8}"},
            ...         "evapotranspiration": {"regex_string": r"\d{8}"},
            ...     },
            ... )
        """
        overrides = per_variable or {}
        unknown = set(overrides) - set(METEO_VARIABLES)
        if unknown:
            raise KeyError(
                f"per_variable names {sorted(unknown)}, which are not drivers; "
                f"expected any of {list(METEO_VARIABLES)}"
            )

        cubes, calendar = {}, None
        for name, path in zip(
            METEO_VARIABLES, (precipitation, temperature, evapotranspiration)
        ):
            collection = read_rasters(path, **{**kwargs, **overrides.get(name, {})})
            cubes[name] = np.moveaxis(np.asarray(collection.values), 0, -1)
            if calendar is None and collection.time is not None:
                calendar = pd.DatetimeIndex(list(collection.time))
        return cls(**cubes, time=calendar)

    @classmethod
    def from_netcdf_files(
        cls,
        precipitation: str | Path,
        temperature: str | Path,
        evapotranspiration: str | Path,
        variable: str | None = None,
    ) -> MeteoInputs:
        """Read the three drivers from one NetCDF per variable.

        Args:
            precipitation: NetCDF holding the rainfall cube.
            temperature: NetCDF holding the temperature cube.
            evapotranspiration: NetCDF holding the evapotranspiration cube.
            variable: Name of the variable to take from each file. `None` (default) takes each
                file's only variable, which is what `DatasetCollection.to_netcdf` writes for a
                single-band collection.

        Returns:
            MeteoInputs: The three cubes plus a calendar -- the rainfall file's when it carries
                one, otherwise the first source that does, and None when none do.

        Raises:
            KeyError: `variable` is not in one of the files.
            ValueError: A file holds several variables and `variable` was not given, or the
                three files do not yield the same shape.
        """
        cubes, calendar = {}, None
        for name, path in zip(
            METEO_VARIABLES, (precipitation, temperature, evapotranspiration)
        ):
            nc = NetCDF.read_file(str(path))
            if variable is None:
                if len(nc.variable_names) != 1:
                    raise ValueError(
                        f"{path} holds {len(nc.variable_names)} variables "
                        f"({nc.variable_names}); pass variable= to pick one, or use "
                        "from_netcdf() for a single file holding all three drivers."
                    )
                target = nc.variable_names[0]
            else:
                target = variable
            cubes[name] = _cube_from_netcdf(nc, target)
            if calendar is None:
                calendar = cls._calendar(nc)
        return cls(**cubes, time=calendar)

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        precipitation: str,
        temperature: str,
        evapotranspiration: str,
    ) -> MeteoInputs:
        """Read all three drivers from one NetCDF holding them as separate variables.

        Args:
            path: The NetCDF file.
            precipitation: Name of the rainfall variable inside it.
            temperature: Name of the temperature variable.
            evapotranspiration: Name of the evapotranspiration variable.

        Returns:
            MeteoInputs: The three cubes plus the file's calendar.

        Raises:
            KeyError: One of the named variables is not in the file.
            ValueError: The three variables do not share a shape.
        """
        nc = NetCDF.read_file(path)
        cubes = {
            name: _cube_from_netcdf(nc, var)
            for name, var in zip(
                METEO_VARIABLES, (precipitation, temperature, evapotranspiration)
            )
        }
        return cls(**cubes, time=cls._calendar(nc))

    @staticmethod
    def raster_folder_to_netcdf(
        path: str | Path,
        out_path: str | Path,
        *,
        glob: str = "*.tif",
        regex_string: str = r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        start: str | int | None = None,
        end: str | int | None = None,
        fmt: str = "%Y-%m-%d",
        gdal_env: dict[str, str] | None = None,
    ) -> Path:
        r"""Pack one driver's folder of dated rasters into a single NetCDF.

        A folder of per-date GeoTIFFs is what the download backends produce, and it is the
        slowest thing the model can be driven from: every run re-opens every file. Packing it
        once into a NetCDF makes later runs read one file, and makes the folder portable --
        the calendar travels inside the file instead of living in the file names.

        The rasters are ordered by :func:`read_rasters`, whose reader arguments are repeated
        here rather than forwarded as `**kwargs`, so a typo is caught at this call rather
        than one frame deeper.

        Args:
            path: Folder holding one variable's rasters.
            out_path: NetCDF file to write. Overwritten if it exists.
            glob: :mod:`fnmatch` pattern selecting the rasters. Defaults to `"*.tif"`.
            regex_string: Where the date sits in each file name.
            date: Whether the matched value is a date. `False` orders by a plain index
                instead, which carries no calendar and so cannot be written here.
            file_name_data_fmt: `strptime` format of the date. Inferred from the names when
                omitted; pass it for a layout the digits cannot settle, such as a day-first
                `03.02.1990`.
            start: Inclusive lower bound, to convert a window rather than the whole folder.
            end: Inclusive upper bound; see `start`.
            fmt: `strptime` format of `start` / `end`.
            gdal_env: GDAL configuration applied for the read, e.g.
                `{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}` to skip the per-open
                directory listing on network storage.

        Returns:
            Path: The file that was written.

        Raises:
            FileNotFoundError: The folder does not exist or matched no raster.
            ValueError: The rasters carry no usable calendar, so the NetCDF would have no
                time axis to write.

        Examples:
            Pack a folder, then drive a model from the result:

            >>> MeteoInputs.raster_folder_to_netcdf(temp_dir, "temp.nc")  # doctest: +SKIP
            >>> MeteoInputs.from_netcdf_files(  # doctest: +SKIP
            ...     "prec.nc", "temp.nc", "evap.nc"
            ... )
        """
        collection = read_rasters(
            path,
            glob=glob,
            regex_string=regex_string,
            date=date,
            file_name_data_fmt=file_name_data_fmt,
            start=start,
            end=end,
            fmt=fmt,
            gdal_env=gdal_env,
        )
        if collection.time is None:
            raise ValueError(
                f"the rasters in {path} carry no calendar, so the NetCDF would have no time "
                "axis; pass regex_string and file_name_data_fmt so the dates can be parsed"
            )

        stamps = list(collection.time)
        if stamps != sorted(stamps):
            raise ValueError(
                f"the rasters in {path} did not come back in chronological order, so the "
                "NetCDF would pair each step with the wrong date"
            )

        out = Path(out_path)
        out.unlink(missing_ok=True)
        collection.to_netcdf(str(out))
        logger.debug(
            f"{collection.time_length} rasters from {path} written to {out} "
            f"({stamps[0]:%Y-%m-%d} to {stamps[-1]:%Y-%m-%d})"
        )
        return out

    @staticmethod
    def combine_netcdf_files(
        precipitation: str | Path,
        temperature: str | Path,
        evapotranspiration: str | Path,
        out_path: str | Path,
    ) -> Path:
        """Merge one NetCDF per driver into a single file holding all three.

        The counterpart to :meth:`from_netcdf`: three single-variable files go in, one file
        comes out whose variables are named `precipitation`, `temperature` and
        `evapotranspiration`, so a reader can ask for them by name rather than guessing at
        whatever `to_netcdf` called the band.

        The first file seeds the container and the other two are copied in with
        `NetCDF.add_variable`; nothing touches disk until the write, so the sources are left
        as they are.

        Args:
            precipitation: NetCDF holding the rainfall cube.
            temperature: NetCDF holding the temperature cube.
            evapotranspiration: NetCDF holding the evapotranspiration cube.
            out_path: File to write. Overwritten if it exists.

        Returns:
            Path: The file that was written.

        Raises:
            ValueError: One of the sources holds more than one variable, so which cube it
                contributes would be a guess.

        Examples:
            >>> MeteoInputs.combine_netcdf_files(  # doctest: +SKIP
            ...     "prec.nc", "temp.nc", "evap.nc", "meteo.nc"
            ... )
            >>> MeteoInputs.from_netcdf(  # doctest: +SKIP
            ...     "meteo.nc",
            ...     precipitation="precipitation",
            ...     temperature="temperature",
            ...     evapotranspiration="evapotranspiration",
            ... )
        """
        sources = dict(
            zip(METEO_VARIABLES, (precipitation, temperature, evapotranspiration))
        )
        for name, source in sources.items():
            holder = NetCDF.read_file(str(source))
            if len(holder.variable_names) != 1:
                raise ValueError(
                    f"{source} holds {len(holder.variable_names)} variables "
                    f"({holder.variable_names}); {name} must come from a file with exactly "
                    "one, as raster_folder_to_netcdf writes"
                )

        (seed_name, seed_path), *rest = sources.items()
        combined = NetCDF.read_file(str(seed_path))
        combined.rename_variable(combined.variable_names[0], seed_name)

        for name, source in rest:
            holder = NetCDF.read_file(str(source))
            combined.add_variable(holder)
            combined.rename_variable(holder.variable_names[0], name)

        out = Path(out_path)
        out.unlink(missing_ok=True)
        combined.to_file(str(out))
        logger.debug(f"three drivers combined into {out}")
        return out

    @staticmethod
    def _calendar(nc: NetCDF) -> pd.DatetimeIndex | None:
        """Return a NetCDF's decoded time axis, or None when it carries no calendar.

        `NetCDF.time_stamp` decodes the axis only for a file holding a single data variable; on
        one holding all three drivers it returns None even though the `time` array is there and
        correct. So fall back to the raw values, which `to_netcdf` writes as nanoseconds since
        the epoch.

        Args:
            nc: An open :class:`~pyramids.netcdf.NetCDF`.

        Returns:
            pd.DatetimeIndex | None: The decoded stamps, or None when the file carries no
                calendar -- `to_netcdf` writes a positional index for an undated collection,
                and those values are left alone rather than misread as 1970.
        """
        try:
            stamps = nc.time_stamp
        except (AttributeError, KeyError, ValueError):
            stamps = None
        if stamps:
            return pd.DatetimeIndex(list(stamps))

        try:
            raw = nc.get_time_values()
        except (AttributeError, KeyError, ValueError):
            return None
        if raw is None:
            # No time dimension. Checking explicitly rather than letting `np.asarray(None)`
            # produce a 0-d object array that happens to fail the dtype test below.
            return None
        values = np.asarray(raw)
        # A positional index runs 0..n-1; a nanosecond epoch stamp is astronomically larger, so
        # the magnitude tells the two apart without depending on an attribute GDAL may not expose.
        if values.size and values.dtype.kind in "iu" and values.min() > 10**12:
            return pd.DatetimeIndex(pd.to_datetime(values))
        return None


PARAMETERS_LIST = [
    "01_tt",
    "02_rfcf",
    "03_sfcf",
    "04_cfmax",
    "05_cwh",
    "06_cfr",
    "07_fc",
    "08_beta",
    "09_etf",
    "10_lp",
    "11_k0",
    "12_k1",
    "13_k2",
    "14_uzl",
    "15_perc",
    "16_maxbas",
    "17_K_muskingum",
    "18_x_muskingum",
]


class Inputs:
    """Rainfall-runoff inputs preparation for distributed hydrological models.

    The Inputs class provides methods to prepare meteorological and parameter
    raster data so they align with a reference DEM. It supports extracting
    HBV model parameter boundaries and computing lumped inputs from distributed
    rasters. Chronological ordering is handled by pyramids at read time
    (`from_files(date_format=...)`), not by renaming files on disk.

    Attributes:
        source_dem: Path to the reference DEM raster used for spatial
            alignment (coordinate system, rows, columns, resolution).

    Examples:
        >>> from hapi.inputs import Inputs
        >>> inp = Inputs("data/dem.tif")
    """

    def __init__(self, src: str):
        """Initialize the Inputs instance with a reference DEM path.

        Args:
            src: Path to the spatial information source raster used to
                obtain the coordinate system, number of rows and columns,
                and resolution. The path should include the file name and
                extension (e.g., `"data/dem.tif"`).
        """
        self.source_dem = src

    def prepare_inputs(self, inputs_dir: str | Path, outputs_dir: str | Path):
        """Align and crop input rasters to match the source DEM.

        Reads all rasters from `inputs_dir`, aligns them to the source
        DEM's spatial properties (CRS, resolution, extent, nodata value),
        crops them to the DEM footprint, and writes the results to
        `outputs_dir`.

        Args:
            inputs_dir: Path to the folder containing the rasters to be
                aligned and cropped to match the source DEM.
            outputs_dir: Path to the output folder where the aligned
                rasters will be saved.

        Each output keeps its source file name, so the ordering of the collection is
        irrelevant here and the rasters are read unordered.
        `outputs_dir` is created if it does not exist; either argument may be a
        `str` or a :class:`pathlib.Path`.

        Returns:
            None: The aligned rasters are written to `outputs_dir`.

        Raises:
            FileNotFoundError: If `inputs_dir` does not exist.

        Examples:
            - Align two rasters onto a DEM grid and read back what was written:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.inputs import Inputs
                >>> root = tempfile.mkdtemp()
                >>> dem_path = os.path.join(root, "dem.tif")
                >>> Dataset.create_from_array(
                ...     np.ones((4, 4), dtype="float32"), top_left_corner=(0.0, 4.0),
                ...     cell_size=1.0, epsg=4326, no_data_value=-9999.0, path=dem_path,
                ... ).close()
                >>> src_dir = os.path.join(root, "src")
                >>> os.makedirs(src_dir)
                >>> for stamp in ("2020.01.01", "2020.01.02"):
                ...     Dataset.create_from_array(
                ...         np.full((4, 4), 5.0, dtype="float32"), top_left_corner=(0.0, 4.0),
                ...         cell_size=1.0, epsg=4326, no_data_value=-9999.0,
                ...         path=os.path.join(src_dir, f"prec_{stamp}.tif"),
                ...     ).close()
                >>> out_dir = os.path.join(root, "out")
                >>> Inputs(dem_path).prepare_inputs(src_dir, out_dir)
                >>> sorted(os.listdir(out_dir))
                ['prec_2020.01.01.tif', 'prec_2020.01.02.tif']

                ```
            - A missing input directory fails fast, before the DEM is opened:
                ```python
                >>> import os, tempfile
                >>> from hapi.inputs import Inputs
                >>> missing = os.path.join(tempfile.mkdtemp(), "absent")
                >>> try:
                ...     Inputs("dem-never-opened.tif").prepare_inputs(missing, "out")
                ... except FileNotFoundError as exc:
                ...     print("does not exist" in str(exc))
                True

                ```

        See Also:
            Inputs.create_lumped_inputs: Reduce the same rasters to catchment averages.
        """
        # Validate before opening the DEM so a missing input directory fails fast
        # rather than after a raster read.
        if not Path(inputs_dir).exists():
            raise FileNotFoundError(f"{inputs_dir} does not exist")

        mask = Dataset.read_file(self.source_dem)
        # Unordered: prepare_inputs realigns every raster in the folder, so no time axis
        # is needed and the files can be taken in whatever order they resolve.
        cube = Datacube.from_files(inputs_dir)
        # in-place align/crop clear the collection's file list, so capture the names first
        file_names = [Path(file).name for file in cube.files]
        cube.align(mask, inplace=True)
        cube.crop(mask, inplace=True)
        path = [f"{outputs_dir}/{name}" for name in file_names]
        cube.to_file(path)

    @staticmethod
    def extract_parameters_boundaries(basin: FeatureCollection):
        """Extract upper and lower parameter boundaries for a catchment.

        Reads the global maximum and minimum HBV parameter rasters from
        the directory specified by the `HAPI_DATA_DIR` environment
        variable, clips them to the given basin polygon, and returns the
        max/min statistics for each parameter.

        The 18 HBV parameters are:
        `tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta, etf, lp, k0, k1, k2, uzl, perc, maxbas, K_muskingum, x_muskingum`.

        Args:
            basin: The catchment polygon, as a
                :class:`~pyramids.feature.FeatureCollection`. Any `GeoDataFrame` is
                accepted too and is wrapped on the way in. Must contain exactly one row;
                merge all polygons first if the shapefile has multiple features.

        Returns:
            pandas.DataFrame: A DataFrame indexed by parameter name with
                columns `"ub"` (upper bound) and `"lb"` (lower bound).

        Raises:
            ValueError: If the `HAPI_DATA_DIR` environment variable is
                not set.
            FileNotFoundError: If the parameter data directory or the
                `max`/`min` subdirectories do not exist.
        """
        data_dir = Inputs._check_data_dir()
        max_dir = data_dir / "max"
        min_dir = data_dir / "min"
        file_path = data_dir / f"max/{PARAMETERS_LIST[0]}.tif"

        if not file_path.exists() or not max_dir.exists() or not min_dir.exists():
            raise FileNotFoundError(
                f"check the following files{file_path}, {max_dir}, {min_dir} does not exist"
            )

        dataset = Dataset.read_file(str(file_path))
        # Wrap on the way in so a plain GeoDataFrame is accepted as readily as a
        # FeatureCollection; the constructor is a no-op for one that is already wrapped.
        basin = FeatureCollection(basin).to_crs(crs=dataset.crs)

        # max values
        ub = list()
        for i in range(len(PARAMETERS_LIST)):
            dataset = Dataset.read_file(f"{data_dir}/max/{PARAMETERS_LIST[i]}.tif")
            vals = dataset.stats(mask=basin)
            ub.append(vals.loc[vals.index[0], "max"])

        # min values
        lb = list()
        for i in range(len(PARAMETERS_LIST)):
            dataset = Dataset.read_file(f"{data_dir}/min/{PARAMETERS_LIST[i]}.tif")
            vals = dataset.stats(mask=basin)
            lb.append(vals.loc[vals.index[0], "min"])

        par = pd.DataFrame(index=PARAMETERS_LIST)

        par["ub"] = ub
        par["lb"] = lb

        return par

    def extract_parameters(
        self,
        gdf: FeatureCollection | None,
        scenario: str,
        as_raster: bool = False,
        save_to: str = "",
    ):
        """Extract HBV parameter values or rasters for a catchment.

        Retrieves one of 12 global HBV parameter sets (Beck et al., 2016)
        from the directory specified by the `HAPI_DATA_DIR` environment
        variable. When `as_raster` is False, computes zonal statistics
        (min, max, mean, std) over the catchment polygon. When
        `as_raster` is True, aligns and crops the parameter rasters to
        the source DEM and saves them to `save_to`.

        Reference:
            Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo,
            Diego G. Miralles, T. R. M. & Jaap Schellekens, and
            L. A. B. (2016). Global-scale regionalization of hydrologic
            model parameters. Water Resources Research, 3599-3622.
            doi:10.1002/2015WR018247.

        The 18 HBV parameters are:
        `tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta, etf, lp, k0, k1, k2, uzl, perc, maxbas, K_muskingum, x_muskingum`.

        Args:
            gdf: The catchment polygon, as a
                :class:`~pyramids.feature.FeatureCollection`. Any `GeoDataFrame` is
                accepted too and is wrapped on the way in. Must contain one row; merge
                all polygons first if the shapefile has multiple features. Ignored (and
                may be `None`) when `as_raster` is True.
            scenario: Name of the parameter set. One of `"1"` through
                `"10"`, `"avg"`, `"max"`, or `"min"`.
            as_raster: If True, save aligned parameter rasters to
                `save_to` instead of returning statistics. Default is
                False.
            save_to: Path to the directory where aligned parameter rasters
                will be saved. Only used when `as_raster` is True.

        Returns:
            pandas.DataFrame: When `as_raster` is False, a DataFrame
                indexed by parameter name with columns `"min"`,
                `"max"`, `"mean"`, and `"std"`. Returns None when
                `as_raster` is True.

        Raises:
            ValueError: If the `HAPI_DATA_DIR` environment variable is
                not set.
            FileNotFoundError: If the parameter data directory does not
                exist.
        """
        data_dir = self._check_data_dir()
        parameters_path = data_dir / scenario

        if not as_raster:
            dataset = Dataset.read_file(f"{parameters_path}/{PARAMETERS_LIST[0]}.tif")
            gdf = FeatureCollection(gdf).to_crs(crs=dataset.crs)

            stats = pd.DataFrame(columns=["min", "max", "mean", "std"])
            for i in range(len(PARAMETERS_LIST)):
                dataset = Dataset.read_file(
                    f"{parameters_path}/{PARAMETERS_LIST[i]}.tif"
                )
                vals = dataset.stats(mask=gdf)
                stats.loc[PARAMETERS_LIST[i], :] = vals.loc[
                    :, ["min", "max", "mean", "std"]
                ].values
            return stats
        else:
            self.prepare_inputs(f"{parameters_path}/", save_to)

    @staticmethod
    def create_lumped_inputs(
        path: str,
        regex_string: str = r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        start: str | None = None,
        end: str | None = None,
        fmt: str = "%Y-%m-%d",
        extension: str = ".tif",
    ) -> list:
        r"""Create lumped inputs by averaging distributed raster values.

        Reads a time series of rasters from the given directory, computes
        the spatial mean of each raster, and returns the averages as a
        list. This is used to convert distributed meteorological or
        parameter data into lumped (catchment-average) values.

        Args:
            path: Path to the folder containing the raster files.
            regex_string: A regex pattern to locate the date (or ordering
                number) within each file name. Default is
                `r"\\d{4}.\\d{2}.\\d{2}"`.
            date: If True, the number extracted from file names is
                interpreted as a date. Default is True.
            file_name_data_fmt: The date format string matching dates in
                the file names (e.g., `"%Y.%m.%d"`). Default is None.
            start: Start date to filter the rasters. If not provided, all
                rasters in the directory are read.
            end: End date to filter the rasters. If not provided, all
                rasters in the directory are read.
            fmt: Format of the `start` and `end` date strings.
                Default is `"%Y-%m-%d"`.
            extension: File extension to filter by. Default is `".tif"`.

        Returns:
            list: The spatial mean of each raster, in chronological order. The elements
                are NumPy scalars (`numpy.float32` for a float32 source) rather than
                built-in `float`, since they come straight from the per-raster
                statistics; wrap them in `float()` if a built-in is required.

        Examples:
            - Reduce two dated rasters to one catchment average each, in date order:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.inputs import Inputs
                >>> src_dir = tempfile.mkdtemp()
                >>> for stamp, value in (("2020.01.02", 4.0), ("2020.01.01", 2.0)):
                ...     Dataset.create_from_array(
                ...         np.full((2, 2), value, dtype="float32"),
                ...         top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326,
                ...         no_data_value=-9999.0,
                ...         path=os.path.join(src_dir, f"prec_{stamp}.tif"),
                ...     ).close()
                >>> averages = Inputs.create_lumped_inputs(
                ...     src_dir, regex_string=r"\d{4}.\d{2}.\d{2}", date=True,
                ...     file_name_data_fmt="%Y.%m.%d",
                ... )
                >>> [float(value) for value in averages]
                [2.0, 4.0]

                ```
            - A uniform raster averages to its own value:
                ```python
                >>> import numpy as np, os, tempfile
                >>> from pyramids.dataset import Dataset
                >>> from hapi.inputs import Inputs
                >>> src_dir = tempfile.mkdtemp()
                >>> Dataset.create_from_array(
                ...     np.full((3, 3), 7.5, dtype="float32"),
                ...     top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326,
                ...     no_data_value=-9999.0,
                ...     path=os.path.join(src_dir, "prec_2021.06.01.tif"),
                ... ).close()
                >>> averages = Inputs.create_lumped_inputs(
                ...     src_dir, regex_string=r"\d{4}.\d{2}.\d{2}", date=True,
                ...     file_name_data_fmt="%Y.%m.%d",
                ... )
                >>> float(averages[0])
                7.5

                ```

        See Also:
            Inputs.prepare_inputs: Align and crop the same rasters onto the DEM grid.
        """
        cube = read_rasters(
            path,
            # pyramids 0.50 replaced `extension` with an fnmatch `glob`.
            glob=f"*{extension}",
            regex_string=regex_string,
            date=date,
            file_name_data_fmt=file_name_data_fmt,
            start=start,
            end=end,
            fmt=fmt,
        )
        avg = []
        for i in range(cube.time_length):
            dataset = cube.iloc(i)
            stats = dataset.stats()
            avg.append(stats.loc[stats.index[0], "mean"])

        return avg

    @staticmethod
    def _check_data_dir() -> Path:
        """Validate and return the HAPI parameter data directory.

        Reads the `HAPI_DATA_DIR` environment variable and verifies
        that the directory exists on disk.

        Returns:
            Path: The resolved path to the HAPI data directory.

        Raises:
            ValueError: If the `HAPI_DATA_DIR` environment variable
                is not set.
            FileNotFoundError: If the directory specified by
                `HAPI_DATA_DIR` does not exist.
        """
        data_dir_env: str | None = os.getenv("HAPI_DATA_DIR")
        if data_dir_env is None:
            raise ValueError("HAPI_DATA_DIR environment variable is not set")
        data_dir: Path = Path(data_dir_env)
        if not data_dir.exists():
            raise FileNotFoundError(f"{data_dir} does not exist")
        return data_dir
