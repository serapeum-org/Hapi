"""Rainfall-runoff Inputs.

The inputs module provides the `Inputs` class for preparing meteorological
and parameter raster data for distributed hydrological modeling. It handles
alignment of rasters to a source DEM, extraction of HBV model parameters
from global datasets, and creation of lumped inputs from distributed data.

Rasters are read in chronological order by ``DatasetCollection.from_files``, which parses the
date out of each file name, so the files themselves never need renaming on disk.

The module relies on the ``pyramids`` library for raster I/O and
manipulation, and uses the ``HAPI_DATA_DIR`` environment variable to
locate pre-downloaded global parameter sets (Beck et al., 2016).
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import pandas as pd
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube
from pyramids.feature import FeatureCollection


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
) -> Datacube:
    r"""Read a folder of rasters into a `DatasetCollection` in the right order.

    A thin adapter over :meth:`DatasetCollection.from_files` -- pyramids does every bit of the
    resolving and reading; this only decides the order the files are handed over in, and
    translates Hapi's string/int `start` / `end` into what `from_files` accepts.

    Three orderings are supported, matching Hapi's public reader arguments:

    * **By date** (`date=True` with a `file_name_data_fmt`) -- delegated wholesale to
      `from_files(date_format=..., date_regex=...)`, which sorts and builds the time axis.
    * **By number** (`date=False`) -- for names carrying a plain index, e.g.
      `01_Par_RFCF.tif` or `1000_Temp_..._1981_9_27.tif`. `from_files` sorts only by date, and
      its default order is lexicographic, which puts `10_` before `2_` whenever the index is
      not zero-padded. So the files are resolved through `from_files`, sorted on the integer in
      each name, and handed back to `from_files` as an explicit sequence -- which it keeps in
      the given order.
    * **Unordered** (`date=True`, no format) -- a plain `from_files(path, glob=...)`.

    Args:
        path: Folder holding the rasters.
        glob: :mod:`fnmatch` pattern selecting them. Defaults to `"*.tif"`.
        regex_string: Where the date (or the index, when `date=False`) sits in each name.
        date: Whether the matched value is a date. `False` selects the numeric ordering.
        file_name_data_fmt: `strptime` format of the date in the names. Without it there is
            nothing to sort dates on, so the read is unordered.
        start: Inclusive lower bound -- a date string parsed with `fmt`, or an integer index
            when `date=False`.
        end: Inclusive upper bound; see `start`.
        fmt: `strptime` format of `start` / `end` when they are date strings.

    Returns:
        DatasetCollection: The collection, ordered as described above.

    Raises:
        FileNotFoundError: The folder does not exist, matched no file, or `start` / `end`
            excluded every file.
        ValueError: `regex_string` matched no number in a file name (numeric ordering only).
    """
    if date and file_name_data_fmt is not None:
        return Datacube.from_files(
            path,
            glob=glob,
            date_format=file_name_data_fmt,
            date_regex=regex_string,
            start=_as_datetime(start, fmt),
            end=_as_datetime(end, fmt),
        )

    if not date:
        keyed = []
        for file in Datacube.from_files(path, glob=glob).files:
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

        return Datacube.from_files([file for _, file in keyed])

    return Datacube.from_files(path, glob=glob)


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
    (``from_files(date_format=...)``), not by renaming files on disk.

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
                extension (e.g., ``"data/dem.tif"``).
        """
        self.source_dem = src

    def prepare_inputs(self, inputs_dir: str | Path, outputs_dir: str | Path):
        """Align and crop input rasters to match the source DEM.

        Reads all rasters from ``inputs_dir``, aligns them to the source
        DEM's spatial properties (CRS, resolution, extent, nodata value),
        crops them to the DEM footprint, and writes the results to
        ``outputs_dir``.

        Args:
            inputs_dir: Path to the folder containing the rasters to be
                aligned and cropped to match the source DEM.
            outputs_dir: Path to the output folder where the aligned
                rasters will be saved.

        Each output keeps its source file name, so the ordering of the collection is
        irrelevant here and the rasters are read with ``with_order=False``.
        ``outputs_dir`` is created if it does not exist; either argument may be a
        ``str`` or a :class:`pathlib.Path`.

        Returns:
            None: The aligned rasters are written to ``outputs_dir``.

        Raises:
            FileNotFoundError: If ``inputs_dir`` does not exist.

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
        the directory specified by the ``HAPI_DATA_DIR`` environment
        variable, clips them to the given basin polygon, and returns the
        max/min statistics for each parameter.

        The 18 HBV parameters are:
        ``tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta, etf, lp, k0, k1,
        k2, uzl, perc, maxbas, K_muskingum, x_muskingum``.

        Args:
            basin: The catchment polygon, as a
                :class:`~pyramids.feature.FeatureCollection`. Any ``GeoDataFrame`` is
                accepted too and is wrapped on the way in. Must contain exactly one row;
                merge all polygons first if the shapefile has multiple features.

        Returns:
            pandas.DataFrame: A DataFrame indexed by parameter name with
                columns ``"ub"`` (upper bound) and ``"lb"`` (lower bound).

        Raises:
            ValueError: If the ``HAPI_DATA_DIR`` environment variable is
                not set.
            FileNotFoundError: If the parameter data directory or the
                ``max``/``min`` subdirectories do not exist.
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
        from the directory specified by the ``HAPI_DATA_DIR`` environment
        variable. When ``as_raster`` is False, computes zonal statistics
        (min, max, mean, std) over the catchment polygon. When
        ``as_raster`` is True, aligns and crops the parameter rasters to
        the source DEM and saves them to ``save_to``.

        Reference:
            Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo,
            Diego G. Miralles, T. R. M. & Jaap Schellekens, and
            L. A. B. (2016). Global-scale regionalization of hydrologic
            model parameters. Water Resources Research, 3599-3622.
            doi:10.1002/2015WR018247.

        The 18 HBV parameters are:
        ``tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta, etf, lp, k0, k1,
        k2, uzl, perc, maxbas, K_muskingum, x_muskingum``.

        Args:
            gdf: The catchment polygon, as a
                :class:`~pyramids.feature.FeatureCollection`. Any ``GeoDataFrame`` is
                accepted too and is wrapped on the way in. Must contain one row; merge
                all polygons first if the shapefile has multiple features. Ignored (and
                may be ``None``) when ``as_raster`` is True.
            scenario: Name of the parameter set. One of ``"1"`` through
                ``"10"``, ``"avg"``, ``"max"``, or ``"min"``.
            as_raster: If True, save aligned parameter rasters to
                ``save_to`` instead of returning statistics. Default is
                False.
            save_to: Path to the directory where aligned parameter rasters
                will be saved. Only used when ``as_raster`` is True.

        Returns:
            pandas.DataFrame: When ``as_raster`` is False, a DataFrame
                indexed by parameter name with columns ``"min"``,
                ``"max"``, ``"mean"``, and ``"std"``. Returns None when
                ``as_raster`` is True.

        Raises:
            ValueError: If the ``HAPI_DATA_DIR`` environment variable is
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
                ``r"\\d{4}.\\d{2}.\\d{2}"``.
            date: If True, the number extracted from file names is
                interpreted as a date. Default is True.
            file_name_data_fmt: The date format string matching dates in
                the file names (e.g., ``"%Y.%m.%d"``). Default is None.
            start: Start date to filter the rasters. If not provided, all
                rasters in the directory are read.
            end: End date to filter the rasters. If not provided, all
                rasters in the directory are read.
            fmt: Format of the ``start`` and ``end`` date strings.
                Default is ``"%Y-%m-%d"``.
            extension: File extension to filter by. Default is ``".tif"``.

        Returns:
            list: The spatial mean of each raster, in chronological order. The elements
                are NumPy scalars (``numpy.float32`` for a float32 source) rather than
                built-in ``float``, since they come straight from the per-raster
                statistics; wrap them in ``float()`` if a built-in is required.

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

        Reads the ``HAPI_DATA_DIR`` environment variable and verifies
        that the directory exists on disk.

        Returns:
            Path: The resolved path to the HAPI data directory.

        Raises:
            ValueError: If the ``HAPI_DATA_DIR`` environment variable
                is not set.
            FileNotFoundError: If the directory specified by
                ``HAPI_DATA_DIR`` does not exist.
        """
        data_dir_env: str | None = os.getenv("HAPI_DATA_DIR")
        if data_dir_env is None:
            raise ValueError("HAPI_DATA_DIR environment variable is not set")
        data_dir: Path = Path(data_dir_env)
        if not data_dir.exists():
            raise FileNotFoundError(f"{data_dir} does not exist")
        return data_dir
