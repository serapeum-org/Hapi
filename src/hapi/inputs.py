"""Rainfall-runoff Inputs.

The inputs module provides the `Inputs` class for preparing meteorological
and parameter raster data for distributed hydrological modeling. It handles
alignment of rasters to a source DEM, extraction of HBV model parameters
from global datasets, and creation of lumped inputs from distributed data.

Rasters are read in chronological order by
``DatasetCollection.read_multiple_files(with_order=True, ...)``, which parses the date
out of each file name, so the files themselves never need renaming on disk.

The module relies on the ``pyramids`` library for raster I/O and
manipulation, and uses the ``HAPI_DATA_DIR`` environment variable to
locate pre-downloaded global parameter sets (Beck et al., 2016).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube
from pyramids.feature import FeatureCollection

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
    (``read_multiple_files(with_order=True, ...)``), not by renaming files on disk.

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
        cube = Datacube.read_multiple_files(inputs_dir, with_order=False)
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
        regex_string=r"\d{4}.\d{2}.\d{2}",
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
