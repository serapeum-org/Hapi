"""Rainfall-runoff Inputs.

The inputs module provides the `Inputs` class for preparing meteorological
and parameter raster data for distributed hydrological modeling. It handles
alignment of rasters to a source DEM, extraction of HBV model parameters
from global datasets, creation of lumped inputs from distributed data, and
renaming of raster files with date-based ordering.

The module relies on the ``pyramids`` library for raster I/O and
manipulation, and uses the ``HAPI_DATA_DIR`` environment variable to
locate pre-downloaded global parameter sets (Beck et al., 2016).
"""

import datetime as dt
import os
from pathlib import Path
from typing import Union

import pandas as pd
from geopandas import GeoDataFrame
from pyramids.datacube import Datacube
from pyramids.dataset import Dataset

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
    HBV model parameter boundaries, computing lumped inputs from distributed
    rasters, and renaming files with date-based ordering.

    Attributes:
        source_dem: Path to the reference DEM raster used for spatial
            alignment (coordinate system, rows, columns, resolution).

    Examples:
        >>> from Hapi.inputs import Inputs
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

    def prepare_inputs(
        self, inputs_dir: Union[str, Path], outputs_dir: Union[str, Path]
    ):
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

        Raises:
            FileNotFoundError: If ``inputs_dir`` does not exist.

        Examples:
            >>> from Hapi.inputs import Inputs
            >>> inp = Inputs("GIS/inputs/acc4000.tif")
            >>> inp.prepare_inputs(
            ...     "Precipitation/CHIRPS/Daily/",
            ...     "outputs/prec",
            ... )
        """
        if not isinstance(outputs_dir, str):
            print("output_folder input should be string type")

        mask = Dataset.read_file(self.source_dem)
        if not Path(inputs_dir).exists():
            raise FileNotFoundError(f"{inputs_dir} does not exist")

        cube = Datacube.read_multiple_files(inputs_dir, with_order=False)
        cube.open_datacube()
        cube.align(mask)
        cube.crop(mask, inplace=True)
        path = [f"{outputs_dir}/{file.split('/')[-1]}" for file in cube.files]
        cube.to_file(path)

    @staticmethod
    def extract_parameters_boundaries(basin: GeoDataFrame):
        """Extract upper and lower parameter boundaries for a catchment.

        Reads the global maximum and minimum HBV parameter rasters from
        the directory specified by the ``HAPI_DATA_DIR`` environment
        variable, clips them to the given basin polygon, and returns the
        max/min statistics for each parameter.

        The 18 HBV parameters are:
        ``tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta, etf, lp, k0, k1,
        k2, uzl, perc, maxbas, K_muskingum, x_muskingum``.

        Args:
            basin: A GeoDataFrame containing the catchment polygon. Must
                contain exactly one row; merge all polygons first if the
                shapefile has multiple features.

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
            raise FileNotFoundError(f"check the following files{file_path}, {max_dir}, {min_dir} does not exist")

        dataset = Dataset.read_file(str(file_path))
        basin = basin.to_crs(crs=dataset.crs)

        # max values
        ub = list()
        for i in range(len(PARAMETERS_LIST)):
            dataset = Dataset.read_file(
                f"{data_dir}/max/{PARAMETERS_LIST[i]}.tif"
            )
            vals = dataset.stats(mask=basin)
            ub.append(vals.loc[vals.index[0], "max"])

        # min values
        lb = list()
        for i in range(len(PARAMETERS_LIST)):
            dataset = Dataset.read_file(
                f"{data_dir}/min/{PARAMETERS_LIST[i]}.tif"
            )
            vals = dataset.stats(mask=basin)
            lb.append(vals.loc[vals.index[0], "min"])

        par = pd.DataFrame(index=PARAMETERS_LIST)

        par["ub"] = ub
        par["lb"] = lb

        return par

    def extract_parameters(
        self,
        gdf: Union[GeoDataFrame, str],
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
            gdf: A GeoDataFrame of the catchment polygon. Must contain
                one row; merge all polygons first if the shapefile has
                multiple features. Can be None when ``as_raster`` is True.
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
            gdf = gdf.to_crs(crs=dataset.crs)

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
        file_name_data_fmt: str = None,
        start: str = None,
        end: str = None,
        fmt: str = "%Y-%m-%d",
        extension: str = ".tif",
    ) -> list:
        """Create lumped inputs by averaging distributed raster values.

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
            list: A list of float values, each being the spatial mean of
                the corresponding raster in chronological order.

        Examples:
            >>> from Hapi.inputs import Inputs
            >>> avg = Inputs.create_lumped_inputs(
            ...     "tests/rrm/data/coello/prec",
            ...     regex_string=r"\\d{4}.\\d{2}.\\d{2}",
            ...     date=True,
            ...     file_name_data_fmt="%Y.%m.%d",
            ... )
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
        cube.open_datacube()
        avg = []
        for i in range(cube.time_length):
            dataset = cube.iloc(i)
            stats = dataset.stats()
            avg.append(stats.loc[stats.index[0], "mean"])

        return avg

    @staticmethod
    def rename_files(
        path: str, prefix: str = "", fmt: str = "%Y.%m.%d", freq: str = "daily"
    ):
        """Rename raster files with a sequential order prefix based on date.

        Reads all ``.tif`` files in the given directory, extracts dates
        from their names, sorts them chronologically, and renames each
        file with a leading index number indicating its temporal order.

        The new file name format is:
        ``{order}_{prefix}_{date_string}.tif``

        Args:
            path: Path to the directory containing the raster files.
            prefix: An optional string to include in the new file names,
                such as a dataset identifier (e.g.,
                ``"precipitation_ecmwf"``). Default is ``""``.
            fmt: The date format in the original file names. Default is
                ``"%Y.%m.%d"``.
            freq: The temporal frequency of the data, which controls the
                date format in the new file names. One of ``"daily"``,
                ``"hourly"``, or any other value for minute-level.
                Default is ``"daily"``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("The directory you have entered does not exist")

        files = os.listdir(path)
        # get only the tif files
        files = [i for i in files if i.endswith(".tif")]

        # get the date
        dates_str = [files[i].split("_")[-1][:-4] for i in range(len(files))]
        dates = [dt.datetime.strptime(dates_str[i], fmt) for i in range(len(files))]

        if freq == "daily":
            new_date_str = [
                str(i.year) + "_" + str(i.month) + "_" + str(i.day) for i in dates
            ]
        elif freq == "hourly":
            new_date_str = [
                str(i.year) + "_" + str(i.month) + "_" + str(i.day) + "_" + str(i.hour)
                for i in dates
            ]
        else:
            new_date_str = [
                f"{i.year}-{i.month}-{i.day}-{i.hour}-{i.minute}" for i in dates
            ]

        df = pd.DataFrame()
        df["files"] = files
        df["DateStr"] = new_date_str
        df["dates"] = dates
        df.sort_values("dates", inplace=True)
        df.reset_index(inplace=True)
        df["order"] = [i for i in range(len(files))]

        df["new_names"] = [
            f"{df.loc[i, 'order']}_{prefix}_{df.loc[i, 'DateStr']}.tif"
            for i in range(len(files))
        ]
        # rename the files
        for i in range(len(files)):
            os.rename(
                f"{path}/{df.loc[i, 'files']}", f"{path}/{df.loc[i, 'new_names']}"
            )

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
        data_dir = os.getenv("HAPI_DATA_DIR")
        if data_dir is None:
            raise ValueError("HAPI_DATA_DIR environment variable is not set")
        else:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                raise FileNotFoundError(f"{data_dir} does not exist")
        return data_dir
