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
import math
import os
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statista.descriptors as metrics
from cleopatra.array_glyph import ArrayGlyph
from loguru import logger
from osgeo import gdal
from pyramids.dataset import Dataset
from pyramids.multidataset import MultiDataset as Datacube

from Hapi.dem import DEM

if TYPE_CHECKING:
    import matplotlib.animation

    from Hapi.rrm.base_model import BaseConceptualModel

STATE_VARIABLES = ["SP", "SM", "UZ", "LZ", "WC"]


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
        self.GaugesTable: gpd.GeoDataFrame | pd.DataFrame | None = None
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
        self.FlowDirArr: np.ndarray | None = None
        self.FDT: dict | None = None
        self.FPLArr: np.ndarray | None = None
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
            FileNotFoundError: If the path does not exist or the
                folder is empty.
            TypeError: If the resulting precipitation array is not a
                numpy ndarray.
        """
        if self.Prec is None:
            # data type
            assert isinstance(path, str), "path input should be string type"
            # check whether the path exists or not
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} you have provided does not exist")
            # check whether the folder has the rasters or not
            if not len(os.listdir(path)) > 0:
                raise FileNotFoundError(f"{path} folder you have provided is empty")
            # read data
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
            cube.open_multi_dataset()
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
            AssertionError: If the path does not exist or the
                resulting array is not a numpy ndarray.
            Exception: If the folder is empty.
        """
        if self.Temp is None:
            # data type
            assert isinstance(path, str), "path input should be string type"
            # check whether the path exists or not
            assert os.path.exists(path), path + " you have provided does not exist"
            # check whether the folder has the rasters or not
            if not len(os.listdir(path)) > 0:
                raise Exception(f"The folder you have provided is empty: {path}")
            # read data
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
            cube.open_multi_dataset()
            self.Temp = np.moveaxis(cube.values, 0, -1)
            assert isinstance(
                self.Temp, np.ndarray
            ), "array should be of type numpy array"

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
            AssertionError: If the path does not exist or the
                resulting array is not a numpy ndarray.
            Exception: If the folder is empty.
        """
        if self.ET is None:
            # data type
            assert isinstance(path, str), "path input should be string type"
            # check whether the path exists or not
            assert os.path.exists(path), path + " you have provided does not exist"
            # check whether the folder has the rasters or not
            if not len(os.listdir(path)) > 0:
                raise Exception(f"The folder you have provided is empty: {path}")
            # read data
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
            cube.open_multi_dataset()
            self.ET = np.moveaxis(cube.values, 0, -1)
            assert isinstance(
                self.ET, np.ndarray
            ), "array should be of type numpy array"
            logger.debug("Potential Evapotranspiration data are read successfully")

    def read_flow_acc(self, path: str):
        """Read flow accumulation raster and compute cell properties.

        Reads the flow accumulation raster, extracts the number of rows,
        columns, NoDataValue, number of domain cells, outlet location,
        cell size, and pixel area.

        Args:
            path (str): Path to the flow accumulation raster file
                (must include the raster name and .tif extension).

        Raises:
            TypeError: If `path` is not a string.
            AssertionError: If the path does not exist or does not
                end with ".tif".
        """
        # data type
        if not isinstance(path, str):
            raise TypeError("path input should be string type")
        # check whether the path exists or not
        assert os.path.exists(path), path + " you have provided does not exist"
        # check the extension of the accumulation file
        assert path.endswith(
            ".tif"
        ), "please add the extension at the end of the Flow accumulation raster path input"
        # check whether the path exists or not
        assert os.path.exists(path), path + " you have provided does not exist"

        flow_acc = gdal.Open(path)
        if flow_acc is None:
            raise FileNotFoundError(f"GDAL could not open: {path}")
        [self.rows, self.cols] = flow_acc.ReadAsArray().shape
        # check flow accumulation input raster
        self.NoDataValue = flow_acc.GetRasterBand(1).GetNoDataValue()
        self.FlowAccArr = flow_acc.ReadAsArray()

        # check if the flow acc array is integer convert it to float
        if self.FlowAccArr.dtype == "int":
            self.FlowAccArr = self.FlowAccArr.astype(float)

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.FlowAccArr[i, j], self.NoDataValue, rel_tol=0.001):
                    self.FlowAccArr[i, j] = np.nan

        self.no_elem = int(np.size(self.FlowAccArr[:, :]) - np.count_nonzero(
            (self.FlowAccArr[np.isnan(self.FlowAccArr)]))
        )
        self.acc_val = [
            int(self.FlowAccArr[i, j])
            for i in range(self.rows)
            for j in range(self.cols)
            if not np.isnan(self.FlowAccArr[i, j])
        ]
        self.acc_val = list(set(self.acc_val))
        self.acc_val.sort()
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

        # calculate area covered by cells
        geo_trans = (
            flow_acc.GetGeoTransform()
        )  # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        dx = np.abs(geo_trans[1]) / 1000.0  # dx in Km
        dy = np.abs(geo_trans[-1]) / 1000.0  # dy in Km
        self.CellSize = dx * 1000

        # area of the cell
        self.px_area = dx * dy
        # no_cells=np.size(raster[:,:])-np.count_nonzero(raster[raster==no_val])
        self.px_tot_area = self.no_elem * self.px_area  # total area of pixels

        logger.debug("Flow Accmulation input is read successfully")

    def read_flow_dir(self, path: str):
        """Read the flow direction raster and build the flow direction table.

        Args:
            path (str): Path to the flow direction raster file (must
                include the raster name and .tif extension).

        Raises:
            AssertionError: If `path` is not a string or does not
                exist.
            ValueError: If the file does not have a ".tif" extension.
            AssertionError: If the raster contains values other than
                1, 2, 4, 8, 16, 32, 64, 128.
        """
        # data type
        assert isinstance(path, str), "path input should be string type"
        # check whether the path exists or not
        assert os.path.exists(path), path + " you have provided does not exist"
        # check the extension of the accumulation file
        if not (path[-4:] == ".tif"):
            raise ValueError(
                "please add the extension at the end of the Flow accumulation raster path input"
            )
        # check whether the path exists or not
        assert os.path.exists(path), path + " you have provided does not exist"
        flow_dir = gdal.Open(path)
        if flow_dir is None:
            raise FileNotFoundError(f"GDAL could not open: {path}")

        [rows, cols] = flow_dir.ReadAsArray().shape
        self.FlowDirArr = flow_dir.ReadAsArray().astype(float)
        # check flow direction input raster
        fd_noval = flow_dir.GetRasterBand(1).GetNoDataValue()

        for i in range(rows):
            for j in range(cols):
                if math.isclose(self.FlowDirArr[i, j], fd_noval, rel_tol=0.001):
                    self.FlowDirArr[i, j] = np.nan

        fd_val = [
            int(self.FlowDirArr[i, j])
            for i in range(rows)
            for j in range(cols)
            if not np.isnan(self.FlowDirArr[i, j])
        ]
        fd_val = list(set(fd_val))
        fd_should = [1, 2, 4, 8, 16, 32, 64, 128]
        assert all(
            fd_val[i] in fd_should for i in range(len(fd_val))
        ), "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "

        # create the flow direction table
        dem = DEM(flow_dir)
        self.FDT = dem.flow_direction_table()
        logger.debug("Flow Direction input is read successfully")

    def read_flow_path_length(self, path: str):
        """Read the flow path length raster.

        Reads the flow path length raster and extracts rows, columns,
        NoDataValue, and the number of domain cells.

        Args:
            path (str): Path to the flow path length raster file
                (must include the raster name and .tif extension).

        Raises:
            AssertionError: If `path` is not a string or does not
                exist.
            ValueError: If the file does not have a ".tif" extension.
        """
        # data type
        assert isinstance(path, str), "path input should be string type"
        # input values
        fpl_ext = path[-4:]
        if not (fpl_ext == ".tif"):
            raise ValueError(
                "please add the extension at the end of the Flow accumulation raster path input"
            )
        # check whether the path exists or not
        assert os.path.exists(path), path + " you have provided does not exist"

        fpl = gdal.Open(path)
        if fpl is None:
            raise FileNotFoundError(f"GDAL could not open: {path}")
        [self.rows, self.cols] = fpl.ReadAsArray().shape
        self.FPLArr = fpl.ReadAsArray()
        self.NoDataValue = fpl.GetRasterBand(1).GetNoDataValue()

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.FPLArr[i, j], self.NoDataValue, rel_tol=0.001):
                    self.FPLArr[i, j] = np.nan
        # check flow accumulation input raster
        self.no_elem = int(np.size(self.FPLArr[:, :]) - np.count_nonzero(
            (self.FPLArr[np.isnan(self.FPLArr)])
        ))

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
            ds = gdal.Open(fpath)
            if ds is None:
                raise FileNotFoundError(
                    f"GDAL could not open {name} file: {fpath}"
                )
            setattr(self, name, ds.ReadAsArray())

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
            # data type
            assert isinstance(path, str), "cpath input should be string type"
            # check whither the path exists or not
            assert os.path.exists(path), f"{path} you have provided does not exist"
            # check whither the folder has the rasters or not
            if not len(os.listdir(path)) > 0:
                raise Exception(f"The folder you have provided is empty: {path}")
            # parameters
            cube = Datacube.read_multiple_files(
                path, with_order=True, regex_string=r"\d+", date=False
            )
            cube.open_multi_dataset()
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
        lumped_model,
        catchment_area: float | int,
        initial_condition: list,
        q_init=None,
    ):
        """Read and set up a lumped conceptual model.

        Args:
            lumped_model: A class representing the lumped conceptual
                model (e.g., HBV).
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

        Args:
            path (str): Path to the gauge file (CSV or GeoJSON).
            flow_acc_file (str, optional): Path to the flow
                accumulation raster used to map gauge coordinates to
                array indices. Default is "".
            fmt (str, optional): Date format for start/end columns
                in the gauge table. Default is "%Y-%m-%d".
        """
        # read the gauge table
        if path.endswith(".geojson"):
            self.GaugesTable = gpd.read_file(path, driver="GeoJSON")
        else:
            self.GaugesTable = pd.read_csv(path)
        col_list = self.GaugesTable.columns.tolist()

        if "start" in col_list:
            for i in range(len(self.GaugesTable)):
                self.GaugesTable.loc[i, "start"] = dt.datetime.strptime(
                    self.GaugesTable.loc[i, "start"], fmt
                )
                self.GaugesTable.loc[i, "end"] = dt.datetime.strptime(
                    self.GaugesTable.loc[i, "end"], fmt
                )
        if flow_acc_file != "" and "cell_row" not in col_list:
            # if hasattr(self, 'flow_acc'):
            flow_acc = gdal.Open(flow_acc_file)
            if flow_acc is None:
                raise FileNotFoundError(
                    f"GDAL could not open: {flow_acc_file}"
                )
            # calculate the nearest cell to each station
            dataset = Dataset(flow_acc)
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
        assert len(upper_bound) == len(
            lower_bound
        ), "the length of UB should be the same as LB"
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
                        metrics.pearson_corre(q_obs, q_sim), 3
                    )
                    self.Metrics.loc["R2", gauge_id] = round(
                        metrics.R2(q_obs, q_sim), 3
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
        domain.

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
                ``ArrayGlyph.animate``. Common options include:
                TicksSpacing (int), Figsize (tuple),
                PlotNumbers (bool), NumSize (int), title (str),
                title_size (int), Backgroundcolorthreshold (float),
                textcolors (tuple), cbarlabel (str),
                cbarlabelsize (int), Cbarlength (float),
                Interval (int), cmap (str), Textloc (list),
                Gaugecolor (str), Gaugesize (int),
                ColorScale (int), orientation (str),
                rotation (float), Points (DataFrame).

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.

        Raises:
            ValueError: If `option` is not between 1 and 11.
        """
        start = dt.datetime.strptime(start, fmt)
        end = dt.datetime.strptime(end, fmt)

        start_i = np.where(self.Index == start)[0][0]
        end_i = np.where(self.Index == end)[0][0]

        if 1 > option > 11:
            raise ValueError("Plotting options are from 1 to 11")

        if option == 1:
            self.Qtot[self.FlowAccArr == self.NoDataValue, :] = np.nan
            arr = self.Qtot[:, :, start_i:end_i]
            title = "Total Discharge"
        elif option == 2:
            self.quz_routed[self.FlowAccArr == self.NoDataValue, :] = np.nan
            arr = self.quz_routed[:, :, start_i:end_i]
            title = "Surface Flow"
        elif option == 3:
            self.qlz_translated[self.FlowAccArr == self.NoDataValue, :] = np.nan
            arr = self.qlz_translated[:, :, start_i:end_i]
            title = "Ground Water Flow"
        elif option == 4:
            self.state_variables[self.FlowAccArr == self.NoDataValue, :, 0] = np.nan
            arr = self.state_variables[:, :, start_i:end_i, 0]
            title = "Snow Pack"
        elif option == 5:
            self.state_variables[self.FlowAccArr == self.NoDataValue, :, 1] = np.nan
            arr = self.state_variables[:, :, start_i:end_i, 1]
            title = "Soil Moisture"
        elif option == 6:
            self.state_variables[self.FlowAccArr == self.NoDataValue, :, 2] = np.nan
            arr = self.state_variables[:, :, start_i:end_i, 2]
            title = "Upper Zone"
        elif option == 7:
            self.state_variables[self.FlowAccArr == self.NoDataValue, :, 3] = np.nan
            arr = self.state_variables[:, :, start_i:end_i, 3]
            title = "Lower Zone"
        elif option == 8:
            self.state_variables[self.FlowAccArr == self.NoDataValue, :, 4] = np.nan
            arr = self.state_variables[:, :, start_i:end_i, 4]
            title = "Water Content"
        elif option == 9:
            self.Prec[self.FlowAccArr == self.NoDataValue, :] = np.nan
            arr = self.Prec[:, :, start_i:end_i]
            title = "Precipitation"
        elif option == 10:
            self.ET[self.FlowAccArr == self.NoDataValue, :] = np.nan
            arr = self.ET[:, :, start_i:end_i]
            title = "ET"
        elif option == 11:
            self.Temp[self.FlowAccArr == self.NoDataValue, :] = np.nan
            arr = self.Temp[:, :, start_i:end_i]
            title = "Temperature"
        else:
            raise ValueError("Plotting options are from 1 to 11")

        time = self.Index[start_i:end_i]

        if gauges:
            kwargs["Points"] = self.GaugesTable

        array = ArrayGlyph(arr)
        anim = array.animate(time, title=title, **kwargs)
        # anim = StaticGlyph.AnimateArray(Arr, Time, self.no_elem, Title=Title, **kwargs)

        self.anim = anim

        return anim

    # def save_animation(self, video_format="gif", path="", save_frames=20):
    #     """saveAnimation. saveAnimation.
    #
    #     Parameters
    #     ----------
    #     video_format : [str], optional
    #         possible formats ['mp4','mov', 'avi', 'gif']. The default is "gif".
    #     path : [str], optional
    #         path inclusinf the video format. The default is ''.
    #     save_frames : [integer], optional
    #         speed of the video. The default is 20.
    #
    #     Returns
    #     -------
    #     None.
    #     """
    #     Vis.SaveAnimation(
    #         self.anim, VideoFormat=video_format, Path=path, SaveFrames=save_frames
    #     )

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

            src = gdal.Open(flow_acc_path)
            if src is None:
                raise FileNotFoundError(
                    f"GDAL could not open: {flow_acc_path}"
                )

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

            cube = Datacube(Dataset(src), time_length=arr.shape[2])
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
        lumped_model,
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
