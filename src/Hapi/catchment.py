"""Catchment."""

__name__ = "catchment"

import datetime as dt
import inspect
import math
import os
from typing import Optional, Union

import geopandas as gpd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statista.descriptors as metrics
from cleopatra.array_glyph import ArrayGlyph
from loguru import logger
from osgeo import gdal
from pyramids.datacube import Datacube
from pyramids.dataset import Dataset

from Hapi.dem import DEM

STATE_VARIABLES = ["SP", "SM", "UZ", "LZ", "WC"]


class Catchment:
    """Catchment.

        The class includes methods to read the meteorological and Spatial inputs of the distributed hydrological model.
        Catchment class also reads the data of the gauges, it is a superclass that has the run subclass,
        so you need to build the catchment object and hand it as an input to the Run class to run the model.

    Methods
    -------
        1-readRainfall
        2-readTemperature
        3-readET
        4-readFlowAcc
        5-readFlowDir
        6-ReadFlowPathLength
        7-readParameters
        8-readLumpedModel
        9-readLumpedInputs
        10-readGaugeTable
        11-readDischargeGauges
        12-readParametersBounds
        13-extractDischarge
        14-plotHydrograph
        15-PlotDistributedQ
        16-saveResults
    """

    def __init__(
        self,
        name: str,
        start_data: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        spatial_resolution: Optional[str] = "Lumped",
        temporal_resolution: Optional[str] = "Daily",
        routing_method: Optional[str] = "Muskingum",
    ):
        """Catchment.

        Parameters
        ----------
        name : [str]
            Name of the Catchment.
        start_data : [str]
            starting date.
        end : [str]
            end date.
        fmt: [str], optional
            format of the given date. The default is "%Y-%m-%d".
        spatial_resolution: [str, optional]
            Lumped or 'Distributed'. The default is 'Lumped'.
        temporal_resolution : TYPE, optional
            "Hourly" or "Daily". The default is "Daily".

        Returns
        -------
        None.
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
        self.Parameters = None
        self.data = None
        self.Prec = None
        self.TS = None
        self.Temp = None
        self.ET = None
        self.ll_temp = None
        self.QGauges = None
        self.Snow = None
        self.Maxbas = None
        self.LumpedModel = None
        self.CatArea = None
        self.InitialCond = None
        self.q_init = None
        self.GaugesTable = None
        self.UB = None
        self.LB = None
        self.cols = None
        self.rows = None
        self.NoDataValue = None
        self.FlowAccArr = None
        self.no_elem = None
        self.acc_val = None
        self.Outlet = None
        self.CellSize = None
        self.px_area = None
        self.px_tot_area = None
        self.FlowDirArr = None
        self.FDT = None
        self.FPLArr = None
        (
            self.DEM,
            self.BankfullDepth,
            self.RiverWidth,
            self.RiverRoughness,
            self.FloodPlainRoughness,
        ) = (None, None, None, None, None)
        self.qout, self.Qtot = None, None
        self.quz_routed, self.qlz_translated, self.state_variables = None, None, None
        self.anim = None
        self.quz, self.qlz = None, None
        self.Qsim = None
        self.Metrics = None

    def read_rainfall(
        self,
        path: str,
        start: str = None,
        end: str = None,
        fmt: str = "%Y-%m-%d",
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str = None,
        extension: str = ".tif",
    ):
        r"""readRainfall.

        Parameters
        ----------
        path: [str]
            A path to the Folder contains precipitation rasters.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        regex_string: [str]
            a regex string that we can use to locate the date in the file names.Default is r"\d{4}.\d{
            2}.\d{2}".
            >>> fname = "MSWEP_YYYY.MM.DD.tif"
            >>> regex_string = r"\d{4}.\d{2}.\d{2}"
            - or
            >>> fname = "MSWEP_YYYY_M_D.tif"
            >>> regex_string = r"\d{4}_\d{1}_\d{1}"
            - if there is a number at the beginning of the name
            >>> fname = "1_MSWEP_YYYY_M_D.tif"
            >>> regex_string = r"\d+"
        date: [bool]. Default is True.
            True if the number in the file name is a date.
        file_name_data_fmt : [str]
            if the files names' have a date and you want to read them ordered .Default is None
            >>> "MSWEP_YYYY.MM.DD.tif"
            >>> file_name_data_fmt = "%Y.%m.%d"
        extension: [str]
            the extension of the files you want to read from the given path. Default is ".tif".


        Returns
        -------
        prec: [array attribute]
            array containing the spatial rainfall values
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
            cube.open_datacube()
            self.Prec = np.moveaxis(cube.values, 0, -1)
            self.TS = self.Prec.shape[2] + 1
            # no of time steps =length of time series +1
            if not isinstance(self.Prec, np.ndarray):
                raise TypeError("Prec should be of type numpy array")

            logger.debug("Rainfall data are read successfully")

    def read_temperature(
        self,
        path: str,
        ll_temp: Union[list, np.ndarray] = None,
        start: str = None,
        end: str = None,
        fmt: str = "%Y-%m-%d",
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str = None,
        extension: str = ".tif",
    ):
        r"""readTemperature.

        Parameters
        ----------
        path: [str]
            A path to the Folder contains temperature rasters.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        ll_temp: [list/ ndarray]
            long-term temperature
        regex_string: [str]
            a regex string that we can use to locate the date in the file names. Default is r"\d{4}.\d{
            2}.\d{2}".
            >>> fname = "MSWEP_YYYY.MM.DD.tif"
            >>> regex_string = r"\d{4}.\d{2}.\d{2}"
            - or
            >>> fname = "MSWEP_YYYY_M_D.tif"
            >>> regex_string = r"\d{4}_\d{1}_\d{1}"
            - if there is a number at the beginning of the name
            >>> fname = "1_MSWEP_YYYY_M_D.tif"
            >>> regex_string = r"\d+"
        date: [bool]
            True if the number in the file name is a date. Default is True.
        file_name_data_fmt : [str]
            if the files names' have a date, and you want to read them ordered .Default is None
            >>> "MSWEP_YYYY.MM.DD.tif"
            >>> file_name_data_fmt = "%Y.%m.%d"
        extension: [str]
            the extension of the files you want to read from the given path. Default is ".tif".

        Returns
        -------
        Temp: [array attribute]
            array containing the spatial temperature values
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
            cube.open_datacube()
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
        start: str = None,
        end: str = None,
        fmt: str = "%Y-%m-%d",
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str = None,
        extension: str = ".tif",
    ):
        r"""readET.

        Parameters
        ----------
        path : [String]
            path to the Folder contains Evapotranspiration rasters.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        regex_string: [str]
            a regex string that we can use to locate the date in the file names.Default is r"\d{4}.\d{
            2}.\d{2}".
            >>> fname = "MSWEP_YYYY.MM.DD.tif"
            >>> regex_string = r"\d{4}.\d{2}.\d{2}"
            - or
            >>> fname = "MSWEP_YYYY_M_D.tif"
            >>> regex_string = r"\d{4}_\d{1}_\d{1}"
            - if there is a number at the beginning of the name
            >>> fname = "1_MSWEP_YYYY_M_D.tif"
            >>> regex_string = r"\d+"
        date: [bool]
            True if the number in the file name is a date. Default is True.
        file_name_data_fmt : [str]
            if the files names' have a date and you want to read them ordered .Default is None
            >>> "MSWEP_YYYY.MM.DD.tif"
            >>> file_name_data_fmt = "%Y.%m.%d"
        extension: [str]
            the extension of the files you want to read from the given path. Default is ".tif".

        Returns
        -------
        ET: [array attribute]
            array containing the spatial Evapotranspiration values
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
            cube.open_datacube()
            self.ET = np.moveaxis(cube.values, 0, -1)
            assert isinstance(
                self.ET, np.ndarray
            ), "array should be of type numpy array"
            logger.debug("Potential Evapotranspiration data are read successfully")

    def read_flow_acc(self, path: str):
        """readFlowAcc.

        Parameters
        ----------
        path : [String]
            path to the Flow Accumulation raster of the catchment
            (it should include the raster name and extension).

        Returns
        -------
        flow_acc: [array attribute]
            array containing the spatial Evapotranspiration values
        rows: [integer]
            number of rows in the flow acc array
        cols:[integer]
            number of columns in the flow acc array
        NoDataValue:[numeric]
            the NoDataValue
        no_elem : [integer]
            number of cells in the domain
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

        self.no_elem = np.size(self.FlowAccArr[:, :]) - np.count_nonzero(
            (self.FlowAccArr[np.isnan(self.FlowAccArr)])
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
        """Read Flow Direction.

            reads the flow direction raster.

        Parameters
        ----------
        path : [str]
            path to the flow direction raster.

        Returns
        -------
        FlowDirArr: [array].
            array of the flow direction raster
        FDT: [dictionary]
            flow direction table
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
        """Read Flow path Length method.

            reads the flow path length.

        Parameters
        ----------
        path : [str]
            path to the file.

        Returns
        -------
        FPLArr : [array]
            flow path length array
        rows : [integer]
            number of rows in the flow acc array
        cols : [integer]
            number of columns in the flow acc array
        NoDataValue : [numeric]
            the NoDataValue
        no_elem : [integer]
            number of cells in the domain
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
        [self.rows, self.cols] = fpl.ReadAsArray().shape
        self.FPLArr = fpl.ReadAsArray()
        self.NoDataValue = fpl.GetRasterBand(1).GetNoDataValue()

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.FPLArr[i, j], self.NoDataValue, rel_tol=0.001):
                    self.FPLArr[i, j] = np.nan
        # check flow accumulation input raster
        self.no_elem = np.size(self.FPLArr[:, :]) - np.count_nonzero(
            (self.FPLArr[np.isnan(self.FPLArr)])
        )

        logger.debug("Flow path length input is read successfully")

    def read_river_geometry(
        self,
        dem_file: str,
        bankfull_depth_file: str,
        river_width_file: str,
        river_roughness_file: str,
        floodplain_roughness_file: str,
    ):
        """ReadRiverGeometry.

        Parameters
        ----------
        dem_file
        bankfull_depth_file
        river_width_file
        river_roughness_file
        floodplain_roughness_file

        Returns
        -------
        None
        """
        self.DEM = gdal.Open(dem_file).ReadAsArray()
        self.BankfullDepth = gdal.Open(bankfull_depth_file).ReadAsArray()
        self.RiverWidth = gdal.Open(river_width_file).ReadAsArray()
        self.RiverRoughness = gdal.Open(river_roughness_file).ReadAsArray()
        self.FloodPlainRoughness = gdal.Open(floodplain_roughness_file).ReadAsArray()

    def read_parameters(self, path: str, snow: bool = False, maxbas: bool = False):
        """read_parameters.

            read_parameters method reads the parameters' raster

        Parameters
        ----------
        path: [str]
            path to the folder where the raster exists.
        snow: [integer]
            False if you don't want to run the snow-related processes and 1 if there is snow.
            in the case of 1 (simulate snow processes), parameters related to snow simulation
            have to be provided. The default is 0.
        maxbas: [bool], optional
            True if the routing is Maxbas. The default is False.

        Returns
        -------
        Parameters: [array].
            3d array containing the parameters
        Snow: [integer]
            0/1
        Maxbas: [bool]
            True/False
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
            cube.open_datacube()
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
        catchment_area: Union[float, int],
        initial_condition: list,
        q_init=None,
    ):
        """readLumpedModel.

        Parameters
        ----------
        lumped_model : [module]
            HBV module.
        catchment_area : [numeric]
            Catchment area (km2).
        initial_condition : [list]
            list of the initial condition [SnowPack, SoilMoisture, Upper Zone,
                                         Lower Zone, Water Content].
        q_init: [numeric], optional
            initial discharge. The default is None.

        Returns
        -------
        LumpedModel: [module].
            the lumped conceptual model.
        q_init : [numeric]
            initial discharge.
        InitialCond : [list]
            initial conditions.
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

    def read_lumped_inputs(self, path: str, ll_temp: Union[list, np.ndarray] = None):
        """readLumpedInputs. readLumpedInputs method read the meteorological data of lumped mode.

        [precipitation, Evapotranspiration, temperature, long-term average temperature]

        Parameters
        ----------
        path : [string]
            Path to the input file, data has to be in the order of
            [date, precipitation, ET, Temp].
        ll_temp: [bool], optional
            average long-term temperature, if None it is calculated inside the
            code. The default is None.

        Returns
        -------
        data: [array].
            meteorological data.
        ll_temp: [array]
            average long-term temperature.
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
        """readGaugeTable. readGaugeTable reads the table where the data about the gauges are listed.

        [x coordinate, y coordinate, 'area ratio', 'weight'], the coordinates are
        mandatory to enter, to locate the gauges and be able to extract the
        discharge at the corresponding cells.

        Parameters
        ----------
        path: [str]
            Path to the gauge file.
        flow_acc_file: [str], optional
            Path to the Flow acc raster. The default is ''.
        fmt: [str]
            Default is "%Y-%m-%d"

        Returns
        -------
        hm_gauges: [dataframe]
            the table of the gauges.
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
        start_date: str = "",
        end_date: str = "",
        readfrom: str = "",
    ):
        """read_discharge_gauges.

        read_discharge_gauges method read the gauge discharge data, discharge of each gauge has to be stored separately
        in a file, and the name of the file has to be stored in the Gauges table you entered using the method
        "readGaugeTable" under the column "id", the file should contain the date at the first column.

        Parameters
        ----------
        path: [str]
            path to the gauge discharge data.
        delimiter: [str], optional
            the delimiter between the date and the discharge column. The default is ",".
        column: [str], optional
            the name of the column where you stored the files' names. The default is 'id'.
        fmt: [str], optional
            date format. The default is "%Y-%m-%d".
        split: bool, optional
            True if you want to split the data between two dates. The default is False.
        start_date: [str], optional
            start date. The default is ''.
        end_date: [str], optional
            end date. The default is ''.
        readfrom: [str]
            Default is "".

        Returns
        -------
        GaugesTable: [dataframe].
            dataframe containing the discharge data
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
        upper_bound: Union[list, np.ndarray],
        lower_bound: Union[list, np.ndarray],
        snow: bool = False,
        maxbas: bool = False,
    ):
        """readParametersBounds. readParametersBounds method reads the lower and upper boundaries for each parameter.

        Parameters
        ----------
        upper_bound : [list]
            upper bound.
        lower_bound : [list]
            lower bound.
        snow : [integer]
            0 if you don't want to run the snow-related processes and 1 if there is snow.
            in case of 1 (simulate snow processes), parameters related to snow simulation
            have to be provided. The default is 0.
        maxbas: [bool]
            True if the parameters have maxbas.

        Returns
        -------
        UB: [list]
            upper bound.
        LB: [list]
            lower bound.
        Snow: [integer]
            Snow
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
        """extractDischarge.

        extractDischarge method extracts and sums the discharge from the Quz_routed and Qlz_translated arrays at the
        location of the gauges.

        Parameters
        ----------
        calculate_metrics : bool, optional
            True if you want to calculate the performance metrics. The default is True.
        frame_work_1: [bool], optional
            True if the routing function is Maxbas. The default is False.
        factor : [list/None]
            list of factor if you want to multiply the simulated discharge by
            a factor you have to provide a list of the factor (as many factors
            as the number of gauges). The default is False.
        only_outlet: [bool]
            True to extract the discharge only at the outlet.

        Returns
        -------
        q_sim : [dataframe]
            dataframe containing the discharge time series of the cells where
            the gauges are located.
        Metrics : [dataframe]
            data frame containing the following metrics ['RMSE', 'NSE', 'NSEhf', 'KGE', 'WB','Pearson-CC',
            'R2'] calculated between the simulated hydrographs and the gauge data.
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
        start_date: str,
        end_date: str,
        gauge: int,
        hapi_color: Union[tuple, str] = "#004c99",
        gauge_color: Union[tuple, str] = "#DC143C",
        line_width: int = 3,
        hapi_order: int = 1,
        gauge_order: int = 0,
        label_font_size: int = 10,
        x_major_fmt: str = "%Y-%m-%d",
        n_ticks: int = 5,
        title: str = "",
        x_axis_fmt: str = "%d\n%m",
        label: str = "",
        fmt: str = "%Y-%m-%d",
    ):
        r"""Plot Hydrograph.

            plot the simulated and gauge hydrograph.

        Parameters
        ----------
        start_date: [str]
            starting date.
        end_date: [str]
            end date.
        gauge: [integer]
            order if the gauge in the GaugeTable.
        hapi_color: [str], optional
            color of the Simulated hydrograph. The default is "#004c99".
        gauge_color: [str], optional
            color of the gauge. The default is "#DC143C".
        line_width: [numeric], optional
            line width. The default is 3.
        hapi_order: [integer], optional
            the order of the simulated hydrograph to control which hydrograph
            is in the front. The default is 1.
        gauge_order: TYPE, optional
            the order of the simulated hydrograph to control which hydrograph
            is in the front. The default is 0.
        label_font_size: numeric, optional
            label size. The default is 10.
        x_major_fmt: [str], optional
            format of the x-axis labels. The default is '%Y-%m-%d'.
        n_ticks: [integer], optional
            number of x-axis ticks. The default is 5.
        title: [str]
            Default is "".
        x_axis_fmt: [str]
            Default is "%d\n%m".
        label: [str]
            Default is "".
        fmt:[str]
            Default is "%Y-%m-%d".

        Returns
        -------
        fig: TYPE
            DESCRIPTION.
        ax: [matplotlib axes]
            you can control the figure from the axes.
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
        start: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        option: int = 1,
        gauges: bool = False,
        **kwargs,
    ):
        """plotDistributedResults.

        plotDistributedResults animate the time series of the meteorological inputs and the result calculated by the
        model like the total discharge, upper zone, and lower zone discharge and the state variables.

        Parameters
        ----------
        start: [str]
            starting date
        end: [str]
            end date
        fmt: [str]
            format of the given date. The default is "%Y-%m-%d"
        option : [str]
            1- Total discharge, 2-Upper zone discharge, 3-ground water,
            4-Snowpack state variable, 5-Soil moisture, 6-Upper zone,
            7-Lower zone, 8-Water content, 9-Precipitation input. 10-ET,
            11-Temperature. The default is 1.
        gauges: [str]
            gauge name. The default is False
        **kwargs :
            possible keyword args
            TicksSpacing: [integer], optional
                Spacing in the color bar ticks. The default is 2.
            Figsize: [tuple], optional
                figure size. The default is (8,8).
            PlotNumbers: [bool], optional
                True to plot the values on top of each cell. The default is True.
            NumSize: integer, optional
                size of the numbers plotted on top of each cell. The default is 8.
            title: [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size: [integer], optional
                title size. The default is 15.
            Backgroundcolorthreshold: [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.
            textcolors: TYPE, optional
                Two colors to be used to plot the values i top of each cell. The default is ("white","black").
            cbarlabel: str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            cbarlabelsize: integer, optional
                size of the color bar label. The default is 12.
            Cbarlength: [float], optional
                ratio to control the height of the colorbar. The default is 0.75.
            Interval: [integer], optional
                number to control the speed of the animation. The default is 200.
            cmap: [str], optional
                color style. The default is 'coolwarm_r'.
            Textloc: [list], optional
                location of the date text. The default is [0.1,0.2].
            Gaugecolor: [str], optional
                color of the points. The default is 'red'.
            Gaugesize: [integer], optional
                size of the points. The default is 100.
            ColorScale: integer, optional
                there are 5 options to change the scale of the colors. The default is 1.
                1- ColorScale 1 is the normal scale
                2- ColorScale 2 is the power scale
                3- ColorScale 3 is the SymLogNorm scale
                4- ColorScale 4 is the PowerNorm scale
                5- ColorScale 5 is the BoundaryNorm scale
                ------------------------------------------------------------------
                gamma: [float], optional
                    value needed for option 2 . The default is 1./2..
                linthresh: [float], optional
                    value needed for option 3. The default is 0.0001.
                linscale: [float], optional
                    value needed for option 3. The default is 0.001.
                midpoint: [float], optional
                    value needed for option 5. The default is 0.
                ------------------------------------------------------------------
            orientation: [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            rotation: [number], optional
                rotation of the colorbar label. The default is -90.
            **kwargs: [dict]
                keys:
                    Points : [dataframe].
                        dataframe contains two columns 'cell_row', and cell_col to
                        plot the point at this location

        Returns
        -------
        animation.FuncAnimation.
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
        start: str = "",
        end: str = "",
        path: str = "",
        prefix: str = "",
        fmt: str = "%Y-%m-%d",
    ):
        """save_results. saveResults save the results into rasters.

        Parameters
        ----------
        flow_acc_path : [path]
            Path to Flow acc raster.
        result: [integer], optional
            1 for the total discharge, 2 for the upper zone discharge, 3 for
            the lower zone discharge, 4 for the snow pack, 5 for the soil
            moisture, 6 upper zone, 7 for the lower zone, 8 for the water content.
            The default is 1.
        start: [str], optional
            start date. The default is ''.
        end: [str], optional
            end date. The default is ''.
        path: [str], optional
            path to the directory where you want to save the results. The default is ''.
        prefix: [str], optional
            prefix to add to the name of the result files. The default is ''.
        fmt: [str], optional
            format of the date. The default is "%Y-%m-%d".

        Returns
        -------
        None.
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
    """Lake.

        Lake class reads the meteorological inputs, and the module to simulate a lake as a lumped model, using a
        rating curve, the lake and the upstream sub-catchments are going to be considered as one lumped model than
        result in a discharge input to the lake, the discharge input is going to change the volume of the water in
        the lake, and from the volume-outflow curve the outflow can be obtained.

    Methods
    -------
        1- readMeteoData
        2- readParameters
        3- readLumpedModel
    """

    def __init__(
        self,
        start: str = "",
        end: str = "",
        fmt: str = "%Y-%m-%d",
        temporal_resolution: str = "Daily",
        split: bool = False,
    ):
        """Lake. Lake class for lake simulation.

        Parameters
        ----------
        start: [str], optional
            start date. The default is ''.
        end: [str], optional
            end date. The default is ''.
        fmt: [str], optional
            date format. The default is "%Y-%m-%d".
        temporal_resolution: [str], optional
            "Daily" ot "Hourly". The default is "Daily".
        split : bool, optional
            true if you want to split the date between two dates. The default is False.

        Returns
        -------
        None.
        """

        self.OutflowCell = None
        self.Snow = None
        self.Split = split
        self.start = dt.datetime.strptime(start, fmt)
        self.end = dt.datetime.strptime(end, fmt)

        if temporal_resolution.lower() == "daily":
            self.Index = pd.date_range(start, end, freq="D")
        elif temporal_resolution.lower() == "hourly":
            self.Index = pd.date_range(start, end, freq="h")
        else:
            assert False, "Error"

        self.MeteoData = None
        self.Parameters = None
        self.LumpedModel, self.CatArea, self.LakeArea, self.InitialCond = (
            None,
            None,
            None,
            None,
        )
        self.StageDischargeCurve = None

    def read_meteo_data(self, path: str, fmt: str):
        """readMeteoData. readMeteoData reads the meteorological data for the lake.

        [rainfall, ET, temperature]

        Parameters
        ----------
        path : [str]
            path to the meteo data file, containing the data in the following
            order [date, rainfall, ET, temperature].
        fmt : [str]
            date format.

        Returns
        -------
        MeteoData: [array].
            array containing the meteorological data
        """

        df = pd.read_csv(path, index_col=0)
        df.index = [dt.datetime.strptime(date, fmt) for date in df.index]

        if self.Split:
            df = df.loc[self.start : self.end, :]

        self.MeteoData = df.values  # lakeCalibArray = lakeCalibArray[:,0:-1]

        logger.debug("Lake Meteo data are read successfully")

    def read_parameters(self, path):
        """readParameters. readParameters method reads the lake parameters.

        Parameters
        ----------
        path : [str]
            path to the parameter file.

        Returns
        -------
        Parameters: [array].
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
        """readLumpedModel.

        readLumpedModel reads the lumped model module

        Parameters
        ----------
        lumped_model : [module]
            lumped conceptual model.
        catchment_area : [numeric]
            catchment area in mk2.
        lake_area : [numeric]
            area of the lake in km2.
        initial_condition : [list]
            initial condition [Snow Pack, Soil Moisture, Upper Zone, Lower Zone,
                               Water Content, Lake volume].
        outflow_cell : [list]
            indexes of the cell where the lake hydrograph is going to be added.
        stage_discharge_curve : [array]
            volume-outflow curve.
        snow : [integer]
            0 if you don't want to run the snow-related processes and 1 if there is snow.
            in case of 1 (simulate snow processes) parameters related to snow simulation
             have to be provided. The default is 0.

        Returns
        -------
        LumpedModel : [module]
            HBV module.
        CatArea : [numeric]
            Catchment area (km2).
        InitialCond : [list]
            list of the initial condition [SnowPack, SoilMoisture, Upper Zone, Lower Zone, Water Content].
        Snow : [integer]
            0/1
        StageDischargeCurve: [array]
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
