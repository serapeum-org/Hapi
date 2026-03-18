"""RUN.

RUN contains functions to connect the parameter spatial distribution
function with both components of the spatial representation of the hydrological
process (conceptual model & spatial routing) to calculate the predicted
runoff at known locations based on given performance function

Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""
import numpy as np
import pandas as pd
from loguru import logger

from Hapi.catchment import Catchment

# from Hapi.hm.saintvenant import SaintVenant
from Hapi.wrapper import Wrapper


class Run(Catchment):
    """Run the catchment model.

        Run sub-class validate the spatial data and hand it to the wrapper class, It is
        a sub-class from the catchment class, so you need to create the Catchment
        object first to run the model

    Methods
    -------
        1- RunHapi
        2- runHAPIwithLake
        3- runFW1
        4- RunFW1withLake
        5- runLumped
    """

    def __init__(self):
        self.Qsim = None
        pass

    def RunHapi(self):
        """Run Model.

        Returns
        -------
        state_variables: [numpy attribute]
            4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
        qlz: [numpy attribute]
            3D array of the lower zone discharge
        quz: [numpy attribute]
            3D array of the upper zone discharge
        qout: [numpy attribute]
            1D timeseries of discharge at the outlet of the catchment
            of unit m3/sec
        quz_routed: [numpy attribute]
            3D array of the upper zone discharge  accumulated and
            routed at each time step
        qlz_translated: [numpy attribute]
            3D array of the lower zone discharge translated at each time step
        """
        # input dimensions
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # run the model
        Wrapper.RRMModel(self)

        print("Model Run has finished")

    def RunFloodModel(self):
        """Run flood model.

        - This function runs the conceptual distributed hydrological model
        """
        # input dimensions
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        assert (
            np.shape(self.BankfullDepth)[0] == self.rows
            and np.shape(self.RiverWidth)[0] == self.rows
            and np.shape(self.RiverRoughness)[0] == self.rows
            and np.shape(self.FloodPlainRoughness)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.BankfullDepth)[1] == self.cols
            and np.shape(self.RiverWidth)[1] == self.cols
            and np.shape(self.RiverRoughness)[1] == self.cols
            and np.shape(self.FloodPlainRoughness)[1] == self.cols
        ), "all input data should have the same number of columns"

        # run the model
        Wrapper.RRMModel(self)
        print("RRM has finished")
        # SV = SaintVenant()
        # SV.KinematicRaster(self)
        # print("1D model Run has finished")

    def runHAPIwithLake(self, Lake):
        """Run model with lake.

            - This function runs the conceptual distributed hydrological model

        Returns
        -------
        st: [4D array]
            state variables.
        q_out: [1D array]
            calculated Discharge at the outlet of the catchment.
        q_uz: [3D array]
            Distributed discharge for each cell.
        """
        # input dimensions
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows and columns"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        assert (
            np.shape(Lake.MeteoData)[0] == np.shape(self.Prec)[2]
        ), "Lake meteorological data has to have the same length as the distributed raster data"
        assert (
            np.shape(Lake.MeteoData)[1] >= 3
        ), "Lake Meteo data has to have at least three columns of rain, ET, and Temp"

        # run the model
        Wrapper.RRMWithlake(self, Lake)

        print("Model Run has finished")

    def runFW1(self):
        """Run DistwithLake.

            This function runs the conceptual distributed hydrological model

        Returns
        -------
        st: [4D array]
            state variables
        q_out: [1D array]
            calculated Discharge at the outlet of the catchment
        q_uz: [3D array]
            Distributed discharge for each cell
        """
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # run the model
        Wrapper.FW1(self)

        print("Model Run has finished")

    def RunFW1withLake(self, Lake):
        """RunDistwithLake.

        this function runs the conceptual distributed hydrological model

        Parameters
        ----------
        1-Paths:
            1-prec_path:
                [String] path to the Folder contains precipitation rasters
            2-evap_path:
                [String] path to the Folder contains Evapotranspiration rasters
            3-temp_path:
                [String] path to the Folder contains Temperature rasters
            4-flow_acc_path:
                [String] path to the Flow Accumulation raster of the catchment (it should
                include the raster name and extension)
            5-flow_direction_path:
                [String] path to the Flow Direction raster of the catchment (it should
                include the raster name and extension)
        7-ParPath:
            [String] path to the Folder contains parameters rasters of the catchment
        8-p2:
            [List] list of unoptimized parameters
            p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
            p2[1] = catchment area in km2

        Returns
        -------
        st: [4D array]
            state variables
        q_out: [1D array]
            calculated Discharge at the outlet of the catchment
        q_uz: [3D array]
            Distributed discharge for each cell

        Example
        -------
        prec_path = prec_path="meteodata/4000/calib/prec"
        evap_path = evap_path="meteodata/4000/calib/evap"
        temp_path = temp_path="meteodata/4000/calib/temp"
        DemPath = "GIS/4000/dem4000.tif"
        flow_acc_path = "GIS/4000/acc4000.tif"
        flow_direction_path = "GIS/4000/fd4000.tif"
        ParPath = "meteodata/4000/parameters"
        p2=[1, 227.31]
        st, q_out, q_uz_routed = RunModel(prec_path,evap_path,temp_path,DemPath,
                                          flow_acc_path,flow_direction_path,ParPath,p2)
        """
        # input data validation

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
            and np.shape(self.Parameters)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
            and np.shape(self.Parameters)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        assert (
            np.shape(Lake.MeteoData)[0] == np.shape(self.Prec)[2]
        ), "Lake meteorological data has to have the same length as the distributed raster data"
        assert (
            np.shape(Lake.MeteoData)[1] >= 3
        ), "Lake Meteo data has to have at least three columns rain, ET, and Temp"

        # run the model
        Wrapper.FW1Withlake(self, Lake)

    def runLumped(
        self,
        Route: int = 0,
        RoutingFn=None,
    ):
        """Run lumped model.

            - This function runs lumped conceptual model

        Parameters
        ----------
        Route: [0 or 1]
            to decide wether t route the generated discharge hydrograph or not
        RoutingFn: [function]
            function to route the dischrge hydrograph.

        Parameters that should be defined before calling the function
            ConceptualModel: [function]
                conceptual model and it should contain a function called simulate
            data: [numpy array]
                meteorological data as array with the first column as precipitation
                second as evapotranspiration, third as temperature and forth column as
                long term average temperature
            parameters: [numpy array]
                conceptual model parameters as array
            p2: [List]
                list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2
            init_st: [list]
                initial state variables values [sp, sm, uz, lz, wc].

        Returns
        -------
        st: [numpy array]
            3d array of the 5 state variable data for each cell
        q_lz: [numpy array]
            1d array of the calculated discharge.

        Examples
        --------
        >>> p2 = [24, 1530]
        >>> #[sp,sm,uz,lz,wc]
        >>> init_st = [0,5,5,5,0]
        >>> snow = 0
        """
        if RoutingFn is None:
            RoutingFn = []
        if self.temporal_resolution.lower() == "daily":
            ind = pd.date_range(self.start, self.end, freq="D")
        else:
            ind = pd.date_range(self.startdate, self.enddate, freq="h")

        Qsim = pd.DataFrame(index=ind)

        Wrapper.Lumped(self, Route, RoutingFn)
        Qsim["q"] = self.Qsim
        self.Qsim = Qsim[:]
        logger.info("Lumped model run has finished successfully")


if __name__ == "__main__":
    print("Run")
