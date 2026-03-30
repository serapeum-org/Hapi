"""Hapi.rrm.parameters module.

This module contains functions responsible for distributing parameters
spatially (totally distributed, totally distributed with some parameters
lumped, all parameters lumped, hydrologic response units) and saving
generated parameters into rasters.
"""
from __future__ import annotations

import datetime as dt
import math
import os

import numpy as np
from osgeo import gdal
from pyramids.dataset import Dataset


class Parameters:
    """Parameter distribution class for hydrological model calibration.

    The Parameters class distributes values from a parameter vector during
    the calibration process into a 3D array, handling lumped parameters
    and hydrologic response units (HRUs).
    """

    def __init__(
        self,
        raster,
        no_parameters: int,
        no_lumped_par: int = 0,
        lumped_par_pos: list[int] | None = None,
        lake: bool = False,
        snow: bool = False,
        hru: bool = False,
        function: int = 1,
        k_upper_bound: int = 1,
        k_lower_bound: int = 50,
        muskingum: bool = False,
    ):
        """Initialize the Parameters class.

        To initiate the Parameters class, you have to provide the Flow Acc
        raster.

        Args:
            raster: A gdal.Dataset raster to get the spatial information
                of the catchment (DEM, flow accumulation or flow direction
                raster).
            no_parameters: Number of parameters in the HBV model.
            no_lumped_par: Number of lumped parameters. You have to enter
                the value of the lumped parameter at the end of the list.
                Defaults to 0 (no lumped parameters).
            lumped_par_pos: List of the order or position of lumped
                parameters among all the parameters of the lumped model
                (order starts from 0 to the length of the model
                parameters). Defaults to None (empty). The following
                order of parameters is used for the lumped HBV model:
                [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta,
                e_corr, etf, lp, c_flux, k, k1, alpha, perc, pcorr,
                Kmuskingum, Xmuskingum].
            lake: True if there is a lake, False otherwise.
                Defaults to False.
            snow: True to run the snow-related processes, False otherwise.
                When True, parameters related to snow simulation have to
                be provided. Defaults to False.
            hru: True if the parameters will consider using HRUs.
                Defaults to False.
            function: Function to use for distributing parameters.
                Defaults to 1.
            k_upper_bound: Upper bound of K value (traveling time in
                muskingum routing method). Defaults to 1 hour.
            k_lower_bound: Lower bound of K value (traveling time in
                muskingum routing method). Defaults to 50.
            muskingum: True if the routing function is muskingum.
                Defaults to False.

        Raises:
            AssertionError: If `raster` is not a gdal.Dataset, if
                `no_parameters` is not an integer, if `no_lumped_par` is
                not an integer, or if the length of `lumped_par_pos` does
                not match `no_lumped_par`.
            ValueError: If `lumped_par_pos` is not a list when
                `no_lumped_par` >= 1.
        """
        if lumped_par_pos is None:
            lumped_par_pos = []

        assert isinstance(
            raster, gdal.Dataset
        ), "raster should be read using gdal (gdal dataset please read it using gdal library) "
        assert isinstance(no_parameters, int), " no_parameters should be integer number"
        assert isinstance(
            no_lumped_par, int
        ), "no of lumped parameters should be integer"

        if no_lumped_par >= 1:
            if isinstance(lumped_par_pos, list):
                assert no_lumped_par == len(lumped_par_pos), (
                    f"you have to entered {no_lumped_par} no of lumped parameters but only {len(lumped_par_pos)} "
                    f"position "
                )
            else:  # if not int or list
                raise ValueError(
                    "you have one or more lumped parameters, so the position has to be entered as a list"
                )

        self.Lake = lake
        self.Snow = snow
        self.no_lumped_par = no_lumped_par
        self.lumped_par_pos = lumped_par_pos
        self.HRUs = hru
        self.Kub = k_upper_bound
        self.Klb = k_lower_bound
        self.Maskingum = muskingum
        # read the raster
        self.raster = raster
        self.raster_A = raster.ReadAsArray().astype(float)
        # get the shape of the raster
        self.rows = raster.RasterYSize
        self.cols = raster.RasterXSize
        # get the no_value of in the raster
        self.noval = raster.GetRasterBand(1).GetNoDataValue()

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.raster_A[i, j], self.noval, rel_tol=0.001):
                    self.raster_A[i, j] = np.nan

        # count the number of non-empty cells
        if self.HRUs:
            self.values = list(
                set(
                    [
                        int(self.raster_A[i, j])
                        for i in range(self.rows)
                        for j in range(self.cols)
                        if not np.isnan(self.raster_A[i, j])
                    ]
                )
            )
            self.no_elem = len(self.values)
        else:
            self.no_elem = np.size(self.raster_A[:, :]) - np.count_nonzero(  # type: ignore[assignment]
                (self.raster_A[np.isnan(self.raster_A)])
            )

        self.no_parameters = no_parameters

        # store the indexes of the non-empty cells
        self.celli = []
        self.cellj = []
        for i in range(self.rows):
            for j in range(self.cols):
                if not np.isnan(self.raster_A[i, j]):
                    self.celli.append(i)
                    self.cellj.append(j)

        # create an empty 3D array [[raster dimension], no_parameters]
        self.Par3d = np.zeros([self.rows, self.cols, self.no_parameters]) * np.nan

        if no_lumped_par >= 1:
            # parameters in an array
            # remove a place for the lumped parameter (k1) lower zone coefficient
            self.no_parameters = self.no_parameters - no_lumped_par

        # all parameters lumped and distributed
        self.totnumberpar = self.no_parameters * self.no_elem + no_lumped_par
        # parameters in array
        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.zeros(
            shape=(self.no_parameters, self.no_elem), dtype=np.float32
        )

        if function == 1:
            self.Function = self.par3d_lumped
        elif function == 2:
            self.Function = self.par3d
        elif function == 3:
            self.Function = self.par2d_lumped_k1_lake  # type: ignore[assignment]
        elif function == 4:
            self.Function = self.hydrologic_response_units
        # to overwrite any choice user choose if the is HRUs
        if self.HRUs == 1:
            self.Function = self.hydrologic_response_units

        self.parameters_number()

        pass

    def par3d(self, par_g):  # , kub=1,klb=0.5, Maskingum=True
        """Distribute parameters horizontally across grid cells.

        Takes a list of parameters (saved as one column or generated as a
        1D list from an optimization algorithm) and distributes them
        horizontally on the number of cells given by a raster.

        Args:
            par_g: 1D list or numpy array of parameters. For totally
                distributed parameters, the length should be
                ``no_elem * no_parameters``. For lumped parameters, the
                lumped parameter values should be appended at the end.

        Raises:
            AssertionError: If the length of `par_g` does not match the
                expected number of parameters based on the number of
                elements and lumped parameters.
        """
        # input data validation
        # data type
        # assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
        # assert isinstance(kub,numbers.Number) , " kub should be a number"
        # assert isinstance(klb,numbers.Number) , " klb should be a number"

        # input values
        if self.no_lumped_par > 0:
            par_no = (self.no_elem * self.no_parameters) + self.no_lumped_par

            assert len(par_g) == par_no, (
                f"As there is {self.no_lumped_par} lumped parameters, length of input parameters should be "
                f"{self.no_elem}"
                + f"*({self.no_parameters + self.no_lumped_par} - {self.no_lumped_par}) + {self.no_lumped_par} = "
                + f"{self.no_elem * (self.no_parameters - self.no_lumped_par) + self.no_lumped_par} not {len(par_g)}"
                + " probably you have to add the value of the lumped parameter at the end of the list"
            )
        else:
            # if there are no lumped parameters
            par_no = self.no_elem * self.no_parameters
            assert len(par_g) == par_no, (
                f"As there is no lumped parameters length of input parameters should be {self.no_elem} * "
                + f"{self.no_parameters} = {self.no_elem * self.no_parameters}"
            )

        # parameters in array
        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.ones((self.no_parameters, self.no_elem))  # type: ignore[assignment]
        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g[
                i * self.no_parameters : (i * self.no_parameters) + self.no_parameters
            ]

        # lumped parameters
        if self.no_lumped_par > 0:
            for i in range(self.no_lumped_par):
                # create a list with the value of the lumped parameter(k1)
                # (stored at the end of the list of the parameters)
                pk1 = (
                    np.ones((1, self.no_elem))
                    * par_g[(self.no_parameters * np.shape(self.Par2d)[1]) + i]
                )
                # put the list of parameter k1 at the 6th row.
                self.Par2d = np.vstack(
                    [
                        self.Par2d[: self.lumped_par_pos[i], :],
                        pk1,
                        self.Par2d[self.lumped_par_pos[i] :, :],
                    ]
                )

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i], self.cellj[i], :] = self.Par2d[:, i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value

        # if Maskingum:
        #     for i in range(self.no_elem):
        #         self.Par3d[self.celli[i],self.cellj[i],-2]=
        #         Parameters.calculateK(
        #               self.Par3d[self.celli[i], self.cellj[i],-1], self.Par3d[self.celli[i], self.cellj[i],-2], kub,
        #               klb
        #              )

    def par3d_lumped(self, par_g):  # , kub=1, klb=0.5, Maskingum = True
        r"""Distribute lumped parameters horizontally across grid cells.

        Takes a list of parameters (saved as one column or generated as a
        1D list from an optimization algorithm) and distributes them
        horizontally on the number of cells given by a raster, where all
        parameters are lumped (same value for every cell).

        Args:
            par_g: 1D list or numpy array of lumped parameters.
                The length should equal ``no_parameters``.

        Raises:
            ValueError: If `par_g` is not a numpy ndarray or a list.
        """
        # input data validation
        # data type
        if not (isinstance(par_g, np.ndarray) or isinstance(par_g, list)):
            raise ValueError("par_g should be of type 1d array or list")
        # assert isinstance(kub,numbers.Number) , " kub should be a number"
        # assert isinstance(klb,numbers.Number) , " klb should be a number"

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i], self.cellj[i], :] = self.Par2d[:, i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        # if Maskingum == True:
        #     for i in range(self.no_elem):
        #         self.Par3d[self.celli[i],self.cellj[i],-2] = Parameters.calculateK(
        #         self.Par3d[self.celli[i],self.cellj[i],-1], self.Par3d[self.celli[i],self.cellj[i],-2],kub,klb)

    @staticmethod
    def calculate_k(x, position, upper_bound, lower_bound):
        """Calculate K parameter for Muskingum routing.

        Takes the value of x parameter and generates 100 random values of
        the K parameter between the upper and lower constraints, then
        returns the value corresponding to the given position.

        Args:
            x: Weighting coefficient to determine the linearity of the
                water surface (one of the parameters of the Muskingum
                routing method).
            position: Random position between upper and lower bounds of
                the K parameter.
            upper_bound: Upper bound for the K parameter.
            lower_bound: Lower bound for the K parameter.

        Returns:
            The K parameter value corresponding to the given position
            within the constrained range.
        """
        # k has to be smaller than this constraint
        constraint1 = 0.5 * 1 / (1 - x)
        # k has to be greater than this constraint
        constraint2 = 0.5 * 1 / x
        # if constraint is higher than UB take UB
        if constraint2 >= upper_bound:
            constraint2 = upper_bound
        # if constraint is lower than LB take UB
        if constraint1 <= lower_bound:
            constraint1 = lower_bound

        generated_k = np.linspace(constraint1, constraint2, 50)
        k = generated_k[int(round(position, 0))]
        return k

    def par2d_lumped_k1_lake(self, par_g, no_parameters_lake):  # ,kub,klb
        """Distribute parameters with a lumped K1 and lake parameters.

        Takes a list of parameters and distributes them horizontally on
        the number of cells given by a raster. All parameters are
        distributed except the lower zone coefficient (K1), which is
        lumped and appended at the end of the parameter list. Lake
        parameters are extracted from the end of the parameter list.

        Args:
            par_g: 1D list or numpy array of parameters. Each cell's
                distributed parameters are listed sequentially, followed
                by the lumped K1 value, followed by lake parameters at
                the end. For example, with 14 cells and 11 distributed
                parameters: ``14 * 11 = 154 + 1 (K1) = 155``.
            no_parameters_lake: Number of lake parameters to extract
                from the end of `par_g`.
        """
        # parameters in array
        # remove a place for the lumped parameter (k1) lower zone coefficient
        no_parameters = self.no_parameters - 1

        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.ones((no_parameters, self.no_elem))  # type: ignore[assignment]

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g[
                i * no_parameters : (i * no_parameters) + no_parameters
            ]

        # create a list with the value of the lumped parameter(k1)
        # (stored at the end of the list of the parameters)
        pk1 = (
            np.ones((1, self.no_elem))
            * par_g[(np.shape(self.Par2d)[0] * np.shape(self.Par2d)[1])]
        )

        # put the list of parameter k1 at the 6 row
        self.Par2d = np.vstack([self.Par2d[:6, :], pk1, self.Par2d[6:, :]])

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i], self.cellj[i], :] = self.Par2d[:, i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        # for i in range(self.no_elem):
        #     self.Par3d[self.celli[i],self.cellj[i],-2] = Parameters.calculateK(
        #     self.Par3d[self.celli[i],self.cellj[i],-1],self.Par3d[self.celli[i],self.cellj[i],-2],kub,klb)

        # lake parameters
        self.lake_par = par_g[len(par_g) - no_parameters_lake :]
        # self.lake_par[-2] = Parameters.calculateK(self.lake_par[-1],self.lake_par[-2],kub,klb)

        # return self.Par3d, lake_par

    def hydrologic_response_units(self, par_g):  # ,kub=1,klb=0.5
        """Distribute parameters using Hydrologic Response Units (HRUs).

        Takes a list of parameters (saved as one column or generated as a
        1D list from an optimization algorithm) and distributes them
        horizontally on the number of cells given by a raster. The input
        raster should be a classified raster (by numbers) into classes to
        define the HRUs. Each HRU receives the same set of generated
        parameters.

        Args:
            par_g: 1D list or numpy array of parameters. For HRU without
                lumped parameters, the length should be
                ``no_elem * no_parameters``. For HRU with lumped
                parameters, the lumped parameter values should be
                appended at the end.

        Raises:
            ValueError: If `par_g` is not a numpy ndarray or a list, or
                if the length of `par_g` does not match the expected
                number of parameters.
            AssertionError: If there are lumped parameters and the length
                of `par_g` does not match the expected total.
        """
        # input data validation
        # data type
        if not (isinstance(par_g, np.ndarray) or isinstance(par_g, list)):
            raise ValueError("par_g should be of type 1d array or list")
        # assert isinstance(kub,numbers.Number) , " kub should be a number"
        # assert isinstance(klb,numbers.Number) , " klb should be a number"

        # input values
        if self.no_lumped_par > 0:
            par_no = (self.no_elem * self.no_parameters) + self.no_lumped_par
            assert len(par_g) == par_no, (
                f"As there is {self.no_lumped_par} lumped parameters, length of input parameters should be "
                f"{self.no_elem}*({self.no_parameters}-{self.no_lumped_par})+{self.no_lumped_par}="
                + str(
                    self.no_elem * (self.no_parameters - self.no_lumped_par)
                    + self.no_lumped_par
                )
                + f" not {len(par_g)} probably you have to add the value of the lumped parameter at the end of the list"
            )
        else:
            # if there is no lumped parameters
            if not len(par_g) == self.no_elem * self.no_parameters:
                raise ValueError(
                    f"As there is no lumped parameters length of input parameters should be {self.no_elem}*"
                    f"{self.no_parameters}={self.no_elem * self.no_parameters}"
                )

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        self.Par2d = np.zeros(
            shape=(self.no_parameters, self.no_elem), dtype=np.float64
        )
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g[
                i * self.no_parameters : (i * self.no_parameters) + self.no_parameters
            ]

        # lumped parameters
        if self.no_lumped_par > 0:
            for i in range(self.no_lumped_par):
                # create a list with the value of the lumped parameter(k1)
                # (stored at the end of the list of the parameters)
                pk1 = (
                    np.ones((1, self.no_elem))
                    * par_g[(self.no_parameters * np.shape(self.Par2d)[1]) + i]
                )
                # put the list of parameter k1 at the 6 row
                self.Par2d = np.vstack(
                    [
                        self.Par2d[: self.lumped_par_pos[i], :],
                        pk1,
                        self.Par2d[self.lumped_par_pos[i] :, :],
                    ]
                )

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        # for i in range(self.no_elem):
        #     self.Par2d[-2,i] = Parameters.calculateK(self.Par2d[-1,i],self.Par2d[-2,i],kub,klb)

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d each soil type will have the same
        # generated parameters
        for i in range(self.no_elem):
            self.Par3d[self.raster_A == self.values[i]] = self.Par2d[:, i]

    @staticmethod
    def hru_hand(dem, flow_direction, flow_path_length, river):
        """Calculate Height Above Nearest Drainage (HAND) for HRU classification.

        Calculates inputs for the HAND method for land use
        classification by tracing flow direction from each cell to the
        nearest river reach, then computing the elevation difference
        and the flow path distance.

        Args:
            dem: A gdal.Dataset of the DEM raster.
            flow_direction: A gdal.Dataset of the flow direction raster.
            flow_path_length: A gdal.Dataset of the flow path length
                raster.
            river: A gdal.Dataset of the river location raster, where
                cells with value 1 indicate river presence.

        Returns:
            A tuple of two numpy ndarrays:
                - hand: Height above nearest drainage for each cell.
                - dist_to_nearest_drain: Distance to nearest drainage
                  for each cell.

        Raises:
            ValueError: If the catchment boundaries contain anomalies
                (e.g., after cropping with a polygon).
        """
        # Use DEM raster information to run all loops
        dem_a = dem.ReadAsArray()
        no_val = np.float32(dem.GetRasterBand(1).GetNoDataValue())
        rows = dem.RasterYSize
        cols = dem.RasterXSize

        # get the indices of the flow direction path
        dem = dem(flow_direction)
        fd_index = dem.flowDirectionIndex()

        # read the river location raster
        river_a = river.ReadAsArray()

        # read the flow path length raster
        fpl_a = flow_path_length.ReadAsArray()

        # trace the flow direction to the nearest river reach and store the location
        # of that nearst reach
        nearest_network = np.ones((rows, cols, 2)) * np.nan
        try:
            for i in range(rows):
                for j in range(cols):
                    if dem_a[i, j] != no_val:
                        f = river_a[i, j]
                        old_row = i
                        old_cols = j

                        while f != 1:
                            # did not reached to the river yet then go to the next down stream cell
                            # get the down stream cell (furure position)
                            new_row = int(fd_index[old_row, old_cols, 0])
                            new_cols = int(fd_index[old_row, old_cols, 1])
                            # print(str(new_row)+","+str(new_cols))
                            # go to the downstream cell
                            f = river_a[new_row, new_cols]
                            # down stream cell becomes the current position (old position)
                            old_row = new_row
                            old_cols = new_cols
                            # at this moment old and new stored position are the same (current position)
                        # store the position in the array
                        nearest_network[i, j, 0] = new_row
                        nearest_network[i, j, 1] = new_cols

        except Exception as e:
            print(e)
            raise ValueError(
                "please check the boundaries of your catchment.  After cropping the catchment using a polygon, it "
                "creates anomalies at the boundary"
            )

        # calculate the elevation difference between the cell and the nearest drainage cell
        # or height above nearst drainage
        hand = np.ones((rows, cols)) * np.nan

        for i in range(rows):
            for j in range(cols):
                if dem_a[i, j] != no_val:
                    hand[i, j] = (
                        dem_a[i, j]
                        - dem_a[
                            int(nearest_network[i, j, 0]), int(nearest_network[i, j, 1])
                        ]
                    )

        # calculate the distance to the nearest drainage c  ell using flow path length or distance to nearest drainage
        dist_to_nearest_drain = np.ones((rows, cols)) * np.nan

        for i in range(rows):
            for j in range(cols):
                if dem_a[i, j] != no_val:
                    dist_to_nearest_drain[i, j] = (
                        fpl_a[i, j]
                        - fpl_a[
                            int(nearest_network[i, j, 0]), int(nearest_network[i, j, 1])
                        ]
                    )

        return hand, dist_to_nearest_drain

    def parameters_number(self):
        """Calculate the total number of parameters for the optimization.

        Calculates the number of parameters that the optimization
        algorithm will search for. Use this only in case of totally
        distributed catchment parameters. In case of lumped parameters,
        the number of parameters is the same as the number of parameters
        of the conceptual model.

        The result is stored in the ``ParametersNO`` attribute.

        Note:
            The Parameters object should have the following attributes
            before calling this method: ``raster``, ``no_parameters``,
            ``no_lumped_par``, and ``HRUs``.
        """
        if not self.HRUs:
            if self.no_lumped_par > 0:
                # self.ParametersNO = (self.no_elem *( self.no_parameters - self.no_lumped_par)) + self.no_lumped_par
                self.ParametersNO = (
                    self.no_elem * self.no_parameters
                ) + self.no_lumped_par
            else:
                # if there is no lumped parameters
                self.ParametersNO = self.no_elem * self.no_parameters
        else:
            if self.no_lumped_par > 0:
                # self.ParametersNO = (self.no_elem * (self.no_parameters - self.no_lumped_par)) + self.no_lumped_par
                self.ParametersNO = (
                    self.no_elem * self.no_parameters
                ) + self.no_lumped_par
            else:
                # if there is no lumped parameters
                self.ParametersNO = self.no_elem * self.no_parameters

    def save_parameters(self, path):
        """Save distributed parameters as raster files.

        Takes the generated 3D parameter array and saves each parameter
        layer as a separate GeoTIFF raster file.

        Args:
            path: Path to the folder where the parameter rasters will
                be saved.

        Raises:
            AssertionError: If `path` is not a string or does not exist.

        Note:
            The Parameters object should have the following attributes
            set before calling this method: ``DistParFn``, ``raster``,
            ``Par``, ``no_parameters``, ``snow``, ``kub``, and ``klb``.
        """
        assert isinstance(path, str), "path should be of type string"
        assert os.path.exists(path), f"{path} you have provided does not exist"

        # save
        if self.Snow == 0:  # now snow subroutine
            pnme = [
                "01_rfcf",
                "02_FC",
                "03_BETA",
                "04_ETF",
                "05_LP",
                "06_K0",
                "07_K1",
                "08_K2",
                "09_UZL",
                "10_PERC",
                "11_Kmuskingum",
                "12_Xmuskingum",
            ]
        else:  # there is snow subtoutine
            pnme = [
                "01_ltt",
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
                "18_perc",
            ]

        if path is not None:
            pnme = [
                path + i + "_" + str(dt.datetime.now())[0:10] + ".tif" for i in pnme
            ]

        for i in range(np.shape(self.Par3d)[2]):
            Dataset.dataset_like(
                self.raster, self.Par3d[:, :, i], driver="geotiff", path=pnme[i]
            )
