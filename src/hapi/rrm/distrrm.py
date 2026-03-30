"""Distributed rainfall-runoff model execution and spatial routing.

This module provides the ``DistributedRRM`` class, which runs a lumped
rainfall-runoff model (e.g., HBV) independently for each grid cell and
then routes the resulting discharge between cells following the river
network defined by a flow direction raster.

The module belongs to the ``hapi.rrm`` package and supports both
Muskingum and triangular (MAXBAS) routing strategies.
"""
from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset

from hapi.routing import Routing as routing


class DistributedRRM:
    """Distributed rainfall-runoff model runner and spatial router.

    Runs a lumped hydrological model separately for each grid cell
    and routes the resulting discharge between cells following the
    river network.

    The class is stateless; all methods are static and operate on a
    ``Model`` object that carries the required arrays and parameters.
    """

    def __init__(self):
        """Distributed constructor."""
        pass

    @staticmethod
    def run_lumped_model(Model):
        """Run lumped rainfall-runoff model for every grid cell.

        Executes the lumped conceptual model (e.g., HBV) independently
        for each non-NaN cell in the catchment grid and converts the
        resulting discharge from mm/time-step to m3/s.

        After execution the following attributes are set on *Model*:
        ``state_variables``, ``quz``, and ``qlz``.

        Args:
            Model: A catchment model object carrying the following
                attributes:

                - ``rows`` (int): Number of grid rows.
                - ``cols`` (int): Number of grid columns.
                - ``TS`` (int): Number of time steps.
                - ``FlowAccArr`` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - ``LumpedModel``: Lumped model instance with a
                  ``simulate`` method.
                - ``Prec`` (numpy.ndarray): 3-D precipitation array
                  ``(rows, cols, TS)``.
                - ``Temp`` (numpy.ndarray): 3-D temperature array.
                - ``ET`` (numpy.ndarray): 3-D evapotranspiration array.
                - ``ll_temp`` (numpy.ndarray): 3-D long-term average
                  temperature array.
                - ``Parameters`` (numpy.ndarray): 3-D parameter array
                  ``(rows, cols, n_params)``.
                - ``InitialCond`` (list): Initial state variable values
                  ``[sp, sm, uz, lz, wc]``.
                - ``q_init`` (float): Initial discharge in m3/s.
                - ``Snow`` (int): Snow module flag (0 or 1).
                - ``CatArea`` (float): Catchment area in km2.
                - ``px_tot_area`` (float): Total pixel area in km2.
                - ``px_area`` (float): Single pixel area in km2.
                - ``conversion_factor`` (float): Unit conversion
                  factor (``tfac * 3.6``).
        """
        Model.state_variables = np.zeros(
            [Model.rows, Model.cols, Model.TS, 5], dtype=np.float32
        )
        Model.quz = np.zeros([Model.rows, Model.cols, Model.TS], dtype=np.float32)
        Model.qlz = np.zeros([Model.rows, Model.cols, Model.TS], dtype=np.float32)

        for x in range(Model.rows):
            for y in range(Model.cols):
                # only for cells in the domain
                if not np.isnan(Model.FlowAccArr[x, y]):
                    (
                        Model.quz[x, y, :],
                        Model.qlz[x, y, :],
                        Model.state_variables[x, y, :, :],
                    ) = Model.LumpedModel.simulate(
                        prec=Model.Prec[x, y, :],
                        temp=Model.Temp[x, y, :],
                        et=Model.ET[x, y, :],
                        ll_temp=Model.ll_temp[x, y, :],
                        par=Model.Parameters[x, y, :],
                        init_st=Model.InitialCond,
                        q_init=Model.q_init,
                        snow=Model.Snow,
                    )

        area_coef = Model.CatArea / Model.px_tot_area
        # convert quz from mm/time step to m3/sec
        Model.quz = (
            Model.quz * Model.px_area * area_coef / Model.conversion_factor
        )  # Timef*3.6
        # convert Qlz to m3/sec
        Model.qlz = (
            Model.qlz * Model.px_area * area_coef / Model.conversion_factor
        )  # Timef*3.6

    @staticmethod
    def SpatialRouting(Model):
        """Route discharge between cells following the flow direction.

        Accumulates and routes upper-zone discharge (``quz``) using
        Muskingum routing from upstream to downstream cells according
        to the flow direction raster.  Lower-zone discharge (``qlz``)
        is translated (accumulated without attenuation) so that total
        discharge can be computed at any internal point.

        After execution the following attributes are set on *Model*:
        ``quz_routed``, ``qlz_translated``, and ``Qtot``.

        Args:
            Model: A catchment model object carrying the following
                attributes:

                - ``rows`` (int): Number of grid rows.
                - ``cols`` (int): Number of grid columns.
                - ``TS`` (int): Number of time steps.
                - ``FlowAccArr`` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - ``quz`` (numpy.ndarray): 3-D upper-zone discharge
                  array ``(rows, cols, TS)`` in m3/s.
                - ``qlz`` (numpy.ndarray): 3-D lower-zone discharge
                  array ``(rows, cols, TS)`` in m3/s.
                - ``acc_val`` (list): Sorted unique flow accumulation
                  values.
                - ``FDT`` (dict): Flow direction table mapping
                  ``"row,col"`` keys to lists of upstream cell
                  index pairs.
                - ``Parameters`` (numpy.ndarray): 3-D parameter array
                  where indices 10 and 11 are Muskingum K and X.
                - ``dt`` (float): Time-step factor (``tfac``).
                - ``routing_method`` (str): Routing method name (e.g.,
                  ``"Muskingum"``).
                - ``BankfullDepth`` (numpy.ndarray): 2-D bankfull
                  depth array used for non-Muskingum methods.
        """
        #    # routing lake discharge with DS cell k & x and adding to cell Q
        #    q_lake=Routing.Muskingum_V(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
        #    q_lake=np.append(q_lake,q_lake[-1])
        #    # both lake & Quz are in m3/s
        #    #new
        #    quz[lakecell[0],lakecell[1],:]=quz[lakecell[0],lakecell[1],:]+q_lake

        # cells at the divider
        Model.quz_routed = np.zeros_like(Model.quz)

        # lower zone discharge is going to be just translated without any attenuation
        # in order to be able to calculate total discharge (uz+lz) at internal points
        # in the catchment

        Model.qlz_translated = np.zeros_like(Model.quz)
        # Model.Qtot = np.zeros_like(Model.quz)
        # for all cells with 0 flow acc put the quz
        for x in range(Model.rows):  # no of rows
            for y in range(Model.cols):  # no of columns
                if not np.isnan(Model.FlowAccArr[x, y]) and Model.FlowAccArr[x, y] == 0:
                    Model.quz_routed[x, y, :] = Model.quz[x, y, :]
                    Model.qlz_translated[x, y, :] = Model.qlz[x, y, :]

        # remaining cells
        for j in range(1, len(Model.acc_val)):
            # TODO parallelize
            # all cells with the same acc_val can run at the same time
            for x in range(Model.rows):  # no of rows
                for y in range(Model.cols):  # no of columns
                    # check from total flow accumulation
                    if (
                        not np.isnan(Model.FlowAccArr[x, y])
                        and Model.FlowAccArr[x, y] == Model.acc_val[j]
                    ):
                        if (
                            Model.routing_method != "Muskingum"
                            and Model.BankfullDepth[x, y] > 0
                        ):
                            continue
                        else:
                            # for UZ
                            q_uzi = np.zeros(Model.TS)
                            # for lz
                            qlzi = np.zeros(Model.TS)
                            # iterate to route uz and translate lz
                            for i in range(
                                len(Model.FDT[str(x) + "," + str(y)])
                            ):  # Model.acc_val[j]
                                # bring the indexes of the us cell
                                x_ind = Model.FDT[str(x) + "," + str(y)][i][0]
                                y_ind = Model.FDT[str(x) + "," + str(y)][i][1]
                                # sum the Q of the US cells (already routed for its cell)
                                # route first with there own k & xthen sum
                                q_uzi = q_uzi + routing.Muskingum_V(
                                    Model.quz_routed[x_ind, y_ind, :],
                                    Model.quz_routed[x_ind, y_ind, 0],
                                    Model.Parameters[x_ind, y_ind, 10],
                                    Model.Parameters[x_ind, y_ind, 11],
                                    Model.dt,
                                )

                                qlzi = qlzi + Model.qlz_translated[x_ind, y_ind, :]

                            # add the routed upstream flows to the current Quz in the cell
                            Model.quz_routed[x, y, :] = Model.quz[x, y, :] + q_uzi
                            Model.qlz_translated[x, y, :] = Model.qlz[x, y, :] + qlzi
        Model.Qtot = Model.qlz_translated + Model.quz_routed

    @staticmethod
    def DistMaxbas1(Model):
        """Route discharge to the outlet using a triangular function.

        Applies triangular (MAXBAS) routing to the upper-zone
        discharge of each cell independently.  The MAXBAS parameter
        is read from the last column of the spatially distributed
        parameter array.

        The ``Model.quz`` array is modified in place.

        Args:
            Model: A catchment model object carrying the following
                attributes:

                - ``rows`` (int): Number of grid rows.
                - ``cols`` (int): Number of grid columns.
                - ``FlowAccArr`` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - ``Parameters`` (numpy.ndarray): 3-D parameter array
                  where the last index holds the MAXBAS value.
                - ``quz`` (numpy.ndarray): 3-D upper-zone discharge
                  array ``(rows, cols, TS)`` in m3/s.
        """
        Maxbas = Model.Parameters[:, :, -1]

        for x in range(Model.rows):
            for y in range(Model.cols):
                if not np.isnan(Model.FlowAccArr[x, y]):
                    Model.quz[x, y, :] = routing.TriangularRouting1(
                        Model.quz[x, y, :], Maxbas[x, y]
                    )

    @staticmethod
    def DistMaxbas2(Model):
        """Route discharge using a triangular function scaled by flow path length.

        Similar to ``DistMaxbas1``, but the MAXBAS parameter for each
        cell is rescaled proportionally to its flow path length so that
        cells farther from the outlet receive more attenuation.

        The ``Model.quz`` array is modified in place.

        Args:
            Model: A catchment model object carrying the following
                attributes:

                - ``rows`` (int): Number of grid rows.
                - ``cols`` (int): Number of grid columns.
                - ``FlowAccArr`` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - ``FPLArr`` (numpy.ndarray): 2-D flow path length
                  array.
                - ``NoDataValue`` (float): No-data value used in the
                  flow path length raster.
                - ``Parameters`` (numpy.ndarray): 3-D parameter array
                  where the last index holds the maximum MAXBAS value.
                - ``quz`` (numpy.ndarray): 3-D upper-zone discharge
                  array ``(rows, cols, TS)`` in m3/s.
        """
        MAXBAS = np.nanmax(Model.Parameters[:, :, -1])
        # replace novalue cells by nan
        Model.FPLArr[Model.FPLArr == Model.NoDataValue] = np.nan

        MaxFPL = np.nanmax(Model.FPLArr)
        MinFPL = np.nanmin(Model.FPLArr)
        # resize_fun = lambda x: np.round(((((x - min_dist)/(max_dist - min_dist))*(1*maxbas - 1)) + 1), 0)
        resize_fun = lambda g: (
            (((g - MinFPL) / (MaxFPL - MinFPL)) * (1 * MAXBAS - 1)) + 1
        )

        NormalizedFPL = resize_fun(Model.FPLArr)

        for x in range(Model.rows):
            for y in range(Model.cols):
                if not np.isnan(Model.FPLArr[x, y]):
                    Model.quz[x, y, :] = routing.TriangularRouting2(
                        Model.quz[x, y, :], NormalizedFPL[x, y]
                    )

    @staticmethod
    def Dist_HBV2(
        conceptual_model,
        lakecell,
        q_lake,
        DEM,
        flow_acc,
        flow_acc_plan,
        sp_prec,
        sp_et,
        sp_temp,
        sp_pars,
        p2,
        init_st=None,
        ll_temp=None,
        q_0=None,
    ):
        """Run distributed HBV model with lake routing (legacy).

        Executes the HBV conceptual model for every grid cell, routes
        lake discharge into the downstream cell using Muskingum
        routing, and then routes upper-zone discharge through the
        river network.  Lower-zone discharge is averaged across all
        cells and converted to m3/s.

        Args:
            conceptual_model: Lumped model object with a ``simulate``
                method.
            lakecell (list[int]): Two-element list ``[row, col]``
                giving the grid indices of the lake cell.
            q_lake (numpy.ndarray): 1-D array of lake discharge
                time series in m3/s.
            DEM: GDAL dataset of the catchment DEM.
            flow_acc (dict): Flow direction table mapping
                ``"row,col"`` keys to lists of upstream cell index
                pairs.
            flow_acc_plan (numpy.ndarray): 2-D array of flow
                accumulation values; NaN marks no-data cells.
            sp_prec (numpy.ndarray): 3-D precipitation array
                ``(rows, cols, time_steps)``.
            sp_et (numpy.ndarray): 3-D evapotranspiration array.
            sp_temp (numpy.ndarray): 3-D temperature array.
            sp_pars (numpy.ndarray): 3-D parameter array
                ``(rows, cols, n_params)``.  Indices 5, 6, 7 are
                K1, K, and alpha; indices 10, 11 are Muskingum
                K and X.
            p2 (list): Unoptimized parameters.

                - ``p2[0]``: tfac -- 1 for hourly, 0.25 for 15 min,
                  24 for daily.
                - ``p2[1]``: Catchment area in km2.
            init_st (list, optional): Initial state variable values
                ``[sp, sm, uz, lz, wc]``.  Defaults to None.
            ll_temp (numpy.ndarray, optional): 3-D long-term average
                temperature array.  Defaults to None.
            q_0 (float, optional): Initial discharge in m3/s.
                Defaults to None.

        Returns:
            tuple: A five-element tuple containing:

                - **qout** (*numpy.ndarray*): 1-D discharge time
                  series at the catchment outlet in m3/s.
                - **st** (*numpy.ndarray*): 4-D state variable array
                  ``(rows, cols, time_steps, 5)`` with states
                  ``[sp, sm, uz, lz, wc]``.
                - **quz_routed** (*numpy.ndarray*): 3-D routed
                  upper-zone discharge array in m3/s.
                - **qlz** (*numpy.ndarray*): 1-D spatially averaged
                  lower-zone discharge in m3/s.
                - **quz** (*numpy.ndarray*): 3-D upper-zone
                  discharge array in m3/s (before spatial routing).
        """
        n_steps = sp_prec.shape[2] + 1  # no of time steps =length of time series +1
        # initialize vector of nans to fill states
        dummy_states = np.empty([n_steps, 5])  # [sp,sm,uz,lz,wc]
        dummy_states[:] = np.nan

        # Get the mask
        # mask, no_val = raster.get_mask(DEM)
        dataset = Dataset(DEM)
        no_val = dataset.no_data_value[0]
        mask = dataset.read_array()
        # shape of the fpl raster (rows, columns)-------------- rows are x and columns are y
        x_ext, y_ext = mask.shape
        #    y_ext, x_ext = mask.shape # shape of the fpl raster (rows, columns)------------ should change rows are y and columns are x

        # Get deltas of pixel
        geo_trans = (
            DEM.GetGeoTransform()
        )  # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        dx = np.abs(geo_trans[1]) / 1000.0  # dx in Km
        dy = np.abs(geo_trans[-1]) / 1000.0  # dy in Km
        px_area = dx * dy  # area of the cell

        # Enumerate the total number of pixels in the catchment
        tot_elem = np.sum(
            np.sum([[1 for elem in mask_i if elem != no_val] for mask_i in mask])
        )  # get row by row and search [mask_i for mask_i in mask]

        # total pixel area
        px_tot_area = tot_elem * px_area  # total area of pixels

        # Get number of non-value data

        st = []  # Spatially distributed states
        qlz = []
        quz = []
        # ------------------------------------------------------------------------------
        for x in range(x_ext):  # no of rows
            st_i = []
            q_lzi = []
            q_uzi = []
            #        q_out_i = []
            # run all cells in one row ----------------------------------------------------
            for y in range(y_ext):  # no of columns
                if mask[x, y] != no_val:  # only for cells in the domain
                    # Calculate the states per cell
                    # TODO optimise for multiprocessing these loops
                    #                _, _st, _uzg, _lzg = conceptual_model.simulate_new_model(avg_prec = sp_prec[x, y,:],
                    _, _st, _uzg, _lzg = conceptual_model.simulate(
                        prec=sp_prec[x, y, :],
                        temp=sp_temp[x, y, :],
                        et=sp_et[x, y, :],
                        par=sp_pars[x, y, :],
                        p2=p2,
                        init_st=init_st,
                        ll_temp=None,
                        q_0=q_0,
                        snow=0,
                    )
                    # append column after column in the same row -----------------
                    st_i.append(np.array(_st))
                    # calculate upper zone Q = K1*(LZ_int_1)
                    q_lz_temp = np.array(sp_pars[x, y, 6]) * _lzg
                    q_lzi.append(q_lz_temp)
                    # calculate lower zone Q = k*(UZ_int_3)**(1+alpha)
                    q_uz_temp = np.array(sp_pars[x, y, 5]) * (
                        np.power(_uzg, (1.0 + sp_pars[x, y, 7]))
                    )
                    q_uzi.append(q_uz_temp)

                    # print("total = "+str(fff)+"/"+str(tot_elem)+" cell, row= "+str(x+1)+" column= "+str(y+1) )
                else:  # if the cell is novalue-------------------------------------
                    # Fill the empty cells with a nan vector
                    st_i.append(
                        dummy_states
                    )  # fill all states(5 states) for all time steps = nan
                    q_lzi.append(
                        dummy_states[:, 0]
                    )  # q lower zone =nan  for all time steps = nan
                    q_uzi.append(
                        dummy_states[:, 0]
                    )  # q upper zone =nan  for all time steps = nan

            # store row by row-------- ----------------------------------------------------
            # st.append(st_i) # state variables
            st.append(st_i)  # state variables
            qlz.append(np.array(q_lzi))  # lower zone discharge mm/timestep
            quz.append(np.array(q_uzi))  # upper zone routed discharge mm/timestep
            # ------------------------------------------------------------------------------
            # convert to arrays
        st = np.array(st)  # type: ignore
        qlz = np.array(qlz)  # type: ignore
        quz = np.array(quz)  # type: ignore
        # convert quz from mm/time step to m3/sec
        area_coef = p2[1] / px_tot_area
        quz = quz * px_area * area_coef / (p2[0] * 3.6)

        no_cells = list(
            set(
                [
                    flow_acc_plan[i, j]
                    for i in range(x_ext)
                    for j in range(y_ext)
                    if not np.isnan(flow_acc_plan[i, j])
                ]
            )
        )
        #    no_cells=list(set([int(flow_acc_plan[i,j]) for i in range(x_ext) for j in range(y_ext) if flow_acc_plan[i,j] != no_val]))
        no_cells.sort()

        # routing lake discharge with DS cell k & x and adding to cell Q
        q_lake = routing.Muskingum_V(
            q_lake,
            q_lake[0],
            sp_pars[lakecell[0], lakecell[1], 10],
            sp_pars[lakecell[0], lakecell[1], 11],
            p2[0],
        )
        q_lake = np.append(q_lake, q_lake[-1])
        # both lake & Quz are in m3/s
        # new
        quz[lakecell[0], lakecell[1], :] = quz[lakecell[0], lakecell[1], :] + q_lake
        # cells at the divider
        quz_routed = np.zeros_like(quz) * np.nan
        # for all cell with 0 flow acc put the quz
        for x in range(x_ext):  # no of rows
            for y in range(y_ext):  # no of columns
                if mask[x, y] != no_val and flow_acc_plan[x, y] == 0:
                    quz_routed[x, y, :] = quz[x, y, :]
        # new
        for j in range(1, len(no_cells)):  # 2):#
            for x in range(x_ext):  # no of rows
                for y in range(y_ext):  # no of columns
                    # check from total flow accumulation
                    if mask[x, y] != no_val and flow_acc_plan[x, y] == no_cells[j]:
                        #                        print(no_cells[j])
                        q_r = np.zeros(n_steps)
                        for i in range(
                            len(flow_acc[str(x) + "," + str(y)])
                        ):  # no_cells[j]
                            # bring the indexes of the us cell
                            x_ind = flow_acc[str(x) + "," + str(y)][i][0]
                            y_ind = flow_acc[str(x) + "," + str(y)][i][1]
                            # sum the Q of the US cells (already routed for its cell)
                            # route first with there own k & xthen sum
                            q_r = q_r + routing.Muskingum_V(
                                quz_routed[x_ind, y_ind, :],
                                quz_routed[x_ind, y_ind, 0],
                                sp_pars[x_ind, y_ind, 10],
                                sp_pars[x_ind, y_ind, 11],
                                p2[0],
                            )
                        #                        q=q_r
                        # add the routed upstream flows to the current Quz in the cell
                        quz_routed[x, y, :] = quz[x, y, :] + q_r
        # check if the max flow _acc is at the outlet
        #    if tot_elem != np.nanmax(flow_acc_plan):
        #        raise ("flow accumulation plan is not correct")
        # outlet is the cell that has the max flow_acc
        outlet = np.where(
            flow_acc_plan == np.nanmax(flow_acc_plan)
        )  # np.nanmax(flow_acc_plan)
        outletx = outlet[0][0]
        outlety = outlet[1][0]

        qlz = np.array(  # type: ignore[assignment]
            [np.nanmean(qlz[:, :, i]) for i in range(n_steps)]  # type: ignore[call-overload]
        )  # average of all cells (not routed mm/timestep)
        # convert Qlz to m3/sec
        qlz = qlz * p2[1] / (p2[0] * 3.6)  # generation

        qout = qlz + quz_routed[outletx, outlety, :]

        return qout, st, quz_routed, qlz, quz
