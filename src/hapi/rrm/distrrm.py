"""Distributed rainfall-runoff model execution and spatial routing.

This module provides the `DistributedRRM` class, which runs a lumped
rainfall-runoff model (e.g., HBV) independently for each grid cell and
then routes the resulting discharge between cells following the river
network defined by a flow direction raster.

The module belongs to the `hapi.rrm` package and supports both
Muskingum and triangular (MAXBAS) routing strategies.
"""

from __future__ import annotations

import numpy as np

from hapi.results import RoutingKind, SimulationResults
from hapi.routing import Routing as routing


class DistributedRRM:
    """Distributed rainfall-runoff model runner and spatial router.

    Runs a lumped hydrological model separately for each grid cell
    and routes the resulting discharge between cells following the
    river network.

    The class is stateless; all methods are static and operate on a
    `Model` object that carries the required arrays and parameters.
    """

    def __init__(self):
        """Distributed constructor."""
        pass

    @staticmethod
    def run_lumped_model(Model) -> SimulationResults:
        """Run lumped rainfall-runoff model for every grid cell.

        Executes the lumped conceptual model (e.g., HBV) independently
        for each non-NaN cell in the catchment grid and converts the
        resulting discharge from mm/time-step to m3/s.

        Builds `Model.results` and returns it, so a caller that needs the arrays does
        not have to read them back off the model and re-narrow them from `| None`.

        Returns:
            SimulationResults: The results object, carrying `state_variables`, `quz` and
            `qlz`, with `routing` still `RoutingKind.UNROUTED`.

        Args:
            Model (Catchment): A catchment model object carrying the following
                attributes:

                - `rows` (int): Number of grid rows.
                - `cols` (int): Number of grid columns.
                - `TS` (int): Number of time steps.
                - `flow_acc_arr` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - `LumpedModel`: Lumped model instance with a
                  `simulate` method.
                - `Prec` (numpy.ndarray): 3-D precipitation array
                  `(rows, cols, TS)`.
                - `Temp` (numpy.ndarray): 3-D temperature array.
                - `ET` (numpy.ndarray): 3-D evapotranspiration array.
                - `ll_temp` (numpy.ndarray): 3-D long-term average
                  temperature array.
                - `Parameters` (numpy.ndarray): 3-D parameter array
                  `(rows, cols, n_params)`.
                - `InitialCond` (list): Initial state variable values
                  `[sp, sm, uz, lz, wc]`.
                - `q_init` (float): Initial discharge in m3/s.
                - `Snow` (int): Snow module flag (0 or 1).
                - `CatArea` (float): Catchment area in km2.
                - `px_tot_area` (float): Total pixel area in km2.
                - `px_area` (float): Single pixel area in km2.
                - `conversion_factor` (float): Unit conversion
                  factor (`tfac * 3.6`).
        """
        grid = (
            Model.flow_network.rows,
            Model.flow_network.cols,
            Model.meteo.simulation_steps,
        )
        # A fresh results object per run, rather than nine attributes overwritten one at a
        # time: a half-finished run is then distinguishable from a finished one, and the
        # routed fields of a *previous* run cannot survive into this one.
        results = SimulationResults(
            routing=RoutingKind.UNROUTED,
            quz=np.zeros(grid, dtype=np.float32),
            qlz=np.zeros(grid, dtype=np.float32),
            state_variables=np.zeros((*grid, 5), dtype=np.float32),
        )
        Model.results = results

        for x in range(Model.flow_network.rows):
            for y in range(Model.flow_network.cols):
                # only for cells in the domain
                if not np.isnan(Model.flow_network.flow_acc_arr[x, y]):
                    (
                        results.quz[x, y, :],
                        results.qlz[x, y, :],
                        results.state_variables[x, y, :, :],
                    ) = Model.lumped_model.simulate(
                        prec=Model.meteo.precipitation[x, y, :],
                        temp=Model.meteo.temperature[x, y, :],
                        et=Model.meteo.evapotranspiration[x, y, :],
                        ll_temp=Model.meteo.ll_temp[x, y, :],
                        par=Model.parameters[x, y, :],
                        init_st=Model.initial_cond,
                        q_init=Model.q_init,
                        snow=Model.snow,
                    )

        area_coef = Model.area / Model.flow_network.px_tot_area
        factor = Model.flow_network.px_area * area_coef / Model.conversion_factor
        # convert quz and qlz from mm/time step to m3/sec  # Timef*3.6
        results.quz = results.quz * factor
        results.qlz = results.qlz * factor
        return results

    @staticmethod
    def route_muskingum(Model):
        """Route discharge between cells following the flow direction.

        Accumulates and routes upper-zone discharge (`quz`) using
        Muskingum routing from upstream to downstream cells according
        to the flow direction raster.  Lower-zone discharge (`qlz`)
        is translated (accumulated without attenuation) so that total
        discharge can be computed at any internal point.

        After execution the following attributes are set on *Model*:
        `quz_routed`, `qlz_translated`, and `q_total`.

        Args:
            Model (Catchment): A catchment model object carrying the following
                attributes:

                - `rows` (int): Number of grid rows.
                - `cols` (int): Number of grid columns.
                - `TS` (int): Number of time steps.
                - `flow_acc_arr` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - `quz` (numpy.ndarray): 3-D upper-zone discharge
                  array `(rows, cols, TS)` in m3/s.
                - `qlz` (numpy.ndarray): 3-D lower-zone discharge
                  array `(rows, cols, TS)` in m3/s.
                - `acc_val` (list): Sorted unique flow accumulation
                  values.
                - `FDT` (dict): Flow direction table mapping
                  `"row,col"` keys to lists of upstream cell
                  index pairs.
                - `Parameters` (numpy.ndarray): 3-D parameter array
                  where indices 10 and 11 are Muskingum K and X.
                - `dt` (float): Time-step factor (`tfac`).
                - `routing_method` (str): Routing method name (e.g.,
                  `"Muskingum"`).
                - `bankfull_depth` (numpy.ndarray): 2-D bankfull
                  depth array used for non-Muskingum methods.
        """
        #    # routing lake discharge with DS cell k & x and adding to cell Q
        #    q_lake=Routing.muskingum_v(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
        #    q_lake=np.append(q_lake,q_lake[-1])
        #    # both lake & Quz are in m3/s
        #    #new
        #    quz[lakecell[0],lakecell[1],:]=quz[lakecell[0],lakecell[1],:]+q_lake

        results = Model.results
        # cells at the divider
        results.quz_routed = np.zeros_like(results.quz)

        # lower zone discharge is going to be just translated without any attenuation
        # in order to be able to calculate total discharge (uz+lz) at internal points
        # in the catchment

        results.qlz_translated = np.zeros_like(results.quz)
        # for all cells with 0 flow acc put the quz
        for x in range(Model.flow_network.rows):  # no of rows
            for y in range(Model.flow_network.cols):  # no of columns
                if (
                    not np.isnan(Model.flow_network.flow_acc_arr[x, y])
                    and Model.flow_network.flow_acc_arr[x, y] == 0
                ):
                    results.quz_routed[x, y, :] = results.quz[x, y, :]
                    results.qlz_translated[x, y, :] = results.qlz[x, y, :]

        # remaining cells
        # Read once: this is the routing inner loop, and `acc_val` scans the whole grid.
        acc_val = Model.flow_network.acc_val
        for j in range(1, len(acc_val)):
            # TODO parallelize
            # all cells with the same acc_val can run at the same time
            for x in range(Model.flow_network.rows):  # no of rows
                for y in range(Model.flow_network.cols):  # no of columns
                    # check from total flow accumulation
                    if (
                        not np.isnan(Model.flow_network.flow_acc_arr[x, y])
                        and Model.flow_network.flow_acc_arr[x, y] == acc_val[j]
                    ):
                        if (
                            Model.routing_method != "Muskingum"
                            and Model.bankfull_depth[x, y] > 0
                        ):
                            continue
                        else:
                            # for UZ
                            q_uzi = np.zeros(Model.meteo.simulation_steps)
                            # for lz
                            qlzi = np.zeros(Model.meteo.simulation_steps)
                            # iterate to route uz and translate lz
                            for i in range(
                                len(Model.flow_network.FDT[str(x) + "," + str(y)])
                            ):  # Model.acc_val[j]
                                # bring the indexes of the us cell
                                x_ind = Model.flow_network.FDT[str(x) + "," + str(y)][
                                    i
                                ][0]
                                y_ind = Model.flow_network.FDT[str(x) + "," + str(y)][
                                    i
                                ][1]
                                # sum the Q of the US cells (already routed for its cell)
                                # route first with there own k & xthen sum
                                q_uzi = q_uzi + routing.muskingum_v(
                                    results.quz_routed[x_ind, y_ind, :],
                                    results.quz_routed[x_ind, y_ind, 0],
                                    Model.parameters[x_ind, y_ind, 10],
                                    Model.parameters[x_ind, y_ind, 11],
                                    Model.dt,
                                )

                                qlzi = qlzi + results.qlz_translated[x_ind, y_ind, :]

                            # add the routed upstream flows to the current Quz in the cell
                            results.quz_routed[x, y, :] = results.quz[x, y, :] + q_uzi
                            results.qlz_translated[x, y, :] = (
                                results.qlz[x, y, :] + qlzi
                            )
        results.q_total = results.qlz_translated + results.quz_routed
        # Muskingum accumulates downstream, so a cell of `q_total` is the discharge at that
        # cell and the outlet-cell shortcut in `extract_discharge` is valid.
        results.routing = RoutingKind.MUSKINGUM

    @staticmethod
    def route_maxbas(Model):
        """Route discharge to the outlet using a triangular function.

        Applies triangular (MAXBAS) routing to the upper-zone
        discharge of each cell independently.  The MAXBAS parameter
        is read from the last column of the spatially distributed
        parameter array.

        The `Model.results.quz` array is modified in place.

        Args:
            Model (Catchment): A catchment model object carrying the following
                attributes:

                - `rows` (int): Number of grid rows.
                - `cols` (int): Number of grid columns.
                - `flow_acc_arr` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - `Parameters` (numpy.ndarray): 3-D parameter array
                  where the last index holds the MAXBAS value.
                - `quz` (numpy.ndarray): 3-D upper-zone discharge
                  array `(rows, cols, TS)` in m3/s.
        """
        Maxbas = Model.parameters[:, :, -1]
        quz = Model.results.quz

        for x in range(Model.flow_network.rows):
            for y in range(Model.flow_network.cols):
                if not np.isnan(Model.flow_network.flow_acc_arr[x, y]):
                    quz[x, y, :] = routing.triangular_routing_1(
                        quz[x, y, :], Maxbas[x, y]
                    )

    @staticmethod
    def route_maxbas_by_path_length(Model):
        """Route discharge using a triangular function scaled by flow path length.

        Similar to `route_maxbas`, but the MAXBAS parameter for each
        cell is rescaled proportionally to its flow path length so that
        cells farther from the outlet receive more attenuation.

        The `Model.results.quz` array is modified in place.

        Args:
            Model (Catchment): A catchment model object carrying the following
                attributes:

                - `rows` (int): Number of grid rows.
                - `cols` (int): Number of grid columns.
                - `flow_acc_arr` (numpy.ndarray): 2-D flow accumulation
                  array; NaN marks cells outside the domain.
                - `flow_path_length_arr` (numpy.ndarray): 2-D flow path length
                  array.
                - `no_data_value` (float): No-data value used in the
                  flow path length raster.
                - `Parameters` (numpy.ndarray): 3-D parameter array
                  where the last index holds the maximum MAXBAS value.
                - `quz` (numpy.ndarray): 3-D upper-zone discharge
                  array `(rows, cols, TS)` in m3/s.
        """
        MAXBAS = np.nanmax(Model.parameters[:, :, -1])
        # `read_flow_path_length` already masks this raster's own no-data cells to NaN via
        # pyramids, so no sentinel comparison is needed here -- and the one that used to sit
        # here compared against the *accumulation* raster's sentinel, which is a different
        # raster and need not share a no-data value.

        MaxFPL = np.nanmax(Model.flow_path_length_arr)
        MinFPL = np.nanmin(Model.flow_path_length_arr)
        # resize_fun = lambda x: np.round(((((x - min_dist)/(max_dist - min_dist))*(1*maxbas - 1)) + 1), 0)
        resize_fun = lambda g: (
            (((g - MinFPL) / (MaxFPL - MinFPL)) * (1 * MAXBAS - 1)) + 1
        )

        NormalizedFPL = resize_fun(Model.flow_path_length_arr)
        quz = Model.results.quz

        for x in range(Model.flow_network.rows):
            for y in range(Model.flow_network.cols):
                if not np.isnan(Model.flow_path_length_arr[x, y]):
                    quz[x, y, :] = routing.triangular_routing_2(
                        quz[x, y, :], NormalizedFPL[x, y]
                    )
