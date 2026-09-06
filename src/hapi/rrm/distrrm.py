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
from hapi.runs import DistributedRun


class DistributedRRM:
    """Distributed rainfall-runoff model runner and spatial router.

    Runs a lumped hydrological model separately for each grid cell
    and routes the resulting discharge between cells following the
    river network.

    The class is stateless. Every method takes a :class:`~hapi.runs.DistributedRun` --
    validated, non-optional inputs -- and either returns the results it built or mutates the
    :class:`~hapi.results.SimulationResults` it is handed. Nothing here reads or writes a
    catchment, so nothing here has to ask whether its inputs were checked.
    """

    def __init__(self):
        """Distributed constructor."""
        pass

    @staticmethod
    def run_lumped_model(run: DistributedRun) -> SimulationResults:
        """Run lumped rainfall-runoff model for every grid cell.

        Args:
            run: The validated inputs. Reads the flow network, the drivers, the parameter set
                and the conceptual model setup.

        Returns:
            SimulationResults: A fresh results object carrying `state_variables`, `quz` and
            `qlz`, with `routing` still `RoutingKind.UNROUTED` -- a routing step sets it.
        """
        grid = (
            run.flow_network.rows,
            run.flow_network.cols,
            run.meteo.simulation_steps,
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

        for x in range(run.flow_network.rows):
            for y in range(run.flow_network.cols):
                # only for cells in the domain
                if not np.isnan(run.flow_network.flow_acc_arr[x, y]):
                    (
                        results.quz[x, y, :],
                        results.qlz[x, y, :],
                        results.state_variables[x, y, :, :],
                    ) = run.model_setup.model.simulate(
                        prec=run.meteo.precipitation[x, y, :],
                        temp=run.meteo.temperature[x, y, :],
                        et=run.meteo.evapotranspiration[x, y, :],
                        ll_temp=run.meteo.ll_temp[x, y, :],
                        par=run.parameter_cube[x, y, :],
                        init_st=run.model_setup.initial_cond,
                        q_init=run.model_setup.q_init,
                        snow=run.parameters.snow,
                    )

        area_coef = run.model_setup.area / run.flow_network.px_tot_area
        factor = run.flow_network.px_area * area_coef / run.period.conversion_factor
        # convert quz and qlz from mm/time step to m3/sec  # Timef*3.6
        results.quz = results.quz * factor
        results.qlz = results.qlz * factor
        return results

    @staticmethod
    def route_muskingum(run: DistributedRun, results: SimulationResults) -> None:
        """Route discharge between cells following the flow direction.

        Accumulates and routes upper-zone discharge from upstream to downstream cells along
        the flow-direction network, and translates the lower zone (accumulated without
        attenuation) so total discharge can be read at any internal point. Fills
        `quz_routed`, `qlz_translated` and `q_total` on `results`, and records
        `RoutingKind.MUSKINGUM`.

        Args:
            run: The validated inputs. `skip_hydraulic_cells` leaves cells with a positive
                `river_geometry.bankfull_depth` unrouted, because a 1D hydraulic model routes
                them instead; the run type has already checked the geometry is present.
            results: The results to route, as returned by :meth:`run_lumped_model`. Mutated
                in place.
        """
        #    # routing lake discharge with DS cell k & x and adding to cell Q
        #    q_lake=Routing.muskingum_v(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
        #    q_lake=np.append(q_lake,q_lake[-1])
        #    # both lake & Quz are in m3/s
        #    #new
        #    quz[lakecell[0],lakecell[1],:]=quz[lakecell[0],lakecell[1],:]+q_lake

        # cells at the divider
        # `DistributedRun` has already refused a skip with no geometry, so this is the value
        # that guard proved is there -- bound once, outside the loop it is read in.
        river_depth = (
            run.river_geometry.bankfull_depth
            if run.skip_hydraulic_cells and run.river_geometry is not None
            else None
        )
        results.quz_routed = np.zeros_like(results.quz)

        # lower zone discharge is going to be just translated without any attenuation
        # in order to be able to calculate total discharge (uz+lz) at internal points
        # in the catchment

        results.qlz_translated = np.zeros_like(results.quz)
        # for all cells with 0 flow acc put the quz
        for x in range(run.flow_network.rows):  # no of rows
            for y in range(run.flow_network.cols):  # no of columns
                if (
                    not np.isnan(run.flow_network.flow_acc_arr[x, y])
                    and run.flow_network.flow_acc_arr[x, y] == 0
                ):
                    results.quz_routed[x, y, :] = results.quz[x, y, :]
                    results.qlz_translated[x, y, :] = results.qlz[x, y, :]

        # remaining cells
        # Read once: this is the routing inner loop, and `acc_val` scans the whole grid.
        acc_val = run.flow_network.acc_val
        for j in range(1, len(acc_val)):
            # TODO parallelize
            # all cells with the same acc_val can run at the same time
            for x in range(run.flow_network.rows):  # no of rows
                for y in range(run.flow_network.cols):  # no of columns
                    # check from total flow accumulation
                    if (
                        not np.isnan(run.flow_network.flow_acc_arr[x, y])
                        and run.flow_network.flow_acc_arr[x, y] == acc_val[j]
                    ):
                        if river_depth is not None and river_depth[x, y] > 0:
                            # A river cell a 1D hydraulic model will route instead. The
                            # caller says so explicitly; this used to be inferred from
                            # `routing_method != "Muskingum"`, which meant any catchment
                            # built with a non-Muskingum method dereferenced
                            # `bankfull_depth` -- None outside the flood model -- and
                            # crashed here.
                            continue
                        else:
                            # for UZ
                            q_uzi = np.zeros(run.meteo.simulation_steps)
                            # for lz
                            qlzi = np.zeros(run.meteo.simulation_steps)
                            # iterate to route uz and translate lz
                            for i in range(
                                len(run.routing_table[str(x) + "," + str(y)])
                            ):
                                # bring the indexes of the us cell
                                x_ind = run.routing_table[str(x) + "," + str(y)][i][0]
                                y_ind = run.routing_table[str(x) + "," + str(y)][i][1]
                                # sum the Q of the US cells (already routed for its cell)
                                # route first with there own k & xthen sum
                                q_uzi = q_uzi + routing.muskingum_v(
                                    results.quz_routed[x_ind, y_ind, :],
                                    results.quz_routed[x_ind, y_ind, 0],
                                    run.parameter_cube[x_ind, y_ind, 10],
                                    run.parameter_cube[x_ind, y_ind, 11],
                                    run.period.dt,
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
    def route_maxbas(run: DistributedRun, results: SimulationResults) -> None:
        """Route discharge to the outlet using a triangular function.

        Applies triangular (MAXBAS) routing to each cell's upper-zone discharge independently,
        reading the MAXBAS parameter from the last column of the parameter array. `results.quz`
        is modified in place.

        Args:
            run: The validated inputs.
            results: The results to route. Mutated in place.
        """
        Maxbas = run.parameter_cube[:, :, -1]
        quz = results.quz

        for x in range(run.flow_network.rows):
            for y in range(run.flow_network.cols):
                if not np.isnan(run.flow_network.flow_acc_arr[x, y]):
                    quz[x, y, :] = routing.triangular_routing_1(
                        quz[x, y, :], Maxbas[x, y]
                    )

    @staticmethod
    def route_maxbas_by_path_length(
        run: DistributedRun, results: SimulationResults
    ) -> None:
        """Route discharge using a triangular function scaled by flow path length.

        Like :meth:`route_maxbas`, but each cell's MAXBAS is rescaled by its flow path length,
        so cells farther from the outlet are attenuated more. `results.quz` is modified in
        place.

        Args:
            run: The validated inputs, whose `flow_path_length` supplies the raster.
            results: The results to route. Mutated in place.

        Raises:
            ValueError: The run carries no flow-path-length raster.
        """
        if run.flow_path_length is None:
            raise ValueError(
                "this routing scales MAXBAS by flow path length, but the run carries no "
                "flow-path-length raster; call read_flow_path_length first"
            )
        MAXBAS = np.nanmax(run.parameter_cube[:, :, -1])
        # `read_flow_path_length` already masks this raster's own no-data cells to NaN via
        # pyramids, so no sentinel comparison is needed here -- and the one that used to sit
        # here compared against the *accumulation* raster's sentinel, which is a different
        # raster and need not share a no-data value.

        MaxFPL = np.nanmax(run.flow_path_length)
        MinFPL = np.nanmin(run.flow_path_length)
        # resize_fun = lambda x: np.round(((((x - min_dist)/(max_dist - min_dist))*(1*maxbas - 1)) + 1), 0)
        resize_fun = lambda g: (
            (((g - MinFPL) / (MaxFPL - MinFPL)) * (1 * MAXBAS - 1)) + 1
        )

        NormalizedFPL = resize_fun(run.flow_path_length)
        quz = results.quz

        for x in range(run.flow_network.rows):
            for y in range(run.flow_network.cols):
                if not np.isnan(run.flow_path_length[x, y]):
                    quz[x, y, :] = routing.triangular_routing_2(
                        quz[x, y, :], NormalizedFPL[x, y]
                    )
