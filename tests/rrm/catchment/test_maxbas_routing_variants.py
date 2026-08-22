"""Tests for the two triangular-routing variants and the lumped routing branches.

`DistributedRRM.DistMaxbas2` is a public entry point that nothing inside the package calls -- it
rescales each cell's MAXBAS by its flow path length -- so it is reached only from a test or a
downstream user. `Wrapper.Lumped` picks its routing call by whether `maxbas` is set, a branch
the existing lumped tests never took.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.routing import Routing
from hapi.rrm.distrrm import DistributedRRM
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


@pytest.fixture(scope="module")
def coello_before_routing(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_maxbas: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> Catchment:
    """Build a distributed catchment with the per-cell model run but no routing applied.

    Returns:
        Catchment: Model whose `quz`/`qlz` are populated and whose `flow_path_length_arr`
            increases away from the top-left corner.
    """
    coello = Catchment(
        "coello-maxbas",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    coello.meteo = MeteoInputs.from_rasters(
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        start=coello_start_date,
        end=coello_end_date,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    coello.flow_network = FlowNetwork.from_rasters(coello_acc_path, coello_fd_path)
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    acc = coello.flow_network.flow_acc_arr
    rows, cols = coello.flow_network.rows, coello.flow_network.cols
    distance = np.add.outer(np.arange(rows), np.arange(cols)).astype("float64") * 1000.0
    coello.flow_path_length_arr = np.where(np.isnan(acc), np.nan, distance)
    DistributedRRM.run_lumped_model(coello)
    return coello


@pytest.fixture(scope="module")
def maxbas_parameters_path(lumped_parameters_path: str, tmp_path_factory) -> str:
    """Provide an eleven-value lumped parameter file, the MAXBAS configuration.

    The bundled lumped set is the twelve-value Muskingum one; the MAXBAS configuration
    replaces the two Muskingum parameters with a single MAXBAS width.

    Returns:
        str: Path to the written parameter file.
    """
    values = [
        line for line in Path(lumped_parameters_path).read_text().splitlines() if line
    ]
    path = tmp_path_factory.mktemp("maxbas") / "parameters.txt"
    path.write_text(os.linesep.join(values[:10] + ["maxbas,3.0"]) + os.linesep)
    return str(path)


class TestDistMaxbas2:
    """Tests for `DistributedRRM.DistMaxbas2`."""

    def test_routes_every_cell_inside_the_flow_path_length_mask(
        self, coello_before_routing: Catchment
    ):
        """Test that the flow-path-length variant attenuates the in-catchment cells.

        Test scenario:
            The routing rescales each cell's MAXBAS between 1 and the grid maximum by its
            distance along the flow path, then convolves `quz` in place. Cells inside the
            mask must change and stay finite; the total volume is redistributed in time
            rather than created, so the sum over the record must not grow.
        """
        model = coello_before_routing
        before = model.quz.copy()
        inside = ~np.isnan(model.flow_path_length_arr)

        DistributedRRM.DistMaxbas2(model)

        assert not np.array_equal(model.quz[inside], before[inside]), (
            "the routing must alter the upper-zone discharge of the masked cells"
        )
        assert np.isfinite(model.quz[inside]).all(), (
            "routed discharge must stay finite inside the catchment"
        )
        assert np.nansum(model.quz[inside]) <= np.nansum(before[inside]) + 1e-6, (
            "triangular routing redistributes volume in time, it must not add any"
        )

    def test_leaves_cells_outside_the_mask_untouched(
        self, coello_before_routing: Catchment
    ):
        """Test that NaN flow-path-length cells are skipped rather than routed.

        Test scenario:
            The mask is what keeps the routing off cells outside the catchment. Those cells
            must come back bit-identical, including their NaNs.
        """
        model = coello_before_routing
        before = model.quz.copy()
        outside = np.isnan(model.flow_path_length_arr)

        DistributedRRM.DistMaxbas2(model)

        np.testing.assert_array_equal(
            model.quz[outside],
            before[outside],
            err_msg="cells outside the flow-path-length mask must not be routed",
        )


class TestLumpedRouting:
    """Tests for the routing branches of `Wrapper.Lumped` reached through `Run.runLumped`."""

    def test_maxbas_routing_uses_the_last_parameter(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        maxbas_parameters_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
    ):
        """Test that a lumped run with `maxbas` routes on the trailing parameter alone.

        Test scenario:
            The lumped wrapper has two routing calls: the MAXBAS one passes only the last
            parameter, the Muskingum one passes three. Which is used is decided by the
            `maxbas` flag `read_parameters` stored, and only the Muskingum branch had
            coverage.
        """
        model = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        model.read_lumped_inputs(lumped_meteo_data_path)
        model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        model.read_parameters(maxbas_parameters_path, False, maxbas=True)

        Run.runLumped(model, Route=1, routing_fn=Routing.triangular_routing_1)

        assert model.Qsim is not None, "a routed lumped run must produce Qsim"
        assert np.isfinite(np.asarray(model.Qsim, dtype=float)).all(), (
            "the routed lumped hydrograph must be finite"
        )

    def test_routing_without_a_function_is_rejected(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        lumped_parameters_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
    ):
        """Test that asking for routing without supplying a routing function raises.

        Test scenario:
            `Route != 0` selects a routing call that would be made with `None`. Catching it
            at the entry point turns an obscure failure deep inside the wrapper into a
            message naming the argument that is missing.
        """
        model = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        model.read_lumped_inputs(lumped_meteo_data_path)
        model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        model.read_parameters(lumped_parameters_path, False, maxbas=False)

        with pytest.raises(ValueError, match="routing_fn"):
            Run.runLumped(model, Route=1)
