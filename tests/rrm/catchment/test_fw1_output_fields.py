"""Tests for the per-cell output fields the triangular (MAXBAS) path produces.

Only ``DistRRM.route_muskingum`` (the Muskingum path) used to set ``q_total`` /
``quz_routed`` / ``qlz_translated``, so after ``Run.run_maxbas`` they stayed ``None`` and every
discharge option of ``save_results`` / ``plot_distributed_results`` raised
``TypeError: 'NoneType' object is not subscriptable``. ``Wrapper._set_maxbas_output_fields``
now fills them; these tests pin both the values and the MAXBAS-specific semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.routing import Routing
from hapi.rrm.distrrm import DistributedRRM
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run
from hapi.runs import DistributedRun

DATE_REGEX = r"\d{4}.\d{2}.\d{2}"


@pytest.fixture(scope="module")
def coello_fw1(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_dist_parameters_maxbas: str,
    coello_cat_area: int,
    coello_initial_cond: list,
    coello_gauges_table: str,
) -> Catchment:
    """Distributed Coello catchment with a completed triangular (MAXBAS) run."""
    coello = Catchment(
        "coello",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    kwargs = dict(
        start=coello_start_date,
        end=coello_end_date,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    coello.meteo = MeteoInputs.from_rasters(
        coello_prec_path, coello_temp_path, coello_evap_path, **kwargs
    )
    coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    # read here rather than inside a test: the fixture is module-scoped, so a test that
    # mutated it would leak into whichever test ran next.
    coello.read_gauge_table(coello_gauges_table, coello_acc_path)
    Run.run_maxbas(coello)
    return coello


@pytest.fixture(scope="module")
def coello_unrouted(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_dist_parameters_maxbas: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> Catchment:
    """The same catchment with the per-cell model run but no routing applied.

    Returns:
        Catchment: Model whose `quz` / `qlz` are the conceptual model's raw output, so the
            triangular routing can be recomputed against them independently.
    """
    coello = Catchment(
        "coello-unrouted",
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
        regex_string=DATE_REGEX,
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.results = DistributedRRM.run_lumped_model(
        DistributedRun.from_model(coello, needs_flow_direction=False)
    )
    return coello


def test_fw1_sets_the_per_cell_output_fields(coello_fw1: Catchment):
    """Test that run_maxbas leaves q_total and the routed/translated fields populated.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.

    Test scenario:
        These three fields back the discharge options of `save_results` and
        `plot_distributed_results`. Before the fix only the Muskingum path set
        them, so they were `None` here and every discharge option raised.
    """
    shape = coello_fw1.results.quz.shape
    for name in ("q_total", "quz_routed", "qlz_translated"):
        field = getattr(coello_fw1.results, name)
        assert field is not None, f"{name} must be set after run_maxbas"
        assert field.shape == shape, f"{name} must be a per-cell, per-timestep field"


def test_fw1_qtot_matches_an_independent_triangular_convolution(
    coello_fw1: Catchment, coello_unrouted: Catchment
):
    """Test `q_total` against the routing recomputed outside the wrapper.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.
        coello_unrouted: The same catchment with only the per-cell model run.

    Test scenario:
        Asserting `q_total == qlz + quz` restates the assignment that produced it, so a wrong
        MAXBAS convolution passes. Recompute the routing here instead -- each in-domain cell
        convolved with `triangular_routing_1` against its own MAXBAS parameter -- and compare
        the whole field. That is the only assertion that can tell a correct kernel from a
        wrong one.
    """
    expected_quz = coello_unrouted.results.quz.copy()
    maxbas = coello_fw1.parameters.values[:, :, -1]
    acc = coello_fw1.flow_network.flow_acc_arr
    for x in range(coello_fw1.flow_network.rows):
        for y in range(coello_fw1.flow_network.cols):
            if not np.isnan(acc[x, y]):
                expected_quz[x, y, :] = Routing.triangular_routing_1(
                    expected_quz[x, y, :], maxbas[x, y]
                )

    np.testing.assert_allclose(
        coello_fw1.results.q_total,
        coello_unrouted.results.qlz + expected_quz,
        rtol=1e-6,
        err_msg="q_total must be the lower zone plus the independently routed upper zone",
    )
    assert not np.allclose(expected_quz, coello_unrouted.results.quz), (
        "the routing must change quz, otherwise this comparison proves nothing"
    )


def test_fw1_routes_the_upper_zone_but_leaves_the_lower_zone_alone(
    coello_fw1: Catchment, coello_unrouted: Catchment
):
    """Test that MAXBAS attenuates `quz` and passes `qlz` through untouched.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.
        coello_unrouted: The same catchment with only the per-cell model run.

    Test scenario:
        `route_maxbas` convolves only the upper zone. Pinning both halves separates a routing
        that ran from one that silently did nothing, and catches a change that started
        routing the lower zone too.
    """
    np.testing.assert_allclose(
        coello_fw1.results.qlz,
        coello_unrouted.results.qlz,
        rtol=1e-9,
        err_msg="the lower zone is not routed by the triangular path",
    )
    assert not np.allclose(coello_fw1.results.quz, coello_unrouted.results.quz), (
        "the upper zone must be attenuated by the triangular routing"
    )


def test_extract_discharge_takes_the_basin_wide_sum_after_fw1(coello_fw1: Catchment):
    """Test that MAXBAS results select the basin-wide hydrograph without being told.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.

    Test scenario:
        Reading `q_total` at the outlet cell is right for Muskingum, where the field
        accumulates downstream, and wrong for MAXBAS, where a cell holds only its own
        contribution. This used to be the caller's job via a `frame_work_1` flag, with a
        `ValueError` when they set it wrong. The routing is a property of the arrays, so
        `extract_discharge` reads it off the results and takes the sum the run computed.
    """
    coello_fw1.extract_discharge(calculate_metrics=False)

    assert coello_fw1.Qsim.shape[1] == 1, (
        f"the basin-wide path yields one hydrograph, got {coello_fw1.Qsim.shape[1]} columns"
    )
    np.testing.assert_allclose(
        coello_fw1.Qsim.iloc[:, 0].to_numpy(dtype="float64"),
        np.asarray(coello_fw1.results.qout, dtype="float64"),
        err_msg="the extracted hydrograph must be the qout the MAXBAS run summed",
    )


def test_save_results_distributed_discharge_after_fw1(
    coello_fw1: Catchment, coello_acc_path: str, tmp_path
):
    """Test that the discharge results can now be written as rasters after run_maxbas.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.
        coello_acc_path: Flow-accumulation raster used as the grid template.
        tmp_path: Destination directory.

    Test scenario:
        `result=1` (total discharge) is the option that used to raise on this
        path. Writes it and reads the first raster back to confirm it carries the
        matching q_total slice.
    """
    out = tmp_path / "q"
    out.mkdir()
    coello_fw1.save_results(
        flow_acc_path=coello_acc_path,
        result=1,
        start="2009-01-01",
        end="2009-01-04",
        path=f"{out}/",
    )

    written = sorted(out.glob("*.tif"))
    assert len(written) == 4, f"expected one raster per date, got {len(written)}"

    start_i = np.where(coello_fw1.period.date_index == np.datetime64("2009-01-01"))[0][
        0
    ]
    np.testing.assert_allclose(
        Dataset.read_file(str(written[0])).read_array(band=0),
        coello_fw1.results.q_total[:, :, start_i],
        rtol=1e-5,
        err_msg="the first raster must hold the first q_total step",
    )


@pytest.mark.plot
@pytest.mark.parametrize("option", [1, 2, 3])
def test_plot_discharge_options_after_fw1(coello_fw1: Catchment, option: int):
    """Test that the three discharge animation options work after run_maxbas.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.
        option: 1 total discharge, 2 upper zone, 3 ground water.

    Test scenario:
        Options 1-3 read q_total / quz_routed / qlz_translated respectively and all
        three raised TypeError on this path before the fix.
    """
    import matplotlib.animation

    anim = coello_fw1.plot_distributed_results(
        "2009-01-01", "2009-01-05", option=option
    )
    assert isinstance(anim, matplotlib.animation.FuncAnimation)
