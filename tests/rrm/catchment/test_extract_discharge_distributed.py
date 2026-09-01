"""Tests for the distributed (multi-gauge) branch of extract_discharge."""

import numpy as np
import pytest
from pandas import DataFrame

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


@pytest.fixture(scope="module")
def coello_muskingum_run(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_muskingum: str,
    coello_cat_area: int,
    coello_initial_cond: list,
    coello_gauges_table: str,
    coello_gauges_path: str,
) -> Catchment:
    """Distributed Coello catchment with a completed Muskingum run.

    Returns:
        Catchment: Model with `q_total` populated by the spatial routing.
    """
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
    coello.flow_network = FlowNetwork.from_rasters(coello_acc_path, coello_fd_path)
    coello.read_parameters(coello_dist_parameters_muskingum, False)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.read_gauge_table(coello_gauges_table, coello_acc_path)
    coello.read_discharge_gauges(coello_gauges_path, column="id", fmt="%Y-%m-%d")
    Run.run_distributed(coello)
    return coello


def test_extract_discharge_distributed_metrics(coello_muskingum_run: Catchment):
    """The distributed branch computes all seven metrics for every gauge.

    Test scenario:
        After a Muskingum run, extract_discharge with the default
        Muskingum-routed results walk the gauge table, extracting Qsim per gauge
        from q_total, and fills the metrics frame (RMSE, NSE, NSEhf, KGE, WB,
        Pearson-CC, R2) with finite numbers.
    """
    coello = coello_muskingum_run
    coello.extract_discharge(calculate_metrics=True)

    assert isinstance(coello.metrics, DataFrame), (
        f"metrics should be a DataFrame, got {type(coello.metrics)}"
    )
    expected_rows = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
    assert list(coello.metrics.index) == expected_rows, (
        f"metrics rows mismatch: {list(coello.metrics.index)}"
    )
    n_gauges = len(coello.GaugesTable)
    assert coello.metrics.shape[1] == n_gauges, (
        f"Expected one metrics column per gauge ({n_gauges}), "
        f"got {coello.metrics.shape[1]}"
    )
    assert np.isfinite(coello.metrics.to_numpy(dtype=float)).all(), (
        "All metric values should be finite"
    )
    assert coello.Qsim.shape == (len(coello.date_index), n_gauges), (
        f"Qsim shape mismatch: {coello.Qsim.shape}"
    )


def test_extract_discharge_distributed_without_metrics(
    coello_muskingum_run: Catchment,
):
    """The distributed branch fills Qsim without touching metrics.

    Test scenario:
        calculate_metrics=False extracts the simulated hydrographs only;
        Qsim has one column per gauge with finite values.
    """
    coello = coello_muskingum_run
    coello.extract_discharge(calculate_metrics=False)
    assert np.isfinite(coello.Qsim.to_numpy(dtype=float)).all(), (
        "Qsim should contain finite discharge values"
    )


@pytest.mark.plot
def test_plot_hydrograph_logs_the_metrics_it_computed(coello_muskingum_run: Catchment):
    """Test that plotting a gauge hydrograph after metrics exist reports them.

    Test scenario:
        `plot_hydrograph` reads `self.metrics` to log the seven scores next to the figure, a
        branch that only runs once `extract_discharge` has filled the frame. It looks the
        scores up by the gauge's id while addressing the gauge table by position, so the plot
        must find both without raising.
    """
    coello = coello_muskingum_run
    coello.extract_discharge(calculate_metrics=True)

    fig, ax = coello.plot_hydrograph("2009-01-01", "2009-01-09", 0)

    assert fig is not None, "plot_hydrograph must return a figure"
    assert ax is not None, "plot_hydrograph must return an axis"
    assert not coello.metrics.empty, "the metrics frame should be populated"
