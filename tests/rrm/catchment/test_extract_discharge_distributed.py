"""Tests for the distributed (multi-gauge) branch of extract_discharge."""

import numpy as np
import pytest
from pandas import DataFrame

from hapi.catchment import Catchment
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
        Catchment: Model with `Qtot` populated by the spatial routing.
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
    coello.read_rainfall(coello_prec_path, **kwargs)
    coello.read_temperature(coello_temp_path, **kwargs)
    coello.read_et(coello_evap_path, **kwargs)
    coello.read_flow_acc(coello_acc_path)
    coello.read_flow_dir(coello_fd_path)
    coello.read_parameters(coello_dist_parameters_muskingum, False)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.read_gauge_table(coello_gauges_table, coello_acc_path)
    coello.read_discharge_gauges(coello_gauges_path, column="id", fmt="%Y-%m-%d")
    Run.RunHapi(coello)
    return coello


def test_extract_discharge_distributed_metrics(coello_muskingum_run: Catchment):
    """The distributed branch computes all seven metrics for every gauge.

    Test scenario:
        After a Muskingum run, extract_discharge with the default
        frame_work_1=False walks the gauge table, extracts Qsim per gauge
        from Qtot, and fills the Metrics frame (RMSE, NSE, NSEhf, KGE, WB,
        Pearson-CC, R2) with finite numbers.
    """
    coello = coello_muskingum_run
    coello.extract_discharge(calculate_metrics=True)

    assert isinstance(coello.Metrics, DataFrame), (
        f"Metrics should be a DataFrame, got {type(coello.Metrics)}"
    )
    expected_rows = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
    assert list(coello.Metrics.index) == expected_rows, (
        f"Metrics rows mismatch: {list(coello.Metrics.index)}"
    )
    n_gauges = len(coello.GaugesTable)
    assert coello.Metrics.shape[1] == n_gauges, (
        f"Expected one metrics column per gauge ({n_gauges}), "
        f"got {coello.Metrics.shape[1]}"
    )
    assert np.isfinite(coello.Metrics.to_numpy(dtype=float)).all(), (
        "All metric values should be finite"
    )
    assert coello.Qsim.shape == (len(coello.Index), n_gauges), (
        f"Qsim shape mismatch: {coello.Qsim.shape}"
    )


def test_extract_discharge_distributed_without_metrics(
    coello_muskingum_run: Catchment,
):
    """The distributed branch fills Qsim without touching Metrics.

    Test scenario:
        calculate_metrics=False extracts the simulated hydrographs only;
        Qsim has one column per gauge with finite values.
    """
    coello = coello_muskingum_run
    coello.extract_discharge(calculate_metrics=False)
    assert np.isfinite(coello.Qsim.to_numpy(dtype=float)).all(), (
        "Qsim should contain finite discharge values"
    )
