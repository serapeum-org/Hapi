"""Tests for the per-cell output fields the triangular (MAXBAS) path produces.

Only ``DistRRM.SpatialRouting`` (the Muskingum path) used to set ``Qtot`` /
``quz_routed`` / ``qlz_translated``, so after ``Run.runFW1`` they stayed ``None`` and every
discharge option of ``save_results`` / ``plot_distributed_results`` raised
``TypeError: 'NoneType' object is not subscriptable``. ``Wrapper._set_maxbas_output_fields``
now fills them; these tests pin both the values and the MAXBAS-specific semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from hapi.catchment import Catchment
from hapi.inputs import MeteoInputs
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


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
    coello.read_flow_acc(coello_acc_path)
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    Run.runFW1(coello)
    return coello


def test_fw1_sets_the_per_cell_output_fields(coello_fw1: Catchment):
    """Test that runFW1 leaves Qtot and the routed/translated fields populated.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.

    Test scenario:
        These three fields back the discharge options of `save_results` and
        `plot_distributed_results`. Before the fix only the Muskingum path set
        them, so they were `None` here and every discharge option raised.
    """
    shape = coello_fw1.quz.shape
    for name in ("Qtot", "quz_routed", "qlz_translated"):
        field = getattr(coello_fw1, name)
        assert field is not None, f"{name} must be set after runFW1"
        assert field.shape == shape, f"{name} must be a per-cell, per-timestep field"


def test_fw1_qtot_is_the_sum_of_the_two_zones(coello_fw1: Catchment):
    """Test that Qtot equals the routed upper zone plus the lower zone.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.

    Test scenario:
        MAXBAS routes quz in place and does not translate qlz, so the per-cell
        total is simply their sum — the triangular-routing analogue of the
        Muskingum path's `qlz_translated + quz_routed`.
    """
    np.testing.assert_allclose(
        coello_fw1.Qtot,
        coello_fw1.qlz + coello_fw1.quz,
        rtol=1e-6,
        err_msg="Qtot must be qlz + quz on the MAXBAS path",
    )


def test_fw1_qtot_summed_over_the_domain_reproduces_qout(coello_fw1: Catchment):
    """Test the invariant that ties Qtot to the outlet hydrograph.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.

    Test scenario:
        This is what makes the field meaningful rather than merely non-None.
        MAXBAS sends every cell straight to the outlet, so summing Qtot over the
        domain must give back the `qout` that `Wrapper.FW1` computed
        independently as `nansum(qlz) + nansum(quz)`. `qout` drops the last
        timestep, so compare against the matching slice.
    """
    summed = np.array(
        [np.nansum(coello_fw1.Qtot[:, :, i]) for i in range(coello_fw1.Qtot.shape[2])]
    )
    np.testing.assert_allclose(
        summed[:-1],
        coello_fw1.qout,
        rtol=1e-6,
        err_msg="nansum(Qtot) over the domain must reproduce qout",
    )


def test_extract_discharge_rejects_the_outlet_cell_shortcut_after_fw1(
    coello_fw1: Catchment, coello_acc_path: str, coello_gauges_table: str
):
    """Test that the Muskingum-only outlet-cell shortcut raises on a MAXBAS run.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.
        coello_acc_path: Flow-accumulation raster, to map the gauges onto the grid.
        coello_gauges_table: Gauge table path.

    Test scenario:
        `extract_discharge(frame_work_1=False)` reads `Qtot` at the outlet cell.
        That is correct for Muskingum, where the field accumulates downstream, but
        wrong for MAXBAS, where one cell holds only its own contribution. Before
        Qtot was populated this crashed with a bare TypeError; now that it holds
        real numbers the wrong answer would be silent, so it must raise instead.
    """
    coello_fw1.read_gauge_table(coello_gauges_table, coello_acc_path)
    with pytest.raises(ValueError, match="MAXBAS"):
        coello_fw1.extract_discharge(calculate_metrics=False)


def test_save_results_distributed_discharge_after_fw1(
    coello_fw1: Catchment, coello_acc_path: str, tmp_path
):
    """Test that the discharge results can now be written as rasters after runFW1.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.
        coello_acc_path: Flow-accumulation raster used as the grid template.
        tmp_path: Destination directory.

    Test scenario:
        `result=1` (total discharge) is the option that used to raise on this
        path. Writes it and reads the first raster back to confirm it carries the
        matching Qtot slice.
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

    start_i = np.where(coello_fw1.date_index == np.datetime64("2009-01-01"))[0][0]
    np.testing.assert_allclose(
        Dataset.read_file(str(written[0])).read_array(band=0),
        coello_fw1.Qtot[:, :, start_i],
        rtol=1e-5,
        err_msg="the first raster must hold the first Qtot step",
    )


@pytest.mark.plot
@pytest.mark.parametrize("option", [1, 2, 3])
def test_plot_discharge_options_after_fw1(coello_fw1: Catchment, option: int):
    """Test that the three discharge animation options work after runFW1.

    Args:
        coello_fw1: Coello catchment with a completed MAXBAS run.
        option: 1 total discharge, 2 upper zone, 3 ground water.

    Test scenario:
        Options 1-3 read Qtot / quz_routed / qlz_translated respectively and all
        three raised TypeError on this path before the fix.
    """
    import matplotlib.animation

    anim = coello_fw1.plot_distributed_results(
        "2009-01-01", "2009-01-05", option=option
    )
    assert isinstance(anim, matplotlib.animation.FuncAnimation)
