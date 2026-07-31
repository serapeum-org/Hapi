"""Smoke tests for the cleopatra-backed animation surface of Catchment."""

import numpy as np
import pytest

from hapi.catchment import Catchment
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


@pytest.fixture(scope="module")
def coello_animated(
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
    """Distributed Coello catchment with a completed FW1 run."""
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
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.read_gauge_table(coello_gauges_table, coello_acc_path)
    Run.runFW1(coello)
    return coello


@pytest.mark.plot
def test_plot_precipitation_with_gauges(coello_animated: Catchment):
    """Animating a meteo input with gauge points returns a FuncAnimation."""
    import matplotlib.animation

    before = coello_animated.Prec.copy()
    anim = coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=9, gauges=True, interval=100
    )
    assert isinstance(anim, matplotlib.animation.FuncAnimation)
    # plotting must not mutate the model arrays stored on the instance
    assert np.array_equal(before, coello_animated.Prec, equal_nan=True)


@pytest.mark.plot
def test_plot_state_variable(coello_animated: Catchment):
    """Animating a state variable after a model run works."""
    import matplotlib.animation

    before = coello_animated.state_variables.copy()
    anim = coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=5
    )
    assert isinstance(anim, matplotlib.animation.FuncAnimation)
    assert np.array_equal(
        before, coello_animated.state_variables, equal_nan=True
    )


@pytest.mark.plot
def test_plot_title_override(coello_animated: Catchment):
    """An explicit title= kwarg overrides the option default."""
    anim = coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=9, title="Custom"
    )
    assert anim is not None


@pytest.mark.plot
def test_save_animation_gif(coello_animated: Catchment, tmp_path):
    """save_animation writes a non-empty gif after plotting."""
    coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=9
    )
    out = tmp_path / "anim.gif"
    coello_animated.save_animation(str(out), fps=2)
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.plot
def test_save_animation_before_plot_raises():
    """save_animation without a prior plot raises a clear error."""
    coello = Catchment("bare", "2009-01-01", "2009-01-10")
    with pytest.raises(ValueError, match="plot_distributed_results"):
        coello.save_animation("never.gif")
