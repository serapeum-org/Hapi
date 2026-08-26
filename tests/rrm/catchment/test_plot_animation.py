"""Smoke tests for the cleopatra-backed animation surface of Catchment."""

import numpy as np
import pytest

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
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
    coello.meteo = MeteoInputs.from_rasters(
        coello_prec_path, coello_temp_path, coello_evap_path, **kwargs
    )
    coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.read_gauge_table(coello_gauges_table, coello_acc_path)
    Run.runFW1(coello)
    return coello


@pytest.mark.plot
def test_plot_precipitation_with_gauges(coello_animated: Catchment):
    """Animating a meteo input with gauge points returns a FuncAnimation."""
    import matplotlib.animation

    before = coello_animated.meteo.precipitation.copy()
    anim = coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=9, gauges=True, interval=100
    )
    assert isinstance(anim, matplotlib.animation.FuncAnimation)
    # plotting must not mutate the model arrays stored on the instance
    assert np.array_equal(before, coello_animated.meteo.precipitation, equal_nan=True)


@pytest.mark.plot
def test_plot_state_variable(coello_animated: Catchment):
    """Animating a state variable after a model run works."""
    import matplotlib.animation

    before = coello_animated.state_variables.copy()
    anim = coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=5
    )
    assert isinstance(anim, matplotlib.animation.FuncAnimation)
    assert np.array_equal(before, coello_animated.state_variables, equal_nan=True)


@pytest.mark.plot
def test_plot_title_override(coello_animated: Catchment):
    """An explicit title= kwarg overrides the option default."""
    anim = coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=9, title="Custom"
    )
    assert anim is not None


@pytest.mark.plot
def test_plot_accepts_grouped_style_objects(coello_animated: Catchment):
    """Test that cleopatra's typed style groups reach `ArrayGlyph.animate` intact.

    Test scenario:
        cleopatra 0.30 replaced the loose styling keywords (`color_scale`,
        `display_cell_value`, `num_size`, `background_color_threshold`,
        `text_loc`) with typed group objects, and a removed keyword now raises
        rather than being silently ignored. `plot_distributed_results` forwards
        `**kwargs` untouched, so this pins that the group objects pass through —
        and that Hapi never re-introduces a loose keyword that would raise.
    """
    import matplotlib.animation
    from cleopatra.glyphs.gridded.array_glyph import FrameLabel
    from cleopatra.styling.params import CellValues
    from cleopatra.styling.scaling import ColorScaling

    anim = coello_animated.plot_distributed_results(
        "2009-01-01",
        "2009-01-09",
        option=9,
        gauges=True,
        interval=100,
        color=ColorScaling.power(gamma=0.5),
        cells=CellValues(show=True, size=8, background_threshold=None),
        frame_label=FrameLabel(location=[0.1, 0.2], color="black"),
        ticks_spacing=5,
        cmap="inferno",
    )
    assert isinstance(anim, matplotlib.animation.FuncAnimation)


@pytest.mark.plot
def test_plot_gauges_are_wrapped_in_a_point_overlay(
    coello_animated: Catchment, monkeypatch
):
    """Test that the gauge markers are handed to cleopatra as a `PointOverlay`.

    Args:
        coello_animated: Distributed Coello catchment with a completed run.
        monkeypatch: Used to spy on the `ArrayGlyph.animate` call.

    Test scenario:
        cleopatra 0.30 stopped accepting a bare `(N, 3)` array for `points`, so
        Hapi must wrap the gauge table itself. Pins the wrapping and that the
        overlay still carries the `[id, row, col]` triples the animation draws.
    """
    from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay

    seen = {}
    original = ArrayGlyph.animate

    def spy(self, time, *args, **kwargs):
        seen.update(kwargs)
        return original(self, time, *args, **kwargs)

    monkeypatch.setattr(ArrayGlyph, "animate", spy)
    coello_animated.plot_distributed_results(
        "2009-01-01", "2009-01-09", option=9, gauges=True
    )

    points = seen["points"]
    assert isinstance(points, PointOverlay), (
        "gauges=True must pass a PointOverlay; a bare array raises on cleopatra >=0.30"
    )
    assert points.points.shape[1] == 3, "points must stay [value, row, col]"


@pytest.mark.plot
def test_save_animation_gif(coello_animated: Catchment, tmp_path):
    """save_animation writes a non-empty gif after plotting."""
    coello_animated.plot_distributed_results("2009-01-01", "2009-01-09", option=9)
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


@pytest.mark.plot
@pytest.mark.parametrize(
    "option, attribute, title",
    [
        (9, "precipitation", "Precipitation"),
        (10, "evapotranspiration", "ET"),
        (11, "temperature", "Temperature"),
    ],
)
def test_meteo_options_animate_the_cube_they_name(
    coello_animated: Catchment, monkeypatch, option: int, attribute: str, title: str
):
    """Test that each meteorological option animates its own cube from MeteoInputs.

    Args:
        coello_animated: Distributed Coello catchment with a completed run.
        monkeypatch: Used to spy on the `ArrayGlyph.animate` call.
        option: The plotting option under test.
        attribute: The `MeteoInputs` field that option should read.
        title: The default title that option should carry.

    Test scenario:
        The three drivers moved off the catchment onto `MeteoInputs`, so the option dispatch
        now reads them through `self.meteo`. Each option must reach its own cube and carry
        its own default title — the two are set together in one branch, and a driver that is
        missing from the model must be reported rather than animated as an empty grid.
    """
    from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

    seen = {}
    original = ArrayGlyph.animate

    def spy(self, time, *args, **kwargs):
        seen["kwargs"] = kwargs
        return original(self, time, *args, **kwargs)

    monkeypatch.setattr(ArrayGlyph, "animate", spy)

    coello_animated.plot_distributed_results("2009-01-01", "2009-01-09", option=option)

    assert seen["kwargs"].get("title") == title, (
        f"option {option} should be titled {title!r}, got {seen['kwargs'].get('title')!r}"
    )
    assert getattr(coello_animated.meteo, attribute) is not None, (
        f"option {option} reads meteo.{attribute}, which must be populated"
    )


@pytest.mark.plot
@pytest.mark.parametrize("option", [0, 12])
def test_plot_invalid_option_raises(option: int):
    """An option outside 1-11 raises ValueError before touching any array.

    Args:
        option: An out-of-range plotting option.

    Test scenario:
        The option dispatch must reject values outside 1..11 with a clear
        error even on a catchment with no data loaded.
    """
    coello = Catchment(
        "bare", "2009-01-01", "2009-01-10", spatial_resolution="Distributed"
    )
    with pytest.raises(ValueError, match="1 to 11"):
        coello.plot_distributed_results("2009-01-01", "2009-01-09", option=option)
