"""Tests for the builder/finished split: narrowing a catchment into a validated run.

`Catchment` is a builder -- its inputs are `X | None` until the matching `read_*` call has run,
and that is honest. The run layer needs the opposite: a catchment that is finished. Conflating
the two meant the engines dereferenced `X | None` on every line, and meant "has this been
validated?" was answered by remembering which entry point you came through -- which is how
`Calibration`, going straight to `Wrapper`, skipped every check `Run` performed.

`DistributedRun.from_model` / `LumpedRun.from_model` are that seam. These tests pin the two
properties it buys: constructing the run *is* the validation, and there is no way to reach an
engine without doing it.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs, RiverGeometry
from hapi.rrm.distrrm import DistributedRRM
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.runs import DistributedRun, LumpedRun
from hapi.wrapper import Wrapper

DATE_REGEX = r"\d{4}.\d{2}.\d{2}"


@pytest.fixture
def built(
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
) -> Catchment:
    """A distributed Coello catchment with every input read and no run behind it."""
    model = Catchment(
        "coello",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    model.meteo = MeteoInputs.from_rasters(
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        start=coello_start_date,
        end=coello_end_date,
        regex_string=DATE_REGEX,
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    model.flow_network = FlowNetwork.from_rasters(coello_acc_path, coello_fd_path)
    model.read_parameters(coello_dist_parameters_muskingum, False)
    model.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    return model


class TestNarrowingIsTheValidation:
    """`from_model` is where the optionality is resolved and the cross-checks happen."""

    def test_a_finished_model_narrows(self, built: Catchment):
        """Test that a fully built catchment produces a run with non-optional inputs.

        Test scenario:
            The point of the split: past this seam nothing is `| None`, so the engines index
            real arrays rather than unions and mypy can check them.
        """
        run = DistributedRun.from_model(built)

        for field in ("period", "meteo", "flow_network", "parameters", "model_setup"):
            assert getattr(run, field) is not None, (
                f"{field} must be settled on the run"
            )
        assert run.meteo is built.meteo, (
            "the run carries the model's own inputs, not copies"
        )

    @pytest.mark.parametrize(
        "missing, expected",
        [
            ("meteo", "needs meteo"),
            ("flow_network", "needs flow_network"),
            ("parameters", "needs parameters"),
            ("model_setup", "needs model_setup"),
        ],
    )
    def test_an_unread_input_is_named(
        self, built: Catchment, missing: str, expected: str
    ):
        """Test that a missing input is reported by name, with the reader that supplies it.

        Test scenario:
            A half-built catchment used to reach the engines and fail on `None` several frames
            in, naming an attribute of an array rather than the reader nobody called.

        Args:
            missing: The input to clear.
            expected: Substring the error must carry.
        """
        setattr(built, missing, None)

        with pytest.raises(ValueError, match=expected):
            DistributedRun.from_model(built)

    def test_the_run_is_frozen(self, built: Catchment):
        """Test that a narrowed run cannot be edited after it is checked.

        Test scenario:
            The checks happen once, at construction. A mutable run would let a caller swap an
            input in afterwards and reach the engines with something never validated -- the
            exact hole this replaces.
        """
        run = DistributedRun.from_model(built)

        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            run.meteo = None

    def test_maxbas_does_not_require_a_flow_direction_raster(self, built: Catchment):
        """Test that the optional input is only required by the paths that read it.

        Test scenario:
            MAXBAS sends every cell straight to the outlet, so it never reads the direction
            raster. Requiring it everywhere would refuse a legitimate run.
        """
        built.flow_network = FlowNetwork(
            built.flow_network.flow_acc_arr,
            no_data_value=built.flow_network.no_data_value,
            cell_size=built.flow_network.cell_size,
            px_area=built.flow_network.px_area,
        )

        run = DistributedRun.from_model(built, needs_flow_direction=False)

        assert run.flow_network.flow_dir_arr is None, "the raster is genuinely absent"
        with pytest.raises(ValueError, match="flow-direction"):
            DistributedRun.from_model(built, needs_flow_direction=True)

    def test_a_skip_without_geometry_is_refused(self, built: Catchment):
        """Test that asking to skip river cells with nothing to identify them raises.

        Test scenario:
            The skip reads `bankfull_depth`. Without the geometry that used to be a
            `TypeError` on `None` partway through the routing loop; the run type refuses it
            before any cell is touched.
        """
        with pytest.raises(ValueError, match="read_river_geometry"):
            DistributedRun.from_model(built, skip_hydraulic_cells=True)

    def test_geometry_off_the_catchment_grid_is_refused(self, built: Catchment):
        """Test that geometry on a different grid than the catchment raises.

        Test scenario:
            `RiverGeometry` settles that the five rasters agree with *each other*; this is the
            other half -- that they agree with the flow network. A cell index would otherwise
            mean a different place in each.
        """
        wrong = np.ones((built.flow_network.rows + 1, built.flow_network.cols))
        built.river_geometry = RiverGeometry(wrong, wrong, wrong, wrong, wrong)

        with pytest.raises(ValueError, match="same number of rows and columns"):
            DistributedRun.from_model(built, with_river_geometry=True)


class TestTheEnginesCannotBeReachedUnvalidated:
    """The seam is enforced by the signatures, not by remembering to call it."""

    @pytest.mark.parametrize(
        "func, expected",
        [
            (DistributedRRM.run_lumped_model, DistributedRun),
            (DistributedRRM.route_muskingum, DistributedRun),
            (DistributedRRM.route_maxbas, DistributedRun),
            (Wrapper.run_muskingum, DistributedRun),
            (Wrapper.run_maxbas, DistributedRun),
            (Wrapper.run_lumped, LumpedRun),
        ],
    )
    def test_every_engine_entry_takes_a_validated_run(self, func, expected):
        """Test that each engine method's first parameter is a run type, not a catchment.

        Test scenario:
            This is what makes the validation unskippable: `Calibration` used to call
            `Wrapper` directly with a catchment and so bypassed every check. It cannot now --
            there is nothing to pass but a `DistributedRun` or a `LumpedRun`.

        Args:
            func: The engine entry point.
            expected: The run type its first parameter must be annotated with.
        """
        first = next(iter(inspect.signature(func).parameters.values()))

        assert first.annotation in (expected, expected.__name__), (
            f"{func.__qualname__}'s first parameter must be {expected.__name__}, got "
            f"{first.annotation!r}"
        )

    def test_the_engines_do_not_write_to_the_catchment(self, built: Catchment):
        """Test that running the engine leaves the catchment untouched.

        Test scenario:
            The engines used to assign results back onto the model they read. Returning them
            instead is what lets the run type be frozen inputs, and means a run cannot
            half-overwrite the object it was handed.
        """
        run = DistributedRun.from_model(built)

        results = Wrapper.run_muskingum(run)

        assert built.results is None, (
            "the engine must not write to the catchment; the entry point in hapi.run is what "
            "assigns model.results"
        )
        assert results.q_total is not None, "the results come back as a return value"


class TestLumpedNarrowing:
    """The lumped side gets the same treatment, against its own record shape."""

    def test_a_driver_record_of_the_wrong_width_is_refused(
        self, built: Catchment, coello_start_date: str, coello_end_date: str
    ):
        """Test that a record without the four driver columns raises.

        Test scenario:
            `Wrapper.run_lumped` reads `data[:, 3]` -- the long-term average -- so a
            three-column record fails inside the run. The shape is settled up front instead.
        """
        built.data = np.ones((len(built.period), 3))

        with pytest.raises(ValueError, match=r"\(time, 4\) array"):
            LumpedRun.from_model(built)

    def test_a_record_that_does_not_span_the_period_is_refused(self, built: Catchment):
        """Test that a record of the wrong length raises rather than misaligning silently.

        Test scenario:
            The run is positional, so a record shorter or longer than the period pairs each
            step with the wrong date and still produces numbers.
        """
        built.data = np.ones((len(built.period) + 5, 4))

        with pytest.raises(ValueError, match="the run is positional"):
            LumpedRun.from_model(built)
