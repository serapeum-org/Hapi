"""Tests for the refusals `Calibration` makes before and during a search.

A calibration runs the model thousands of times, so the difference between a check that
fires at the call that got it wrong and one that fires on the first trial is the difference
between an error a caller can act on and a search that burns minutes before failing. These
are the guards that decide that, plus the lumped entry point's own objective loop -- the one
path where the optimiser's vector *is* the parameter set the conceptual model reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statista.descriptors as metrics

from hapi.calibration import Calibration, ObjectiveFunctionArityError
from hapi.catchment import Catchment
from hapi.conceptual import ParameterBounds
from hapi.conceptual.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.inputs import MeteoInputs
from hapi.results import RoutingKind, SimulationResults

CANNED_RESULT = (0.25, np.arange(12, dtype=float), {"time": 1.0})


def _optimization_args() -> list:
    """Build the three-element optimisation argument list the entry points unpack.

    Returns:
        list: `[api_obj_args, pll_type, api_solve_args]`.
    """
    return [
        dict(hms=2, hmcr=0.95, par=0.65, dbw=10, fileout=0, xinit=0, filename=""),
        None,
        dict(store_sol=False, display_opts=False, store_hst=False, hot_start=False),
    ]


def _tiny_meteo() -> MeteoInputs:
    """Build the smallest driver set the gauge accessor needs to get past its meteo guard.

    Returns:
        MeteoInputs: Three `(2, 3, 4)` cubes.
    """
    cube = np.zeros((2, 3, 4))
    return MeteoInputs(cube, cube.copy(), cube.copy())


@pytest.fixture
def stub_engine(monkeypatch) -> dict:
    """Replace the Harmony Search engine with a stub that drives the objective once.

    Args:
        monkeypatch: Used to swap `HSapi` out of the calibration module.

    Returns:
        dict: Records the objective's return value under `"scored"`.
    """
    from hapi import calibration as calibration_module

    recorded: dict = {}

    class _StubEngine:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, opt_prob, *args, **kwargs):
            recorded["scored"] = opt_prob.obj_fun(np.full(12, 0.5))
            return CANNED_RESULT

    monkeypatch.setattr(calibration_module, "HSapi", _StubEngine)
    return recorded


@pytest.fixture
def lumped(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
    lumped_gauges_path: str,
    coello_gauges_date_fmt: str,
) -> Calibration:
    """A lumped calibration with drivers, a model, bounds and an observed record.

    Returns:
        Calibration: Ready for `calibrate_lumped`.
    """
    coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))
    coello.model.read_lumped_inputs(lumped_meteo_data_path)
    coello.model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    coello.model.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    coello.read_parameters_bound([0.0] * 12, [1.0] * 12, False)
    coello.read_objective_function(metrics.rmse, [])
    return coello


class TestReadParametersBound:
    """The reader that settles the search space."""

    @pytest.mark.parametrize("snow", [0, 1, "yes", None])
    def test_a_non_bool_snow_flag_is_refused(self, coello_rrm_date: list, snow):
        """Test that `snow` must be a bool, not something merely truthy.

        Args:
            coello_rrm_date: Start and end dates for the model.
            snow: A value that is not a bool.

        Test scenario:
            `snow` selects the parameter count through `PARAMETER_COUNTS[(snow, maxbas)]`,
            and `0`/`1` hash equal to `False`/`True` — so a caller passing an int gets the
            right answer by accident and a caller passing a string gets a `KeyError` much
            later. The type is checked where it enters.
        """
        coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))

        with pytest.raises(ValueError, match="True or False"):
            coello.read_parameters_bound([0.0] * 12, [1.0] * 12, snow)


class TestReadObjectiveFunction:
    """The reader that settles what a trial is scored with."""

    def test_something_that_cannot_be_called_is_refused(self, coello_rrm_date: list):
        """Test that a non-callable objective is named at the call that supplied it.

        Args:
            coello_rrm_date: Start and end dates for the model.

        Test scenario:
            Otherwise the failure is a `TypeError: 'str' object is not callable` from inside
            the optimiser's first trial, several frames from the registration.
        """
        coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))

        with pytest.raises(TypeError, match="should be a function"):
            coello.read_objective_function("rmse", [])

    def test_omitting_the_extra_arguments_leaves_an_empty_list(
        self, coello_rrm_date: list
    ):
        """Test that `args=None` becomes an empty list rather than staying `None`.

        Args:
            coello_rrm_date: Start and end dates for the model.

        Test scenario:
            The entry points forward `*of_args`, which cannot unpack `None`. The default is
            normalised here so every caller of `_objective()` gets something iterable.
        """
        coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))

        coello.read_objective_function(metrics.rmse, None)

        assert coello.OFArgs == [], f"expected [], got {coello.OFArgs!r}"


class TestTheGuardsBeforeTheSearch:
    """What each entry point refuses before the optimisation problem is declared."""

    def test_no_bounds_names_the_reader_that_supplies_them(self, coello_rrm_date: list):
        """Test that starting without a search space says which reader was skipped.

        Args:
            coello_rrm_date: Start and end dates for the model.

        Test scenario:
            Every entry point and the variable declaration read `self.bounds`, and all of
            them used to index it straight — so a caller who forgot got `TypeError` on
            `None`.
        """
        coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))

        with pytest.raises(ValueError, match="read_parameters_bound"):
            coello._search_space()

    def test_no_objective_names_the_reader_that_supplies_it(
        self, coello_rrm_date: list
    ):
        """Test that starting without an objective says which reader was skipped.

        Args:
            coello_rrm_date: Start and end dates for the model.

        Test scenario:
            The same shape as the bounds guard: the objective is read once per trial, so
            reporting it up front is what keeps a caller from waiting for the first one.
        """
        coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))

        with pytest.raises(ValueError, match="read_objective_function"):
            coello._objective()

    def test_an_objective_with_no_readable_signature_is_left_alone(
        self, coello_rrm_date: list
    ):
        """Test that a builtin without an introspectable signature skips the arity check.

        Args:
            coello_rrm_date: Start and end dates for the model.

        Test scenario:
            `inspect.signature` raises for callables that decline to describe themselves —
            some C functions, and anything wrapping them. Refusing those would reject a
            legitimate objective for being unreadable, so the check declines to judge rather
            than guessing; the runtime handler still scores a bad call as infeasible.
        """

        class _Unreadable:
            """A callable that refuses to describe its own signature."""

            @property
            def __signature__(self):
                """Raise the way an un-introspectable callable does."""
                raise ValueError("no signature for this one")

            def __call__(self, *args):
                """Accept anything."""
                return 0.0

        coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))
        coello.read_objective_function(_Unreadable(), [])

        coello._check_objective_arity(99)

    @pytest.mark.parametrize(
        "missing, basic_inputs",
        [
            ("RoutingFn", {"Route": 1}),
            ("Route", {"RoutingFn": None}),
            ("Route", {}),
        ],
    )
    def test_incomplete_basic_inputs_name_the_missing_key(
        self, lumped: Calibration, missing: str, basic_inputs: dict
    ):
        """Test that `calibrate_lumped` names the key its bundle is missing.

        Args:
            lumped: A lumped calibration ready to run.
            missing: A key the bundle omits.
            basic_inputs: The incomplete bundle.

        Test scenario:
            `basic_inputs` is a loose dict, so a typo is a missing key rather than a missing
            argument, and the failure would otherwise be a `KeyError` inside the objective
            once per trial.
        """
        args = _optimization_args()

        with pytest.raises(ValueError, match=missing):
            lumped.calibrate_lumped(basic_inputs, args)

    def test_no_observed_record_names_the_reader_that_supplies_it(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
        stub_engine: dict,
    ):
        """Test that a lumped calibration without gauges says what it cannot score against.

        Args:
            coello_rrm_date: Start and end dates for the model.
            lumped_meteo_data_path: Driver record.
            coello_AreaCoeff: Catchment area.
            coello_InitialCond: Initial state.
            stub_engine: Drives the objective once.

        Test scenario:
            The objective is handed `QGauges` as its first argument. Unread, that is `None`,
            and the metric fails on it inside a trial that is then scored infeasible — so
            the whole search reports nothing but `nan`.
        """
        coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))
        coello.model.read_lumped_inputs(lumped_meteo_data_path)
        coello.model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        coello.read_parameters_bound([0.0] * 12, [1.0] * 12, False)
        coello.read_objective_function(metrics.rmse, [])

        args = _optimization_args()

        with pytest.raises(ValueError, match="read_discharge_gauges"):
            coello.calibrate_lumped({"Route": 0, "RoutingFn": None}, args)


class TestGaugedResults:
    """The accessor every gauge-reading path narrows through."""

    @pytest.mark.parametrize(
        "clear, expected",
        [
            ("results", "no trial has completed"),
            ("meteo", "no drivers"),
            ("GaugesTable", "read_gauge_table"),
        ],
    )
    def test_each_missing_input_is_named(
        self, lumped: Calibration, clear: str, expected: str
    ):
        """Test that each of the three inputs is reported by the reader that supplies it.

        Args:
            lumped: A lumped calibration ready to run.
            clear: The attribute to clear on the model.
            expected: Substring the error must carry.

        Test scenario:
            All three are indexed straight afterwards, so an unset one used to fail on
            `None` inside the extraction loop -- naming an attribute of an array rather than
            the step nobody took.
        """
        lumped.model.results = SimulationResults(
            RoutingKind.MUSKINGUM,
            np.zeros((2, 3, 4), dtype="float32"),
            np.zeros((2, 3, 4), dtype="float32"),
            None,
            q_total=np.zeros((2, 3, 4), dtype="float32"),
        )
        lumped.model.meteo = _tiny_meteo()
        lumped.model.GaugesTable = pd.DataFrame({"cell_row": [0], "cell_col": [0]})
        setattr(lumped.model, clear, None)

        with pytest.raises(ValueError, match=expected):
            lumped._gauged_results()

    def test_results_with_no_routed_discharge_are_refused(self, lumped: Calibration):
        """Test that routed-looking results carrying no `q_total` say so.

        Args:
            lumped: A lumped calibration ready to run.

        Test scenario:
            The routing kind and the arrays are set separately, so a results object can
            claim a routing it does not have the fields for -- which is a different failure
            from "nothing routed these", and gets its own message.
        """
        lumped.model.results = SimulationResults(
            RoutingKind.MUSKINGUM,
            np.zeros((2, 3, 4), dtype="float32"),
            np.zeros((2, 3, 4), dtype="float32"),
            None,
        )
        lumped.model.meteo = _tiny_meteo()
        lumped.model.GaugesTable = pd.DataFrame({"cell_row": [0], "cell_col": [0]})

        with pytest.raises(ValueError, match="no routed discharge"):
            lumped.extract_discharge()


class TestCalibrateLumped:
    """The one entry point where the optimiser's vector is the parameter set itself."""

    def test_a_trial_runs_the_model_and_scores_it(
        self, lumped: Calibration, stub_engine: dict
    ):
        """Test that the objective loop runs the model and returns a finite score.

        Args:
            lumped: A lumped calibration ready to run.
            stub_engine: Drives the objective once and records what it returned.

        Test scenario:
            This is the body nothing exercised: the trial installs its parameters, runs the
            lumped model, scores `Qsim` against the observed record, and builds the two
            Muskingum stability constraints. A stubbed optimiser is what makes it a unit
            test rather than a search.
        """
        result = lumped.calibrate_lumped(
            {"Route": 0, "RoutingFn": None}, _optimization_args()
        )

        assert result is CANNED_RESULT, "the optimiser's result comes back untouched"
        error, constraints, fail = stub_engine["scored"]
        assert fail == 0, f"a valid trial must not be scored infeasible, got {fail}"
        assert np.isfinite(error), f"the score must be a real number, got {error}"
        assert len(constraints) == 2, (
            f"the two Muskingum stability constraints must be returned, got {constraints}"
        )

    def test_a_failing_trial_is_scored_infeasible_rather_than_ending_the_search(
        self, lumped: Calibration, stub_engine: dict
    ):
        """Test that a trial that blows up numerically does not stop the calibration.

        Args:
            lumped: A lumped calibration ready to run.
            stub_engine: Drives the objective once and records what it returned.

        Test scenario:
            One bad candidate out of thousands is normal, and the handler exists so the
            search survives it. The arity escape added this round had to not break that.
        """

        def explodes(observed, simulated):
            """Fail on the values, with the arity the call site passes."""
            raise ZeroDivisionError("no discharge at this gauge")

        lumped.read_objective_function(explodes, [])

        lumped.calibrate_lumped({"Route": 0, "RoutingFn": None}, _optimization_args())

        error, constraints, fail = stub_engine["scored"]
        assert fail == 1, "a failing trial is reported as infeasible"
        assert np.isnan(error), f"an infeasible trial scores nan, got {error}"
        assert constraints == [], (
            f"no constraints survive a failed trial, got {constraints}"
        )

    def test_a_wrongly_wired_objective_stops_before_the_optimiser_is_built(
        self, lumped: Calibration, stub_engine: dict
    ):
        """Test that the lumped path checks arity too, before declaring the problem.

        Args:
            lumped: A lumped calibration ready to run.
            stub_engine: Records whether the engine was reached.

        Test scenario:
            The lumped call shape is `objective(observed, Qsim, *of_args)`. An objective that
            cannot accept that many is a wiring error, not a bad candidate, and every trial
            would fail identically.
        """

        def needs_five(a, b, c, d, e):
            """Take more arguments than the entry point passes."""
            return 0.0

        lumped.read_objective_function(needs_five, [])

        args = _optimization_args()

        with pytest.raises(ObjectiveFunctionArityError, match="needs more inputs"):
            lumped.calibrate_lumped({"Route": 0, "RoutingFn": None}, args)

        assert "scored" not in stub_engine, (
            "the optimiser must not be built when the objective cannot be called"
        )

    def test_a_search_width_the_model_cannot_read_is_refused(
        self, lumped: Calibration, stub_engine: dict
    ):
        """Test that the lumped width rule fires before the problem is declared.

        Args:
            lumped: A lumped calibration ready to run.
            stub_engine: Records whether the engine was reached.

        Test scenario:
            A lumped calibration is the one case where the optimiser's vector *is* the
            parameter set, so its width has to match `PARAMETER_COUNTS`. The rule cannot
            live on `ParameterBounds`, which also bounds distributed searches of 980 values.
        """
        lumped.bounds = ParameterBounds(np.zeros(9), np.ones(9))

        args = _optimization_args()

        with pytest.raises(ValueError, match="takes 12 parameters"):
            lumped.calibrate_lumped({"Route": 0, "RoutingFn": None}, args)

        assert "scored" not in stub_engine, (
            "the optimiser must not be built for a width the model cannot read"
        )
