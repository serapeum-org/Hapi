"""Tests for the distributed surface of `Calibration` after the FlowNetwork/MeteoInputs move.

The optimiser itself is not exercised here: `HSapi` is replaced by a stub that returns a canned
result, so the tests pin the code around it — the input validation that now reads through
`flow_network` and `meteo`, and the assignment of the optimiser's answer back onto the instance.
"""

from __future__ import annotations

import numpy as np
import pytest
import statista.descriptors as metrics
from pandas import DataFrame

from hapi import calibration as calibration_module
from hapi.calibration import Calibration
from hapi.conceptual import ParameterBounds
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.results import RoutingKind, SimulationResults
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped

CANNED_RESULT = (0.42, np.arange(12, dtype="float64"))


@pytest.fixture
def stub_optimizer(monkeypatch) -> dict:
    """Replace `HSapi` with a stub that records its call and returns a canned result.

    Args:
        monkeypatch: Used to swap the Oasis optimiser out of the calibration module.

    Returns:
        dict: Populated with the solver keywords the calibration passed through and with
            `objective` -- the triple the objective function returned for one trial vector,
            so the body the optimiser would drive is exercised exactly once.
    """
    seen: dict = {}

    class _StubEngine:
        def __init__(self, pll_type=None, options=None):
            seen["pll_type"] = pll_type
            seen["options"] = options

        def __call__(self, opt_prob, **kwargs):
            seen["solve_kwargs"] = kwargs
            seen["n_vars"] = len(opt_prob.getVarSet())
            trial = np.full(len(opt_prob.getVarSet()), 0.5)
            seen["objective"] = opt_prob.obj_fun(trial)
            return CANNED_RESULT

    monkeypatch.setattr(calibration_module, "HSapi", _StubEngine)
    return seen


@pytest.fixture
def gauged_calibration(
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
) -> Calibration:
    """Build a distributed Calibration with inputs loaded and a two-row gauge table.

    Returns:
        Calibration: Instance carrying `meteo`, `flow_network`, `GaugesTable` and a
            synthetic `q_total` field, with no model run behind it.
    """
    coello = Calibration(
        "coello",
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
    # Needed even though the objective overwrites `parameters`: `read_parameters` is also
    # what sets `snow`, and HBV's parameter parse branches on it being exactly 0 or 1. Left
    # as None it raises inside the run, which the objective's bare `except` swallows.
    coello.read_parameters(coello_dist_parameters_muskingum, False)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.GaugesTable = DataFrame(
        {"id": [1, 2], "cell_row": [2, 5], "cell_col": [3, 6]}
    )
    rows, cols = coello.flow_network.rows, coello.flow_network.cols
    steps = coello.meteo.time_steps
    rng = np.random.default_rng(1337)
    # Stage the post-run state the way the run layer builds it. `q_total` and the rest are
    # read-only views onto `results`, so a finished Muskingum run is described rather than
    # poked in field by field.
    coello.results = SimulationResults(
        routing=RoutingKind.MUSKINGUM,
        quz=np.zeros((rows, cols, steps + 1)),
        qlz=np.zeros((rows, cols, steps + 1)),
        state_variables=np.zeros((rows, cols, steps + 1, 5)),
        q_total=rng.random((rows, cols, steps + 1)),
    )
    coello.QGauges = DataFrame(rng.random((steps, 2)), columns=[1, 2])
    return coello


class TestExtractDischarge:
    """Tests for `Calibration.extract_discharge`."""

    def test_fills_qsim_from_qtot_at_each_gauge_cell(
        self, gauged_calibration: Calibration
    ):
        """Test that every gauge column is read from its own cell of `q_total`.

        Test scenario:
            The override reads `q_total[row, col, :-1]` per gauge and sizes the result from
            `meteo.time_steps` — the count that moved onto MeteoInputs. Both columns must
            match the cells the gauge table names, and the trailing step must be dropped.
        """
        coello = gauged_calibration

        coello.extract_discharge()

        expected_shape = (coello.meteo.time_steps, 2)
        assert coello.Qsim.shape == expected_shape, (
            f"Expected Qsim shape {expected_shape}, got {coello.Qsim.shape}"
        )
        np.testing.assert_allclose(
            coello.Qsim[:, 0],
            coello.results.q_total[2, 3, :-1],
            err_msg="gauge 1 must come from cell (2, 3) of q_total",
        )
        np.testing.assert_allclose(
            coello.Qsim[:, 1],
            coello.results.q_total[5, 6, :-1],
            err_msg="gauge 2 must come from cell (5, 6) of q_total",
        )

    def test_factor_scales_each_gauge_independently(
        self, gauged_calibration: Calibration
    ):
        """Test that a per-gauge factor multiplies only its own column.

        Test scenario:
            `factor` carries one multiplier per gauge, applied as the hydrograph is read.
            Passing distinct values must scale the two columns by different amounts.
        """
        coello = gauged_calibration

        coello.extract_discharge(factor=[2.0, 10.0])

        np.testing.assert_allclose(
            coello.Qsim[:, 0],
            coello.results.q_total[2, 3, :-1] * 2.0,
            err_msg="gauge 1 must be scaled by its own factor",
        )
        np.testing.assert_allclose(
            coello.Qsim[:, 1],
            coello.results.q_total[5, 6, :-1] * 10.0,
            err_msg="gauge 2 must be scaled by its own factor",
        )

    def test_rejects_a_catchment_routed_with_maxbas(
        self, gauged_calibration: Calibration
    ):
        """Test that reading gauge cells after a MAXBAS run raises instead of under-reporting.

        Test scenario:
            Triangular routing sends every cell straight to the outlet, so a cell of `q_total`
            is that cell's contribution rather than the discharge at it. Calibrating against
            it would fit the wrong signal, so the guard must refuse rather than return numbers.
        """
        coello = gauged_calibration
        coello.results.routing = RoutingKind.MAXBAS

        with pytest.raises(ValueError, match="MAXBAS") as exc_info:
            coello.extract_discharge()

        assert "under-report" in str(exc_info.value), (
            f"the error should explain the under-reporting, got: {exc_info.value}"
        )


class TestRunCalibration:
    """Tests for `Calibration.run_calibration` (Muskingum)."""

    def test_stores_the_optimizer_result_on_the_instance(
        self, gauged_calibration: Calibration, stub_optimizer: dict, spatial_var_stub
    ):
        """Test that the optimiser's answer lands on `parameters` and `OFvalue`.

        Test scenario:
            The two attributes are the whole output of a calibration run. Writing them under
            a different spelling leaves the caller reading the pre-calibration values, which
            is silent — nothing raises — so pin the exact names and the res-tuple ordering.
        """
        coello = gauged_calibration
        coello.read_objective_function(metrics.rmse, [])
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))

        res = coello.run_calibration(spatial_var_stub, _optimization_args())

        assert res is CANNED_RESULT, "the optimiser result must be returned untouched"
        assert coello.OFvalue == pytest.approx(CANNED_RESULT[0]), (
            f"OFvalue must be res[0], got {coello.OFvalue}"
        )
        np.testing.assert_array_equal(
            coello.best_parameters,
            CANNED_RESULT[1],
            err_msg=(
                "the optimiser's answer belongs on best_parameters: `parameters` is the "
                "runnable ParameterSet, a different shape describing a different thing"
            ),
        )

    def test_rejects_meteo_that_does_not_cover_the_grid(
        self, gauged_calibration: Calibration, stub_optimizer: dict, spatial_var_stub
    ):
        """Test that the grid check runs before the optimiser is ever built.

        Test scenario:
            The validation now reads through `flow_network` and `meteo`. Cropping a column
            off the cubes must be caught up front — a calibration that started on mismatched
            grids would burn a full optimisation before failing.
        """
        coello = gauged_calibration
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))
        # Built in one go: replacing the cubes one at a time is now refused, because a
        # half-applied crop is exactly the inconsistency MeteoInputs guarantees against.
        coello.meteo = MeteoInputs(
            precipitation=coello.meteo.precipitation[:, :-1, :],
            temperature=coello.meteo.temperature[:, :-1, :],
            evapotranspiration=coello.meteo.evapotranspiration[:, :-1, :],
        )

        with pytest.raises(ValueError, match="must share the catchment's grid"):
            coello.run_calibration(spatial_var_stub, _optimization_args())

        assert "solve_kwargs" not in stub_optimizer, (
            "the optimiser must not run when the inputs do not line up"
        )

    def test_the_objective_runs_the_model_on_the_trial_parameters(
        self,
        gauged_calibration: Calibration,
        stub_optimizer: dict,
        spatial_var_stub,
        monkeypatch,
    ):
        """Test that each trial's parameters are what the model is actually run on.

        Test scenario:
            The objective maps the flat trial vector onto the 3D array through
            `SpatialVarFun`, assigns it, and runs the model — all inside a bare `except:`
            that turns any failure into `(nan, [], 1)`. So writing the array under the wrong
            attribute name never raises: the run simply proceeds on whatever parameters were
            already loaded, every trial scores the same, and the optimiser converges on
            noise. That was this branch's C1.

            Neither `fail == 0` nor the value of `parameters` after the run can see it — the
            first because a stale-but-valid array still runs, the second because the
            optimiser's result overwrites it afterwards. What discriminates is the array the
            model held *at the moment it was run*, so spy on the wrapper and compare it to
            the trial vector.
        """
        coello = gauged_calibration
        coello.read_objective_function(_pairwise_objective, [])
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))

        ran_with: list[np.ndarray] = []
        original = calibration_module.Wrapper.run_muskingum

        def spy(model, *args, **kwargs):
            ran_with.append(np.asarray(model.parameters.values, dtype=float).copy())
            return original(model, *args, **kwargs)

        monkeypatch.setattr(
            calibration_module.Wrapper, "run_muskingum", staticmethod(spy)
        )

        coello.run_calibration(spatial_var_stub, _optimization_args())

        error, constraints, fail = stub_optimizer["objective"]
        assert fail == 0, (
            f"the objective must reach its success path; fail={fail} means the run was "
            f"swallowed by the bare except and every trial scores nan (error={error})"
        )
        assert np.isfinite(error), f"a completed trial must score finitely, got {error}"
        assert len(constraints) == 2, (
            f"the Muskingum k/x pair yields two constraints, got {len(constraints)}"
        )
        np.testing.assert_allclose(
            spatial_var_stub.seen_par,
            np.full(12, 0.5),
            err_msg="the optimiser's trial vector must reach SpatialVarFun.Function",
        )
        assert ran_with, "the objective must run the model"
        np.testing.assert_allclose(
            ran_with[0],
            spatial_var_stub.Par3d,
            err_msg=(
                "the model must be run on the trial's distributed parameters; a mismatch "
                "means the objective wrote them somewhere the run does not read"
            ),
        )


class TestFW1Calibration:
    """Tests for `Calibration.calibrate_maxbas` (triangular routing)."""

    def test_stores_the_optimizer_result_on_the_instance(
        self, gauged_calibration: Calibration, stub_optimizer: dict, spatial_var_stub
    ):
        """Test that the FW1 entry point writes back the same two attributes.

        Test scenario:
            Same contract as the Muskingum path — `parameters` and `OFvalue` — through a
            separate code path that also had to be rewired onto `flow_network`.
        """
        coello = gauged_calibration
        coello.read_objective_function(metrics.rmse, [])
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))

        res = coello.calibrate_maxbas(spatial_var_stub, _optimization_args())

        assert res is CANNED_RESULT, "the optimiser result must be returned untouched"
        assert coello.OFvalue == pytest.approx(CANNED_RESULT[0]), (
            f"OFvalue must be res[0], got {coello.OFvalue}"
        )
        np.testing.assert_array_equal(
            coello.best_parameters,
            CANNED_RESULT[1],
            err_msg=(
                "the optimiser's answer belongs on best_parameters: `parameters` is the "
                "runnable ParameterSet, a different shape describing a different thing"
            ),
        )


class TestCheckOptimizationArgs:
    """Tests for `_check_optimization_args`, the guard every entry point runs first."""

    @pytest.mark.parametrize(
        "args, bad_index, bad_kind",
        [
            (["not-a-dict", None, {}], 0, "objective-function"),
            ([{}, None, "not-a-dict"], 2, "solver"),
        ],
        ids=["objective-args", "solver-args"],
    )
    def test_a_non_dict_bundle_is_refused_before_the_optimizer_is_built(
        self,
        gauged_calibration: Calibration,
        stub_optimizer: dict,
        spatial_var_stub,
        args,
        bad_index,
        bad_kind,
    ):
        """Test that each of the two argument bundles is checked, naming which one.

        Args:
            gauged_calibration: A ready-to-run distributed Calibration.
            stub_optimizer: Records whether the optimiser was reached.
            spatial_var_stub: Stand-in for the spatial parameter function.
            args: The `[api_obj_args, pll_type, api_solve_args]` triple, one entry bad.
            bad_index: Which position in `args` carries the non-dict value.
            bad_kind: The word the error message should use for that position.

        Test scenario:
            Both bundles are unpacked with `**` inside Oasis, so anything but a dict fails
            there instead of at the call that supplied it -- unless this guard catches it
            first, before `Optimization(...)` and the harmony-search engine are built at all.
        """
        coello = gauged_calibration
        coello.read_objective_function(metrics.rmse, [])
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))

        with pytest.raises(TypeError, match=f"{bad_kind} arguments should be a dict"):
            coello.run_calibration(spatial_var_stub, args)

        assert "solve_kwargs" not in stub_optimizer, (
            "the optimiser must not be reached when an argument bundle is malformed"
        )


class TestLumpedCalibration:
    """Tests for `Calibration.calibrate_lumped`."""

    def test_stores_the_optimizer_result_on_the_instance(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        stub_optimizer: dict,
    ):
        """Test that the lumped entry point writes back `parameters` and `OFvalue`.

        Test scenario:
            The lumped path assigns the pair in the opposite source order to the distributed
            ones, so it is pinned separately: `OFvalue` is still res[0] and `parameters`
            still res[1].
        """
        coello = Calibration("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_lumped_inputs(lumped_meteo_data_path)
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))
        basic_inputs = dict(
            Route=0, RoutingFn=Routing.triangular_routing_1, InitialValues=[]
        )

        res = coello.calibrate_lumped(basic_inputs, _optimization_args())

        assert res is CANNED_RESULT, "the optimiser result must be returned untouched"
        assert coello.OFvalue == pytest.approx(CANNED_RESULT[0]), (
            f"OFvalue must be res[0], got {coello.OFvalue}"
        )
        np.testing.assert_array_equal(
            coello.best_parameters,
            CANNED_RESULT[1],
            err_msg=(
                "the optimiser's answer belongs on best_parameters: `parameters` is the "
                "runnable ParameterSet, a different shape describing a different thing"
            ),
        )

    def test_initial_values_are_seeded_into_the_problem(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        stub_optimizer: dict,
    ):
        """Test that `InitialValues` reaches the optimisation problem as a starting point.

        Test scenario:
            `basic_inputs["InitialValues"]` selects the seeded branch of the variable setup.
            Both branches must declare one variable per bound, so the problem is the same
            size whether or not a warm start was given.
        """
        coello = Calibration("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_lumped_inputs(lumped_meteo_data_path)
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))
        basic_inputs = dict(
            Route=0,
            RoutingFn=Routing.triangular_routing_1,
            InitialValues=list(np.full(12, 0.5)),
        )

        coello.calibrate_lumped(basic_inputs, _optimization_args())

        assert stub_optimizer["n_vars"] == 12, (
            f"Expected one variable per bound (12), got {stub_optimizer['n_vars']}"
        )

    def test_a_mismatched_initial_values_length_is_refused(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        stub_optimizer: dict,
    ):
        """Test that `InitialValues` shorter than the bounds is rejected, not indexed out of range.

        Args:
            coello_rrm_date: [start, end] dates for the lumped fixture.
            lumped_meteo_data_path: CSV of catchment-average drivers.
            stub_optimizer: Records whether the optimiser was reached.

        Test scenario:
            The seeded branch loops `range(len(self.bounds.lower))` and indexes `initial_values[i]`, so
            a shorter list used to run past its end partway through building the problem,
            leaving `opt_prob` half-populated and raising `IndexError` far from the call that
            supplied the list. The length is now compared up front.
        """
        coello = Calibration("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_lumped_inputs(lumped_meteo_data_path)
        coello.bounds = ParameterBounds(np.zeros(12), np.ones(12))
        basic_inputs = dict(
            Route=0,
            RoutingFn=Routing.triangular_routing_1,
            InitialValues=[0.5, 0.5, 0.5],
        )

        optimization_args = _optimization_args()

        with pytest.raises(ValueError, match="one value per parameter") as exc:
            coello.calibrate_lumped(basic_inputs, optimization_args)

        assert "3" in str(exc.value), (
            f"the error should name the given length: {exc.value}"
        )
        assert "12" in str(exc.value), (
            f"the error should name the expected length too: {exc.value}"
        )
        assert "n_vars" not in stub_optimizer, (
            "the optimiser must not be reached when the seed does not match the bounds"
        )


def _pairwise_objective(qgauges, gauges_table) -> float:
    """Score a trial the way `run_calibration` actually calls the objective.

    The objective is invoked as `objective_function(self.QGauges, self.GaugesTable)`, so a
    metric expecting two aligned series raises `TypeError` and the bare `except` in `opt_fun`
    turns that into `(nan, [], 1)`. This has the signature the call site uses, so the
    objective body runs to completion and `fail` reports the code path rather than the stub.

    Args:
        qgauges: Observed discharge frame.
        gauges_table: Gauge metadata frame.

    Returns:
        float: A finite score derived from both frames.
    """
    return float(np.abs(qgauges.to_numpy(dtype=float)).mean() + len(gauges_table))


class _SpatialVarStub:
    """Stand-in for the SpatialVarFun callable the optimiser drives."""

    no_parameters = 12
    no_elem = 1

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.Par3d = np.ones((rows, cols, 12))
        self.seen_par: np.ndarray | None = None

    def Function(self, par, *args, **kwargs):
        """Broadcast the flat trial vector across the grid, as the real one does.

        Echoing `par` rather than returning a constant is what lets a caller tell whether the
        optimiser's vector actually reached the parameter array.

        Args:
            par: The flat parameter vector the optimiser proposed.

        Returns:
            numpy.ndarray: The `(rows, cols, 12)` array built from `par`.
        """
        self.seen_par = np.asarray(par, dtype=float)
        self.Par3d = np.broadcast_to(
            self.seen_par, (self.rows, self.cols, len(self.seen_par))
        ).copy()
        return self.Par3d


@pytest.fixture
def spatial_var_stub(gauged_calibration: Calibration) -> _SpatialVarStub:
    """Provide a SpatialVarFun stand-in sized to the catchment grid.

    Returns:
        _SpatialVarStub: Object exposing `Function`, `Par3d`, `no_parameters`, `no_elem`.
    """
    return _SpatialVarStub(
        gauged_calibration.flow_network.rows, gauged_calibration.flow_network.cols
    )


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
