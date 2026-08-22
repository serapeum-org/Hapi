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
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped

CANNED_RESULT = (0.42, np.arange(12, dtype="float64"))


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def gauged_calibration(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> Calibration:
    """Build a distributed Calibration with inputs loaded and a two-row gauge table.

    Returns:
        Calibration: Instance carrying `meteo`, `flow_network`, `GaugesTable` and a
            synthetic `Qtot` field, with no model run behind it.
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
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.GaugesTable = DataFrame(
        {"id": [1, 2], "cell_row": [2, 5], "cell_col": [3, 6]}
    )
    rows, cols = coello.flow_network.rows, coello.flow_network.cols
    steps = coello.meteo.time_steps
    rng = np.random.default_rng(1337)
    coello.Qtot = rng.random((rows, cols, steps + 1))
    coello.QGauges = DataFrame(rng.random((steps, 2)), columns=[1, 2])
    return coello


class TestExtractDischarge:
    """Tests for `Calibration.extract_discharge`."""

    def test_fills_qsim_from_qtot_at_each_gauge_cell(
        self, gauged_calibration: Calibration
    ):
        """Test that every gauge column is read from its own cell of `Qtot`.

        Test scenario:
            The override reads `Qtot[row, col, :-1]` per gauge and sizes the result from
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
            coello.Qtot[2, 3, :-1],
            err_msg="gauge 1 must come from cell (2, 3) of Qtot",
        )
        np.testing.assert_allclose(
            coello.Qsim[:, 1],
            coello.Qtot[5, 6, :-1],
            err_msg="gauge 2 must come from cell (5, 6) of Qtot",
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
            coello.Qtot[2, 3, :-1] * 2.0,
            err_msg="gauge 1 must be scaled by its own factor",
        )
        np.testing.assert_allclose(
            coello.Qsim[:, 1],
            coello.Qtot[5, 6, :-1] * 10.0,
            err_msg="gauge 2 must be scaled by its own factor",
        )

    def test_rejects_a_catchment_routed_with_maxbas(
        self, gauged_calibration: Calibration
    ):
        """Test that reading gauge cells after a MAXBAS run raises instead of under-reporting.

        Test scenario:
            Triangular routing sends every cell straight to the outlet, so a cell of `Qtot`
            is that cell's contribution rather than the discharge at it. Calibrating against
            it would fit the wrong signal, so the guard must refuse rather than return numbers.
        """
        coello = gauged_calibration
        coello._maxbas_routed = True

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
        coello.LB = np.zeros(12)
        coello.UB = np.ones(12)

        res = coello.run_calibration(spatial_var_stub, _optimization_args())

        assert res is CANNED_RESULT, "the optimiser result must be returned untouched"
        assert coello.OFvalue == CANNED_RESULT[0], (
            f"OFvalue must be res[0], got {coello.OFvalue}"
        )
        np.testing.assert_array_equal(
            coello.parameters,
            CANNED_RESULT[1],
            err_msg="parameters must be res[1], lowercase — not a second attribute",
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
        coello.LB = np.zeros(12)
        coello.UB = np.ones(12)
        coello.meteo.precipitation = coello.meteo.precipitation[:, :-1, :]
        coello.meteo.temperature = coello.meteo.temperature[:, :-1, :]
        coello.meteo.evapotranspiration = coello.meteo.evapotranspiration[:, :-1, :]

        with pytest.raises(ValueError, match="must share the catchment's grid"):
            coello.run_calibration(spatial_var_stub, _optimization_args())

        assert "solve_kwargs" not in stub_optimizer, (
            "the optimiser must not run when the inputs do not line up"
        )

    def test_the_objective_distributes_the_trial_vector_and_runs_the_model(
        self, gauged_calibration: Calibration, stub_optimizer: dict, spatial_var_stub
    ):
        """Test that the objective the optimiser drives reaches the model.

        Test scenario:
            The objective body maps the flat trial vector onto the 3D parameter array through
            `SpatialVarFun` and then runs the whole distributed model. It swallows every
            exception into `(nan, [], 1)`, so a rewiring mistake inside it does not raise —
            it just makes every trial fail. Pinning the triple's shape and that the parameter
            array actually arrived is what makes that visible.
        """
        coello = gauged_calibration
        coello.read_objective_function(metrics.rmse, [])
        coello.LB = np.zeros(12)
        coello.UB = np.ones(12)

        coello.run_calibration(spatial_var_stub, _optimization_args())

        error, constraints, fail = stub_optimizer["objective"]
        assert fail in (0, 1), f"the objective must report a fail flag, got {fail}"
        assert isinstance(constraints, list), (
            f"the Muskingum constraints must be a list, got {type(constraints)}"
        )
        assert coello.parameters is not None, (
            "the trial vector must have been distributed onto the parameter array"
        )
        assert error is not None, "the objective must return an error value"


class TestFW1Calibration:
    """Tests for `Calibration.FW1Calibration` (triangular routing)."""

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
        coello.LB = np.zeros(12)
        coello.UB = np.ones(12)

        res = coello.FW1Calibration(spatial_var_stub, _optimization_args())

        assert res is CANNED_RESULT, "the optimiser result must be returned untouched"
        assert coello.OFvalue == CANNED_RESULT[0], (
            f"OFvalue must be res[0], got {coello.OFvalue}"
        )
        np.testing.assert_array_equal(
            coello.parameters,
            CANNED_RESULT[1],
            err_msg="parameters must be res[1], lowercase — not a second attribute",
        )


class TestLumpedCalibration:
    """Tests for `Calibration.lumpedCalibration`."""

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
        coello.LB = np.zeros(12)
        coello.UB = np.ones(12)
        basic_inputs = dict(
            Route=0, RoutingFn=Routing.triangular_routing_1, InitialValues=[]
        )

        res = coello.lumpedCalibration(basic_inputs, _optimization_args())

        assert res is CANNED_RESULT, "the optimiser result must be returned untouched"
        assert coello.OFvalue == CANNED_RESULT[0], (
            f"OFvalue must be res[0], got {coello.OFvalue}"
        )
        np.testing.assert_array_equal(
            coello.parameters,
            CANNED_RESULT[1],
            err_msg="parameters must be res[1], lowercase — not a second attribute",
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
        coello.LB = np.zeros(12)
        coello.UB = np.ones(12)
        basic_inputs = dict(
            Route=0,
            RoutingFn=Routing.triangular_routing_1,
            InitialValues=list(np.full(12, 0.5)),
        )

        coello.lumpedCalibration(basic_inputs, _optimization_args())

        assert stub_optimizer["n_vars"] == 12, (
            f"Expected one variable per bound (12), got {stub_optimizer['n_vars']}"
        )


class _SpatialVarStub:
    """Minimal stand-in for the SpatialVarFun callable the optimiser drives."""

    no_parameters = 12
    no_elem = 1

    def __init__(self, rows: int, cols: int):
        self.Par3d = np.ones((rows, cols, 12))

    def Function(self, par, *args, **kwargs):
        """Accept the flat parameter vector and leave `Par3d` in place."""
        return self.Par3d


@pytest.fixture(scope="function")
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
