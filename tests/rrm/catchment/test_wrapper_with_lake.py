"""End-to-end tests for the two lake-aware wrapper entry points.

`Wrapper.run_muskingum_with_lake` and `Wrapper.run_maxbas_with_lake` run the lake as a lumped inflow, add its routed
discharge to the outflow cell, and then route the sub-catchment — Muskingum in the first case,
triangular (MAXBAS) in the second. Neither had test coverage, so the sizes they read off
`MeteoInputs` and `FlowNetwork` and the `_maxbas_routed` flag they leave behind were unpinned.

The bundled Jiboa lake fixture is hourly and twenty steps long while its distributed rasters are
absent from the repository, so the lake here is driven over the Coello grid instead: real
parameters and rating curve, synthetic meteorological record sized to the Coello run.
"""

from __future__ import annotations

import numpy as np
import pytest

from hapi.catchment import Catchment, Lake
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.rrm.hbv_lake import HBVLake
from hapi.run import Run
from hapi.runs import DistributedRun
from hapi.wrapper import Wrapper

JIBOA_ROOT = "tests/rrm/data/jiboa"
# Must be inside the Coello domain. (2, 1) -- the cell the Jiboa fixture drains into -- is
# masked here, so the lake's Muskingum parameters read NaN, `muskingum_v` divides by NaN, and
# the injected series is all-NaN. The cell then routes to zeros and every assertion on it
# passes without testing anything.
OUTFLOW_CELL = [1, 5]
LAKE_CAT_AREA = 133.98
LAKE_AREA = 70.64
INITIAL_COND_LAKE = [0, 5, 5, 5, 0, 1.021144022048255e10]
STAGE_DISCHARGE_CURVE = np.array(
    [
        [1.00000000e-02, 1.01196261e10],
        [8.40000000e-02, 1.01543596e10],
        [9.99000000e-01, 1.01958839e10],
        [1.11000000e00, 1.01980622e10],
        [1.22600000e00, 1.02002405e10],
        [2.90200000e00, 1.02244921e10],
        [3.15200000e00, 1.02273965e10],
        [4.18500000e00, 1.02382879e10],
        [1.05920000e01, 1.02847580e10],
        [1.16520000e01, 1.02905668e10],
        [1.53900000e01, 1.03087192e10],
        [2.50430000e01, 1.03450240e10],
        [3.85590000e01, 1.03824905e10],
        [8.88000000e01, 1.04539383e10],
    ]
)


def _build_coello(
    start: str,
    end: str,
    prec: str,
    temp: str,
    evap: str,
    acc: str,
    fd: str,
    parameters: str,
    area: int,
    initial_cond: list,
    maxbas: bool = False,
) -> Catchment:
    """Assemble a distributed Coello catchment with no run behind it.

    Args:
        start: Simulation start date, `%Y-%m-%d`.
        end: Simulation end date, `%Y-%m-%d`.
        prec: Folder of precipitation rasters.
        temp: Folder of temperature rasters.
        evap: Folder of evapotranspiration rasters.
        acc: Flow accumulation raster.
        fd: Flow direction raster.
        parameters: Folder of distributed parameter rasters.
        area: Catchment area in km2.
        initial_cond: Initial HBV state.
        maxbas: Whether `parameters` carries the triangular-routing parameter. The two sets
            differ in their last band -- MAXBAS against the Muskingum X -- and `route_maxbas`
            routes with whatever sits there, so a triangular run needs the MAXBAS set.

    Returns:
        Catchment: Model with `meteo`, `flow_network`, parameters, a lumped model and a
            synthetic flow-path-length grid for the triangular-routing path.
    """
    coello = Catchment(
        "coello-lake",
        start,
        end,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    coello.meteo = MeteoInputs.from_rasters(
        prec,
        temp,
        evap,
        start=start,
        end=end,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    coello.flow_network = FlowNetwork.from_rasters(acc, fd)
    coello.read_parameters(parameters, False, maxbas=maxbas)
    coello.read_lumped_model(HBVLumped, area, initial_cond)
    coello.flow_path_length_arr = _flow_path_length_like(coello)
    return coello


@pytest.fixture
def coello_with_lake_inputs(
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
    """Provide a fresh distributed Coello catchment for each test.

    Function-scoped deliberately: the wrappers mutate the model in place, so a shared
    instance lets one test's routed fields satisfy the next test's assertions.

    Returns:
        Catchment: Model ready for a lake-aware run, with no run behind it.
    """
    return _build_coello(
        coello_start_date,
        coello_end_date,
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        coello_acc_path,
        coello_fd_path,
        coello_dist_parameters_muskingum,
        coello_cat_area,
        coello_initial_cond,
    )


@pytest.fixture
def coello_no_lake(
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
    """Provide a second, identical catchment to run without a lake as the control.

    Returns:
        Catchment: Same inputs as `coello_with_lake_inputs`, to be routed by `run_muskingum`.
    """
    return _build_coello(
        coello_start_date,
        coello_end_date,
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        coello_acc_path,
        coello_fd_path,
        coello_dist_parameters_muskingum,
        coello_cat_area,
        coello_initial_cond,
    )


@pytest.fixture
def coello_with_lake_inputs_maxbas(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_maxbas: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> Catchment:
    """Provide the same catchment carrying a MAXBAS parameter set.

    `route_maxbas` routes each cell with `parameters[..., -1]`, which is MAXBAS in this set and
    the Muskingum X in the other. Handed the Muskingum set, every cell routed with X = 0.2 --
    below the one whole step a triangle needs -- and `triangular_routing_1` returned an all-zero
    hydrograph without raising, so the triangular tests asserted their shapes and flags against
    output that carried nothing. In-domain MAXBAS here runs 1.4 to 2.4.

    Returns:
        Catchment: Model ready for a triangular lake-aware run.
    """
    return _build_coello(
        coello_start_date,
        coello_end_date,
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        coello_acc_path,
        coello_fd_path,
        coello_dist_parameters_maxbas,
        coello_cat_area,
        coello_initial_cond,
        maxbas=True,
    )


def _flow_path_length_like(model: Catchment) -> np.ndarray:
    """Build a flow-path-length grid masked to the catchment's active cells.

    Args:
        model: Model whose flow accumulation defines which cells are inside the catchment.

    Returns:
        numpy.ndarray: Distances increasing away from the top-left corner, NaN outside the
            catchment so the MAXBAS normalisation skips those cells.
    """
    acc = model.flow_network.flow_acc_arr
    rows, cols = model.flow_network.rows, model.flow_network.cols
    distance = np.add.outer(np.arange(rows), np.arange(cols)).astype("float64") * 1000.0
    return np.where(np.isnan(acc), np.nan, distance)


def _make_lake(
    model: Catchment, start: str, end: str, seed: int, snow: int = 0
) -> Lake:
    """Assemble a Lake whose record runs step for step with the catchment.

    Args:
        model: Model supplying the number of steps the lake record must cover.
        start: Simulation start date, `%Y-%m-%d`.
        end: Simulation end date, `%Y-%m-%d`.
        seed: Seed for the synthetic meteorological record.
        snow: Whether the lake runs the snow routine.

    Returns:
        Lake: Lake with parameters, rating curve and a four-column meteorological record.
    """
    lake = Lake(start=start, end=end, fmt="%Y-%m-%d", temporal_resolution="Daily")
    lake.read_parameters(f"{JIBOA_ROOT}/lake-parameters.txt")
    lake.read_lumped_model(
        HBVLake,
        LAKE_CAT_AREA,
        LAKE_AREA,
        INITIAL_COND_LAKE,
        OUTFLOW_CELL,
        STAGE_DISCHARGE_CURVE,
        snow,
    )
    rng = np.random.default_rng(seed)
    steps = model.meteo.time_steps
    temperature = 18.0 + 6.0 * rng.random(steps)
    lake.MeteoData = np.column_stack(
        [
            rng.random(steps) * 5.0,
            rng.random(steps) * 3.0,
            temperature,
            np.full(steps, temperature.mean()),
        ]
    )
    return lake


class TestRRMWithLake:
    """Tests for `Wrapper.run_muskingum_with_lake` (Muskingum routing)."""

    def test_the_lake_raises_discharge_at_the_outflow_cell(
        self,
        coello_with_lake_inputs: Catchment,
        coello_no_lake: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that the lake's routed outflow actually reaches the cell it drains into.

        Test scenario:
            Injecting the lake into `quz` at the outflow cell is the one thing `run_muskingum_with_lake`
            does that `run_muskingum` does not. Running the same catchment both ways isolates it:
            everything else is identical, so the whole difference is the lake.

            The increase is *not* `QlakeR` — the lake series is routed a second time, through
            the outflow cell's own Muskingum K and x — so comparing against that second
            routing pins which cell the lake lands in and whose parameters carried it there.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=7)

        coello_no_lake.results = Wrapper.run_muskingum(
            DistributedRun.from_model(coello_no_lake)
        )
        model.results = Wrapper.run_muskingum_with_lake(
            DistributedRun.from_model(model), lake
        )

        row, col = OUTFLOW_CELL
        without = coello_no_lake.results.quz_routed[row, col, :]
        with_lake = model.results.quz_routed[row, col, :]

        assert np.isfinite(with_lake).all(), (
            "the outflow cell must stay finite -- an all-zero or all-NaN series here means "
            "the cell is outside the domain and the comparison below is vacuous"
        )
        assert with_lake.sum() > without.sum(), (
            f"the lake must raise the outflow cell's discharge: {with_lake.sum()} vs "
            f"{without.sum()} without it"
        )
        expected = Routing.muskingum_v(
            lake.QlakeR,
            lake.QlakeR[0],
            model.parameters.values[row, col, 10],
            model.parameters.values[row, col, 11],
            model.period.conversion_factor,
        )
        np.testing.assert_allclose(
            with_lake - without,
            expected,
            rtol=1e-6,
            err_msg=(
                "the increase must be the lake series routed through the outflow cell's own "
                "Muskingum parameters -- a different cell or different parameters would not "
                "reproduce it"
            ),
        )

    def test_the_routed_lake_series_covers_every_simulation_step(
        self,
        coello_with_lake_inputs: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that the lake series lines up slot for slot with the arrays it is added to.

        Test scenario:
            `HBVLake.simulate` prepends the initial-state slot and `muskingum_v` preserves
            length, so the routed series is already `simulation_steps` long. Padding it again
            -- which the wrapper used to do -- made it one longer than `quz` and raised for
            every input.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=11)

        model.results = Wrapper.run_muskingum_with_lake(
            DistributedRun.from_model(model), lake
        )

        steps = model.meteo.simulation_steps
        rows, cols = model.flow_network.rows, model.flow_network.cols
        assert len(lake.QlakeR) == steps, (
            f"the routed lake series must be {steps} long, got {len(lake.QlakeR)}"
        )
        assert model.results.q_total.shape == (rows, cols, steps), (
            f"Expected q_total {(rows, cols, steps)}, got {model.results.q_total.shape}"
        )
        assert np.isfinite(lake.QlakeR).all(), (
            "the routed lake series must be finite; NaN means the outflow cell's Muskingum "
            "parameters were read from a masked cell"
        )

    def test_a_muskingum_lake_run_clears_a_flag_a_triangular_run_set(
        self,
        coello_with_lake_inputs_maxbas: Catchment,
        coello_dist_parameters_muskingum: str,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test the real handshake: a triangular run sets the flag, a Muskingum run clears it.

        Test scenario:
            The flag tells `extract_discharge` that a cell of `q_total` is a contribution rather
            than a discharge. Setting it by hand would test the assignment against itself, so
            drive both paths on the same model in the order that makes the flag matter. The
            two read different parameter layouts -- Muskingum takes bands 10 and 11, the
            triangular path the last one -- so each leg is given the set it indexes into. The
            model, and therefore the flag under test, is the same object throughout.
        """
        model = coello_with_lake_inputs_maxbas
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=13)

        model.results = Wrapper.run_maxbas_with_lake(
            DistributedRun.from_model(model, needs_flow_direction=False), lake
        )
        assert not model.results.outlet_shortcut_valid, (
            "the triangular path must mark the model before this test means anything"
        )

        model.read_parameters(coello_dist_parameters_muskingum, False)
        model.results = Wrapper.run_muskingum_with_lake(
            DistributedRun.from_model(model), lake
        )

        assert model.results.outlet_shortcut_valid, (
            "a Muskingum lake run makes the outlet-cell shortcut valid again"
        )


class TestFW1WithLake:
    """Tests for `Wrapper.run_maxbas_with_lake` (triangular/MAXBAS routing)."""

    def test_fills_the_distributed_output_fields(
        self,
        coello_with_lake_inputs_maxbas: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that the triangular lake path populates `q_total`, `quz_routed`, `qlz_translated`.

        Test scenario:
            `save_results` and `plot_distributed_results` read those three fields, and only
            the Muskingum path used to set them. The fixture is function-scoped so the model
            has been through `run_maxbas_with_lake` and nothing else -- with a shared instance an
            earlier Muskingum run would have filled them and deleting the fix left this green.
        """
        model = coello_with_lake_inputs_maxbas
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=17)
        assert model.results is None, "the fixture must arrive with no run behind it"

        model.results = Wrapper.run_maxbas_with_lake(
            DistributedRun.from_model(model, needs_flow_direction=False), lake
        )

        rows, cols = model.flow_network.rows, model.flow_network.cols
        steps = model.meteo.simulation_steps
        for name in ("q_total", "quz_routed", "qlz_translated"):
            field = getattr(model.results, name)
            assert field is not None, f"{name} must be set by the triangular path"
            assert field.shape == (rows, cols, steps), (
                f"{name} should be {(rows, cols, steps)}, got {field.shape}"
            )

    def test_the_outlet_series_carries_the_lake_and_drops_the_extra_slot(
        self,
        coello_with_lake_inputs_maxbas: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that `qout` is the sub-catchment sum plus the lake, one slot shorter.

        Test scenario:
            The triangular path aggregates the grid into an outlet series and adds the lake
            to it. Both carry the initial-state slot and the non-lake FW1 path drops the last
            entry, so the lake series has to be trimmed the same way -- it was not, and the
            two could not be added at all.
        """
        model = coello_with_lake_inputs_maxbas
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=19)

        model.results = Wrapper.run_maxbas_with_lake(
            DistributedRun.from_model(model, needs_flow_direction=False), lake
        )

        expected_len = model.meteo.simulation_steps - 1
        assert len(model.results.qout) == expected_len, (
            f"qout should drop the trailing slot: expected {expected_len}, "
            f"got {len(model.results.qout)}"
        )
        subcatchment = np.array(
            [
                np.nansum(model.results.qlz[:, :, i])
                + np.nansum(model.results.quz[:, :, i])
                for i in range(model.meteo.simulation_steps)
            ]
        )[:-1]
        np.testing.assert_allclose(
            model.results.qout,
            subcatchment + lake.QlakeR[:-1],
            rtol=1e-6,
            err_msg="qout must be the trimmed sub-catchment sum plus the trimmed lake series",
        )

    def test_marks_the_model_as_maxbas_routed(
        self,
        coello_with_lake_inputs_maxbas: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that the triangular lake path sets `_maxbas_routed` from a clean model.

        Test scenario:
            Triangular routing sends every cell straight to the outlet, so reading a gauge
            cell of `q_total` under-reports. The flag is what makes `extract_discharge` refuse
            rather than return the wrong hydrograph.
        """
        model = coello_with_lake_inputs_maxbas
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=23)
        assert model.results is None, "a fresh model must arrive with no results"

        model.results = Wrapper.run_maxbas_with_lake(
            DistributedRun.from_model(model, needs_flow_direction=False), lake
        )

        assert not model.results.outlet_shortcut_valid, (
            "a triangular lake run must mark the model as MAXBAS-routed"
        )


class TestRunHapiWithLakeEndToEnd:
    """Tests that the public entry point completes with the record it documents."""

    def test_the_entry_point_runs_with_a_record_of_the_documented_length(
        self,
        coello_with_lake_inputs: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that `Run.run_distributed_with_lake` validates and then completes.

        Test scenario:
            The entry point asserts the lake record covers exactly `meteo.time_steps` and
            hands the model to the wrapper. Driving the whole chain is what pins that the two
            agree on the length: they disagreed by one step, so this call raised for every
            input it accepted.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=29)

        Run.run_distributed_with_lake(model, lake)

        assert lake.MeteoData.shape[0] == model.meteo.time_steps, (
            "the record the entry point accepts must be the one the wrapper can run"
        )
        assert model.results.q_total is not None, (
            "the run must leave a routed discharge field"
        )
