"""End-to-end tests for the two lake-aware wrapper entry points.

`Wrapper.RRMWithlake` and `Wrapper.FW1Withlake` run the lake as a lumped inflow, add its routed
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
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.rrm.hbv_lake import HBVLake
from hapi.run import Run
from hapi.wrapper import Wrapper

JIBOA_ROOT = "tests/rrm/data/jiboa"
OUTFLOW_CELL = [2, 1]
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


@pytest.fixture(scope="module")
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
    """Build a distributed Coello catchment ready for a lake-aware run.

    Returns:
        Catchment: Model with `meteo`, `flow_network`, parameters and a lumped model, plus a
            synthetic flow-path-length grid for the triangular-routing path.
    """
    coello = Catchment(
        "coello-lake",
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
    coello.read_parameters(coello_dist_parameters_muskingum, False)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    coello.flow_path_length_arr = _flow_path_length_like(coello)
    return coello


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
    """Tests for `Wrapper.RRMWithlake` (Muskingum routing)."""

    def test_routes_the_lake_into_the_outflow_cell_and_fills_the_outputs(
        self,
        coello_with_lake_inputs: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that a lake-aware Muskingum run produces the routed discharge field.

        Test scenario:
            The lake is simulated, routed with its own Muskingum parameters, and added to
            `quz` at the outflow cell before the spatial routing runs. Like the non-lake
            Muskingum path, the result is the spatial `Qtot` field rather than an aggregated
            outlet series, so it must come back at the size `MeteoInputs` reports — and the
            lake series must carry one slot per simulation step, the initial state included.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=7)

        Wrapper.RRMWithlake(model, lake)

        steps = model.meteo.simulation_steps
        rows, cols = model.flow_network.rows, model.flow_network.cols
        assert model.Qtot.shape == (rows, cols, steps), (
            f"Expected Qtot {(rows, cols, steps)}, got {model.Qtot.shape}"
        )
        assert len(lake.QlakeR) == steps, (
            f"the routed lake series must cover the simulation: {len(lake.QlakeR)} vs "
            f"{steps}"
        )
        outflow = model.quz_routed[OUTFLOW_CELL[0], OUTFLOW_CELL[1], :]
        assert np.isfinite(outflow).all(), (
            "the outflow cell's routed upper-zone discharge must be finite"
        )

    def test_clears_the_maxbas_flag_left_by_an_earlier_triangular_run(
        self,
        coello_with_lake_inputs: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that the Muskingum lake path resets `_maxbas_routed`.

        Test scenario:
            The flag tells `extract_discharge` that a cell of `Qtot` is a contribution rather
            than a discharge, and it is set by the triangular path. Re-running the same model
            through Muskingum makes the outlet-cell shortcut valid again, so a stale True
            would refuse a reading that is now correct.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=11)
        model._maxbas_routed = True

        Wrapper.RRMWithlake(model, lake)

        assert model._maxbas_routed is False, (
            "a Muskingum lake run must clear the MAXBAS flag"
        )

    def test_the_public_entry_point_completes_with_a_record_of_the_documented_length(
        self,
        coello_with_lake_inputs: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that `Run.runHAPIwithLake` runs end to end, validation included.

        Test scenario:
            The entry point asserts the lake record covers exactly `meteo.time_steps` and
            then hands the model to the wrapper. Driving the whole chain — rather than
            spying the wrapper out — is what pins that the two agree on the length: they
            disagreed by one step, so this call raised for every input it accepted.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=23)

        Run.runHAPIwithLake(model, lake)

        assert lake.MeteoData.shape[0] == model.meteo.time_steps, (
            "the record the entry point accepts must be the one the wrapper can run"
        )
        assert model.Qtot is not None, "the run must leave a routed discharge field"


class TestFW1WithLake:
    """Tests for `Wrapper.FW1Withlake` (triangular/MAXBAS routing)."""

    def test_fills_the_distributed_output_fields(
        self,
        coello_with_lake_inputs: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that the triangular lake path populates `Qtot`, `quz_routed`, `qlz_translated`.

        Test scenario:
            `save_results` and `plot_distributed_results` read those three fields, and only
            the Muskingum path used to set them. The triangular path fills them from the
            sub-catchment alone — the lake is a lumped inflow with no spatial extent — so all
            three must come back at grid size and finite.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=13)

        Wrapper.FW1Withlake(model, lake)

        rows, cols = model.flow_network.rows, model.flow_network.cols
        steps = model.meteo.simulation_steps
        for name in ("Qtot", "quz_routed", "qlz_translated"):
            field = getattr(model, name)
            assert field.shape == (rows, cols, steps), (
                f"{name} should be {(rows, cols, steps)}, got {field.shape}"
            )
        assert np.isfinite(model.qout).all(), "the routed outlet series must be finite"

    def test_marks_the_model_as_maxbas_routed(
        self,
        coello_with_lake_inputs: Catchment,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that the triangular lake path sets `_maxbas_routed`.

        Test scenario:
            Triangular routing sends every cell straight to the outlet, so reading a gauge
            cell of `Qtot` under-reports. The flag is what makes `extract_discharge` refuse
            rather than return the wrong hydrograph, so it must be set by this path too.
        """
        model = coello_with_lake_inputs
        lake = _make_lake(model, coello_start_date, coello_end_date, seed=17)
        model._maxbas_routed = False

        Wrapper.FW1Withlake(model, lake)

        assert model._maxbas_routed is True, (
            "a triangular lake run must mark the model as MAXBAS-routed"
        )
