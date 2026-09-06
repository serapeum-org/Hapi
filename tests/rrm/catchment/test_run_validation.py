"""Tests for the input validation `Run`'s lake and flood entry points perform.

These three entry points (`run_flood`, `run_distributed_with_lake`, `run_maxbas_with_lake`) each open with a
block of dimension checks that now read through `flow_network` and `meteo` rather than through
attributes on the catchment. The wrapper each one dispatches to is replaced by a spy, so the tests
pin the validation and the dispatch without running a lake or a hydraulic model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hapi import run as run_module
from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs, RiverGeometry
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run
from hapi.runs import DistributedRun


class _LakeStub:
    """Stand-in for `Lake` carrying only the `MeteoData` the validation reads."""

    def __init__(self, time_steps: int, columns: int = 3):
        self.MeteoData = np.ones((time_steps, columns))


@pytest.fixture
def spied_wrapper(monkeypatch) -> dict:
    """Replace every `Wrapper` entry point with a spy that records its call.

    Args:
        monkeypatch: Used to swap the wrapper methods out of the run module.

    Returns:
        dict: Maps the wrapper name to the positional arguments it was called with.
    """
    calls: dict = {}

    def _make(name: str):
        def _spy(*args, **kwargs):
            calls[name] = args

        return _spy

    for name in ("run_muskingum", "run_muskingum_with_lake", "run_maxbas_with_lake"):
        monkeypatch.setattr(run_module.Wrapper, name, staticmethod(_make(name)))
    return calls


@pytest.fixture
def coello_loaded(
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
    """Build a distributed Coello model with every input loaded but no run behind it.

    Returns:
        Catchment: Model carrying `meteo`, `flow_network`, parameters and a lumped model.
    """
    coello = Catchment(
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
    coello.read_parameters(coello_dist_parameters_muskingum, False)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    return coello


def _load_flat_river_geometry(model: Catchment) -> None:
    """Attach uniform river-geometry rasters sized to the catchment grid.

    Args:
        model: Model whose `flow_network` supplies the grid shape.
    """
    shape = (model.flow_network.rows, model.flow_network.cols)
    model.river_geometry = RiverGeometry(
        dem=np.full(shape, 100.0),
        bankfull_depth=np.full(shape, 2.0),
        river_width=np.full(shape, 10.0),
        river_roughness=np.full(shape, 0.03),
        flood_plain_roughness=np.full(shape, 0.06),
    )


class TestRunFloodModel:
    """Tests for `Run.run_flood`."""

    def test_dispatches_once_every_input_lines_up(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that a consistent model reaches the RRM wrapper.

        Test scenario:
            The flood entry point checks the flow-direction grid, the meteo cubes, the
            parameter array and the four river-geometry rasters before dispatching. With all
            of them on the catchment grid it must call `Wrapper.run_muskingum` with the model.
        """
        _load_flat_river_geometry(coello_loaded)

        Run.run_flood(coello_loaded)

        assert "run_muskingum" in spied_wrapper, (
            "run_muskingum should have been dispatched"
        )
        run = spied_wrapper["run_muskingum"][0]
        assert isinstance(run, DistributedRun), (
            f"the wrapper must receive a validated DistributedRun, got {type(run).__name__}"
        )
        assert run.meteo is coello_loaded.meteo, (
            "the run must carry the model's own inputs, not copies"
        )

    def test_rejects_river_geometry_off_the_catchment_grid(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that a mis-sized river raster stops the run before the wrapper.

        Test scenario:
            The river geometry is read from separate rasters, so it can disagree with the
            flow network. A width raster one row short must raise rather than index past the
            end of the grid inside the hydraulic routing.
        """
        shape = (coello_loaded.flow_network.rows, coello_loaded.flow_network.cols)
        flat = np.full(shape, 1.0)

        # `RiverGeometry` refuses the set outright: the five must describe one grid, and that
        # is settled where they are built rather than inside the flood entry point.
        with pytest.raises(ValueError, match="must share one grid"):
            RiverGeometry(flat, flat, flat[:-1, :], flat, flat)

        assert "run_muskingum" not in spied_wrapper, (
            "the wrapper must not run on inconsistent geometry"
        )

    def test_rejects_meteo_that_does_not_cover_the_grid(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that the meteo/grid cross-check runs on the flood path too.

        Test scenario:
            The cubes agree with each other by construction; this is the other half of the
            check — that they cover the catchment. Cropping a column must be caught.
        """
        _load_flat_river_geometry(coello_loaded)
        # Built in one go: replacing the cubes one at a time is now refused, because a
        # half-applied crop is exactly the inconsistency MeteoInputs guarantees against.
        coello_loaded.meteo = MeteoInputs(
            precipitation=coello_loaded.meteo.precipitation[:, :-1, :],
            temperature=coello_loaded.meteo.temperature[:, :-1, :],
            evapotranspiration=coello_loaded.meteo.evapotranspiration[:, :-1, :],
        )

        with pytest.raises(ValueError, match="must share the catchment's grid"):
            Run.run_flood(coello_loaded)

        assert "run_muskingum" not in spied_wrapper, (
            "the wrapper must not run on inconsistent meteo inputs"
        )


class TestRunHapiWithLake:
    """Tests for `Run.run_distributed_with_lake`."""

    def test_dispatches_once_the_lake_record_matches_the_simulation(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that a lake covering the simulation reaches the lake wrapper.

        Test scenario:
            The lake is a lumped inflow whose own meteorological record must run step for
            step with the distributed cubes. A matching record must dispatch to
            `Wrapper.run_muskingum_with_lake` with both the model and the lake.
        """
        lake = _LakeStub(coello_loaded.meteo.time_steps)

        Run.run_distributed_with_lake(coello_loaded, lake)

        assert "run_muskingum_with_lake" in spied_wrapper, (
            "run_muskingum_with_lake should have been dispatched"
        )
        run, seen_lake = spied_wrapper["run_muskingum_with_lake"]
        assert isinstance(run, DistributedRun) and seen_lake is lake, (
            "the wrapper must receive a validated DistributedRun and the lake"
        )

    def test_rejects_a_lake_record_of_the_wrong_length(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that a short lake record is refused.

        Test scenario:
            The lake record is paired with the distributed steps positionally, so a record
            that is one step short would silently shift the lake inflow by a day for the
            whole run. `time_steps` now comes from MeteoInputs — pin that it is the count
            the check reads.
        """
        lake = _LakeStub(coello_loaded.meteo.time_steps - 1)

        with pytest.raises(ValueError, match="same length"):
            Run.run_distributed_with_lake(coello_loaded, lake)

        assert "run_muskingum_with_lake" not in spied_wrapper, (
            "the wrapper must not run against a mismatched lake record"
        )

    def test_rejects_a_lake_record_missing_a_column(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that a lake record without all three drivers is refused.

        Test scenario:
            The lake model reads rain, ET and temperature by position, so fewer than three
            columns cannot be interpreted.
        """
        lake = _LakeStub(coello_loaded.meteo.time_steps, columns=2)

        with pytest.raises(ValueError, match="three columns"):
            Run.run_distributed_with_lake(coello_loaded, lake)

    def test_a_flow_direction_grid_of_the_wrong_shape_cannot_be_installed(
        self, coello_loaded: Catchment
    ):
        """Test that the two rasters cannot be left describing different grids.

        Test scenario:
            `FlowNetwork` sizes itself from the accumulation raster, so a flow-direction array
            of a different shape means the two do not describe the same catchment and the
            routing table would be indexed out of range. `__post_init__` checked that at
            construction but `__setattr__` did not re-check on replacement, so this used to be
            stageable and the run layer re-checked the shape to catch it. The guard now lives
            where the mistake is made, which is why that run-layer check could go.
        """
        network = coello_loaded.flow_network

        with pytest.raises(ValueError, match="must stay"):
            network.flow_dir_arr = network.flow_dir_arr[:-1, :]

        assert network.flow_dir_arr.shape == (network.rows, network.cols), (
            "the refused assignment must leave the network as it was"
        )

    def test_dispatches_once_the_lake_record_matches_the_simulation(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that the triangular-routing lake path validates and dispatches.

        Test scenario:
            The FW1 lake path skips the flow-direction check — triangular routing needs no
            direction grid — but keeps the meteo, parameter and lake checks. A consistent
            model must reach `Wrapper.run_maxbas_with_lake`.
        """
        lake = _LakeStub(coello_loaded.meteo.time_steps)

        Run.run_maxbas_with_lake(coello_loaded, lake)

        assert "run_maxbas_with_lake" in spied_wrapper, (
            "run_maxbas_with_lake should have been dispatched"
        )
        run, seen_lake = spied_wrapper["run_maxbas_with_lake"]
        assert isinstance(run, DistributedRun) and seen_lake is lake, (
            "the wrapper must receive a validated DistributedRun and the lake"
        )

    def test_rejects_parameters_off_the_catchment_grid(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that the parameter array is checked against the flow network.

        Test scenario:
            The parameter rasters are read independently of the GIS inputs, so they can
            disagree with the grid. A parameter cube one row short must raise.
        """
        # Trimming a row keeps the parameter *width*, so `ParameterSet` still accepts it --
        # its rule is the count per cell. Covering the grid is the flow network's business,
        # which is what `_check_parameters_cover_grid` is for.
        coello_loaded.parameters = coello_loaded.parameters.with_values(
            coello_loaded.parameters.values[:-1, :, :]
        )
        lake = _LakeStub(coello_loaded.meteo.time_steps)

        with pytest.raises(ValueError, match="as many rows as the catchment grid"):
            Run.run_maxbas_with_lake(coello_loaded, lake)

        assert "run_maxbas_with_lake" not in spied_wrapper, (
            "the wrapper must not run on mis-shaped parameters"
        )


class TestFloodModelHonoursKinematic:
    """`run_flood` reads `routing_method` to decide whether Muskingum skips the river cells."""

    @pytest.mark.parametrize(
        "declared, expected_skip",
        [("Kinematic", True), ("Muskingum", False)],
    )
    def test_the_skip_is_derived_from_the_declared_routing(
        self, coello_loaded: Catchment, monkeypatch, declared: str, expected_skip: bool
    ):
        """Test that the kinematic-wave declaration reaches the routing as a real argument.

        Test scenario:
            `"Kinematic"` means the wave model routes the river cells, so the Muskingum pass
            must leave them alone. That used to be a `routing_method != "Muskingum"` compare
            *inside* the routing loop, which is why it also fired on plain distributed runs.
            It is now read once, by this entry point, and passed down explicitly.

        Args:
            declared: The routing method the catchment declares.
            expected_skip: Whether the wrapper should be told to skip hydraulic cells.
        """
        _load_flat_river_geometry(coello_loaded)
        coello_loaded.routing_method = declared
        seen: dict = {}

        def _spy(run):
            seen["skip"] = run.skip_hydraulic_cells

        monkeypatch.setattr(run_module.Wrapper, "run_muskingum", staticmethod(_spy))

        Run.run_flood(coello_loaded)

        assert seen["skip"] is expected_skip, (
            f"routing_method={declared!r} should give skip_hydraulic_cells="
            f"{expected_skip}, got {seen['skip']}"
        )

    def test_an_explicit_argument_overrides_the_declaration(
        self, coello_loaded: Catchment, monkeypatch
    ):
        """Test that a caller can ask for the skip without declaring Kinematic.

        Test scenario:
            Deriving from `routing_method` keeps the historical spelling working, but the
            request is a run-time choice, so it stays expressible directly.
        """
        _load_flat_river_geometry(coello_loaded)
        coello_loaded.routing_method = "Muskingum"
        seen: dict = {}

        def _spy(run):
            seen["skip"] = run.skip_hydraulic_cells

        monkeypatch.setattr(run_module.Wrapper, "run_muskingum", staticmethod(_spy))

        Run.run_flood(coello_loaded, skip_hydraulic_cells=True)

        assert seen["skip"] is True, (
            "an explicit argument must win over the declaration"
        )

    def test_a_derived_kinematic_skip_warns_that_the_river_cells_go_unrouted(
        self, coello_loaded: Catchment, monkeypatch
    ):
        """Test that inferring the skip from `routing_method` says what it costs.

        Test scenario:
            Skipping the river cells is the handoff the flood model was built around -- a 1D
            hydraulic model routes them instead -- but that model left this package for
            Serapis, so nothing here picks them up and their discharge is simply absent.
            Someone who set `routing_method="Kinematic"` without a downstream model gets a
            number that is not a hydrograph, so the derived case has to say so.
        """
        _load_flat_river_geometry(coello_loaded)
        coello_loaded.routing_method = "Kinematic"
        monkeypatch.setattr(
            run_module.Wrapper, "run_muskingum", staticmethod(lambda *a, **k: None)
        )

        with pytest.warns(UserWarning, match="unrouted") as record:
            Run.run_flood(coello_loaded)

        message = str(record[0].message)
        assert "skip_hydraulic_cells=True" in message, (
            f"the warning should name the way to silence it, got: {message}"
        )
        # The geometry helper marks every cell as a river cell, and the Coello grid has 89
        # inside the flow-accumulation mask -- so the count must be the domain, not rows x
        # cols, which is what counting the raster alone reported.
        inside = int(
            np.count_nonzero(~np.isnan(coello_loaded.flow_network.flow_acc_arr))
        )
        assert f"{inside} river cells of {inside}" in message, (
            f"the counts must be taken inside the catchment, got: {message}"
        )

    def test_an_explicit_skip_does_not_warn(
        self, coello_loaded: Catchment, monkeypatch
    ):
        """Test that asking for the skip outright is left quiet.

        Test scenario:
            `skip_hydraulic_cells=True` is a statement that something downstream takes the
            river cells. That is the supported workflow, so it must not nag every run; the
            warning exists for the case nobody chose.
        """
        _load_flat_river_geometry(coello_loaded)
        coello_loaded.routing_method = "Kinematic"
        monkeypatch.setattr(
            run_module.Wrapper, "run_muskingum", staticmethod(lambda *a, **k: None)
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Run.run_flood(coello_loaded, skip_hydraulic_cells=True)

        unrouted = [w for w in caught if "unrouted" in str(w.message)]
        assert not unrouted, (
            f"an explicit skip must not warn, got: {[str(w.message) for w in unrouted]}"
        )
