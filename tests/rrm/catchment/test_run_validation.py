"""Tests for the input validation `Run`'s lake and flood entry points perform.

These three entry points (`RunFloodModel`, `runHAPIwithLake`, `RunFW1withLake`) each open with a
block of dimension checks that now read through `flow_network` and `meteo` rather than through
attributes on the catchment. The wrapper each one dispatches to is replaced by a spy, so the tests
pin the validation and the dispatch without running a lake or a hydraulic model.
"""

from __future__ import annotations

import numpy as np
import pytest

from hapi import run as run_module
from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


class _LakeStub:
    """Stand-in for `Lake` carrying only the `MeteoData` the validation reads."""

    def __init__(self, time_steps: int, columns: int = 3):
        self.MeteoData = np.ones((time_steps, columns))


@pytest.fixture(scope="function")
def spied_wrapper(monkeypatch) -> dict:
    """Replace every `Wrapper` entry point with a spy that records its call.

    Args:
        monkeypatch: Used to swap the wrapper methods out of the run module.

    Returns:
        dict: Maps the wrapper name to the positional arguments it was called with.
    """
    calls: dict = {}

    def _make(name: str):
        def _spy(*args):
            calls[name] = args

        return _spy

    for name in ("RRMModel", "RRMWithlake", "FW1Withlake"):
        monkeypatch.setattr(run_module.Wrapper, name, staticmethod(_make(name)))
    return calls


@pytest.fixture(scope="function")
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
    model.bankfull_depth = np.full(shape, 2.0)
    model.river_width = np.full(shape, 10.0)
    model.river_roughness = np.full(shape, 0.03)
    model.flood_plain_roughness = np.full(shape, 0.06)


class TestRunFloodModel:
    """Tests for `Run.RunFloodModel`."""

    def test_dispatches_once_every_input_lines_up(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that a consistent model reaches the RRM wrapper.

        Test scenario:
            The flood entry point checks the flow-direction grid, the meteo cubes, the
            parameter array and the four river-geometry rasters before dispatching. With all
            of them on the catchment grid it must call `Wrapper.RRMModel` with the model.
        """
        _load_flat_river_geometry(coello_loaded)

        Run.RunFloodModel(coello_loaded)

        assert "RRMModel" in spied_wrapper, "RRMModel should have been dispatched"
        assert spied_wrapper["RRMModel"][0] is coello_loaded, (
            "the wrapper must receive the model itself"
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
        _load_flat_river_geometry(coello_loaded)
        coello_loaded.river_width = coello_loaded.river_width[:-1, :]

        with pytest.raises(AssertionError, match="number of rows"):
            Run.RunFloodModel(coello_loaded)

        assert "RRMModel" not in spied_wrapper, (
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
            Run.RunFloodModel(coello_loaded)

        assert "RRMModel" not in spied_wrapper, (
            "the wrapper must not run on inconsistent meteo inputs"
        )


class TestRunHapiWithLake:
    """Tests for `Run.runHAPIwithLake`."""

    def test_dispatches_once_the_lake_record_matches_the_simulation(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that a lake covering the simulation reaches the lake wrapper.

        Test scenario:
            The lake is a lumped inflow whose own meteorological record must run step for
            step with the distributed cubes. A matching record must dispatch to
            `Wrapper.RRMWithlake` with both the model and the lake.
        """
        lake = _LakeStub(coello_loaded.meteo.time_steps)

        Run.runHAPIwithLake(coello_loaded, lake)

        assert "RRMWithlake" in spied_wrapper, "RRMWithlake should have been dispatched"
        assert spied_wrapper["RRMWithlake"] == (coello_loaded, lake), (
            "the wrapper must receive the model and the lake"
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

        with pytest.raises(AssertionError, match="same length"):
            Run.runHAPIwithLake(coello_loaded, lake)

        assert "RRMWithlake" not in spied_wrapper, (
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

        with pytest.raises(AssertionError, match="three columns"):
            Run.runHAPIwithLake(coello_loaded, lake)

    def test_rejects_a_flow_direction_grid_of_the_wrong_shape(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that the flow-direction grid is checked against the network's own shape.

        Test scenario:
            `FlowNetwork` sizes itself from the accumulation raster, so a flow-direction
            array of a different shape means the two rasters do not describe the same
            catchment and the routing table would be indexed out of range.
        """
        coello_loaded.flow_network.flow_dir_arr = (
            coello_loaded.flow_network.flow_dir_arr[:-1, :]
        )
        lake = _LakeStub(coello_loaded.meteo.time_steps)

        with pytest.raises(AssertionError, match="rows and columns"):
            Run.runHAPIwithLake(coello_loaded, lake)


class TestRunFW1WithLake:
    """Tests for `Run.RunFW1withLake`."""

    def test_dispatches_once_the_lake_record_matches_the_simulation(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that the triangular-routing lake path validates and dispatches.

        Test scenario:
            The FW1 lake path skips the flow-direction check — triangular routing needs no
            direction grid — but keeps the meteo, parameter and lake checks. A consistent
            model must reach `Wrapper.FW1Withlake`.
        """
        lake = _LakeStub(coello_loaded.meteo.time_steps)

        Run.RunFW1withLake(coello_loaded, lake)

        assert "FW1Withlake" in spied_wrapper, "FW1Withlake should have been dispatched"
        assert spied_wrapper["FW1Withlake"] == (coello_loaded, lake), (
            "the wrapper must receive the model and the lake"
        )

    def test_rejects_parameters_off_the_catchment_grid(
        self, coello_loaded: Catchment, spied_wrapper: dict
    ):
        """Test that the parameter array is checked against the flow network.

        Test scenario:
            The parameter rasters are read independently of the GIS inputs, so they can
            disagree with the grid. A parameter cube one row short must raise.
        """
        coello_loaded.parameters = coello_loaded.parameters[:-1, :, :]
        lake = _LakeStub(coello_loaded.meteo.time_steps)

        with pytest.raises(AssertionError, match="as many rows as the catchment grid"):
            Run.RunFW1withLake(coello_loaded, lake)

        assert "FW1Withlake" not in spied_wrapper, (
            "the wrapper must not run on mis-shaped parameters"
        )
