"""Tests for how `Run` and `Catchment` are coupled, and for the results object between them.

`Run` used to subclass `Catchment` and be called unbound (`Run.run_distributed(model)`, with the
catchment landing on `self`), writing nine result arrays back onto the model plus a private
`_maxbas_routed` flag recording which routing had produced them. It is now a namespace of
static entry points that state what they need as a protocol and return a
`SimulationResults`.

These tests pin the three properties that change buys: the entry points do not depend on
`Catchment`, the results are one object rather than nine attributes, and the routing
provenance travels with the arrays instead of as a flag that the next run has to clear.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import numpy as np
import pytest

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.results import RoutingKind, SimulationResults
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run

DATE_REGEX = r"\d{4}.\d{2}.\d{2}"

RESULT_FIELDS = (
    "quz",
    "qlz",
    "state_variables",
    "quz_routed",
    "qlz_translated",
    "q_total",
    "qout",
)


def _build(name: str, parameters: str, **fixtures) -> Catchment:
    """Assemble a distributed Coello catchment ready to run.

    Args:
        name: Catchment name.
        parameters: Path to the parameter folder, which decides maxbas vs muskingum.
        **fixtures: The `coello_*` fixture values the build reads.

    Returns:
        Catchment: A model with meteo, flow network, parameters and conceptual model set.
    """
    model = Catchment(
        name,
        fixtures["start"],
        fixtures["end"],
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    model.meteo = MeteoInputs.from_rasters(
        fixtures["prec"],
        fixtures["temp"],
        fixtures["evap"],
        start=fixtures["start"],
        end=fixtures["end"],
        regex_string=DATE_REGEX,
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    model.flow_network = FlowNetwork.from_rasters(fixtures["acc"], fixtures.get("fd"))
    model.read_parameters(parameters, False, maxbas=fixtures.get("maxbas", False))
    model.read_lumped_model(HBVLumped, fixtures["area"], fixtures["initial_cond"])
    return model


@pytest.fixture
def coello_fixtures(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> dict:
    """Bundle the `coello_*` fixtures the builder reads, so each test names one thing."""
    return {
        "start": coello_start_date,
        "end": coello_end_date,
        "prec": coello_prec_path,
        "temp": coello_temp_path,
        "evap": coello_evap_path,
        "acc": coello_acc_path,
        "fd": coello_fd_path,
        "area": coello_cat_area,
        "initial_cond": coello_initial_cond,
    }


class TestRunIsNotACatchment:
    """The coupling itself: `Run` no longer inherits from or imports `Catchment`."""

    def test_run_does_not_subclass_catchment(self):
        """Test that `Run` is a plain namespace rather than a `Catchment` subclass.

        Test scenario:
            The inheritance existed only so a catchment could be passed as `self`. Nothing
            ever instantiated `Run` or checked `isinstance(x, Run)`, so the IS-A was never
            true; asserting it is gone stops it coming back.
        """
        assert not issubclass(Run, Catchment), (
            "Run must not subclass Catchment; it takes the model as an explicit parameter"
        )
        assert Catchment not in Run.__mro__, (
            f"Catchment must not be in Run's MRO, got {Run.__mro__}"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "run_distributed",
            "run_flood",
            "run_distributed_with_lake",
            "run_maxbas",
            "run_maxbas_with_lake",
            "run_lumped",
        ],
    )
    def test_every_entry_point_is_a_static_method(self, name: str):
        """Test that each entry point is a staticmethod taking the model explicitly.

        Test scenario:
            An instance method would reintroduce the unbound-call pattern, where the first
            parameter is named `self` but is really the model. A staticmethod cannot.

        Args:
            name: The entry point being checked.
        """
        attribute = inspect.getattr_static(Run, name)
        assert isinstance(attribute, staticmethod), (
            f"Run.{name} must be a staticmethod, got {type(attribute).__name__}"
        )
        first = next(iter(inspect.signature(getattr(Run, name)).parameters))
        assert first != "self", (
            f"Run.{name}'s first parameter must not be named self, got {first!r}"
        )

    def test_run_does_not_import_catchment_at_runtime(self):
        """Test that importing `hapi.run` does not pull in `hapi.catchment`.

        Test scenario:
            The dependency is inverted: `Run` owns protocols describing what it needs, and
            `Catchment` satisfies them structurally. A runtime import would mean the
            inversion is only cosmetic. Checked in a subprocess rather than by clearing
            `sys.modules` here, which would leave every later test importing a second copy
            of the package.
        """
        probe = (
            "import sys; import hapi.run; "
            "print('catchment-imported' if 'hapi.catchment' in sys.modules else 'clean')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )

        assert completed.stdout.strip() == "clean", (
            "importing hapi.run must not import hapi.catchment; the protocols in "
            "hapi.protocols exist so the run layer does not depend on the concrete class"
        )

    def test_run_offers_nothing_a_configuration_could_build(self):
        """Test that `Run` exposes no constructor-like surface.

        Test scenario:
            The old inheritance made `Run.from_yaml` resolve to `Catchment.from_yaml`, which
            had to be overridden to refuse. With the inheritance gone the name is absent,
            and `Run` carries only the entry points.
        """
        assert not hasattr(Run, "from_yaml"), (
            "Run must not offer from_yaml; build a Catchment and pass it to an entry point"
        )
        public = {n for n in vars(Run) if not n.startswith("_")}
        assert public == {
            "run_distributed",
            "run_distributed_with_lake",
            "run_maxbas",
            "run_maxbas_with_lake",
            "run_lumped",
            "run_flood",
        }, f"Run should carry only the six entry points, got {sorted(public)}"


class TestEntryPointsReturnTheirResults:
    """Entry points return a `SimulationResults` rather than only mutating the model."""

    def test_run_hapi_returns_the_object_it_put_on_the_model(
        self, coello_fixtures: dict, coello_dist_parameters_muskingum: str
    ):
        """Test that the returned results are the same object assigned to `model.results`.

        Test scenario:
            Returning the results is what lets a caller work without reaching back into the
            model. It must be the same object, not a copy, so the historical attribute reads
            and the returned value can never disagree.
        """
        model = _build("coello", coello_dist_parameters_muskingum, **coello_fixtures)

        results = Run.run_distributed(model)

        assert isinstance(results, SimulationResults), (
            f"run_distributed must return SimulationResults, got {type(results).__name__}"
        )
        assert results is model.results, (
            "the returned results must be the same object assigned to model.results"
        )
        assert results.routing is RoutingKind.MUSKINGUM, (
            f"a run_distributed run is Muskingum-routed, got {results.routing}"
        )

    def test_run_fw1_returns_maxbas_routed_results(
        self, coello_fixtures: dict, coello_dist_parameters_maxbas: str
    ):
        """Test that the triangular path records MAXBAS on the results it returns.

        Test scenario:
            The routing kind is what tells `extract_discharge` whether the outlet-cell
            shortcut is valid, so the FW1 path must record it on the arrays it produced.
        """
        model = _build(
            "coello", coello_dist_parameters_maxbas, maxbas=True, **coello_fixtures
        )

        results = Run.run_maxbas(model)

        assert results.routing is RoutingKind.MAXBAS, (
            f"a run_maxbas run is MAXBAS-routed, got {results.routing}"
        )
        assert not results.outlet_shortcut_valid, (
            "MAXBAS sends every cell to the outlet, so the outlet-cell shortcut is invalid"
        )


class TestResultsAreTheOnlyHomeForTheArrays:
    """The catchment carries no result attributes; `results` is where they live."""

    @pytest.mark.parametrize("field", RESULT_FIELDS)
    def test_the_catchment_does_not_carry_the_field(self, field: str):
        """Test that a result name is absent from the catchment entirely.

        Test scenario:
            These were nullable attributes on `Catchment`, then briefly properties
            forwarding to `results`. Both are gone: there is one home for the arrays, so a
            reader cannot pick the stale one by habit.

        Args:
            field: The result field being checked.
        """
        model = Catchment("empty", "2009-01-01", "2009-01-10")

        assert not hasattr(model, field), (
            f"Catchment must not carry {field}; read it as model.results.{field}"
        )

    @pytest.mark.parametrize("field", RESULT_FIELDS)
    def test_the_results_object_carries_the_field_after_a_run(
        self, coello_fixtures: dict, coello_dist_parameters_muskingum: str, field: str
    ):
        """Test that a completed Muskingum run populates every result field.

        Test scenario:
            The Muskingum path fills all of them except `qout`, which needs the gauge table
            and so is left for `extract_discharge`. Everything else must be an array.

        Args:
            field: The result field being checked.
        """
        model = _build("coello", coello_dist_parameters_muskingum, **coello_fixtures)
        results = Run.run_distributed(model)

        value = getattr(results, field)
        if field == "qout":
            assert value is None, (
                "the Muskingum path leaves qout for extract_discharge to fill"
            )
        else:
            assert isinstance(value, np.ndarray), (
                f"results.{field} must be an array after a run, got {type(value).__name__}"
            )

    def test_results_is_none_before_a_run(self):
        """Test that a catchment that has not run has no results at all.

        Test scenario:
            One `None` to check instead of nine, which is the point: a model either has a
            finished run behind it or it does not.
        """
        model = Catchment("empty", "2009-01-01", "2009-01-10")

        assert model.results is None, (
            f"a catchment that has not run must have results=None, got {model.results}"
        )


class TestRoutingProvenanceTravelsWithTheArrays:
    """Routing is a field of the results, so it cannot outlive the run that set it."""

    def test_a_muskingum_run_after_a_maxbas_run_reads_as_muskingum(
        self,
        coello_fixtures: dict,
        coello_dist_parameters_maxbas: str,
        coello_dist_parameters_muskingum: str,
    ):
        """Test that running MAXBAS then Muskingum leaves the model reading as Muskingum.

        Test scenario:
            This is the case the old boolean needed hand-clearing for: it was set by the
            MAXBAS path and had to be reset by every Muskingum path, with a comment saying
            so. Carrying the routing on the results makes that impossible to forget, because
            a new run replaces the object the reading comes from.
        """
        model = _build(
            "coello", coello_dist_parameters_maxbas, maxbas=True, **coello_fixtures
        )
        maxbas_results = Run.run_maxbas(model)
        assert maxbas_results.routing is RoutingKind.MAXBAS, (
            "the MAXBAS run should record its own routing"
        )

        # Re-read the parameters the Muskingum path needs, then run it on the same model.
        model.read_parameters(coello_dist_parameters_muskingum, False)
        muskingum_results = Run.run_distributed(model)

        assert muskingum_results is not maxbas_results, (
            "a second run must build a new results object, not overwrite fields in place"
        )
        assert model.results.routing is RoutingKind.MUSKINGUM, (
            f"after a Muskingum run the model must read as Muskingum, got "
            f"{model.results.routing}"
        )
        assert model.results.outlet_shortcut_valid, (
            "the outlet-cell shortcut is valid again once Muskingum has routed the results"
        )


class TestValidationNamesWhatIsMissing:
    """The guards the protocols exposed: an optional input that this path does require."""

    def test_a_muskingum_run_without_a_flow_direction_raster_says_so(
        self, coello_fixtures: dict, coello_dist_parameters_muskingum: str
    ):
        """Test that cell-to-cell routing without a flow-direction raster raises clearly.

        Test scenario:
            `FlowNetwork` takes the direction raster as optional because MAXBAS never reads
            it, so a Muskingum run can be assembled without one. It used to fail on None
            inside the validation; it now names the missing raster.
        """
        fixtures = dict(coello_fixtures, fd=None)
        model = _build("coello", coello_dist_parameters_muskingum, **fixtures)

        with pytest.raises(ValueError, match="flow-direction raster"):
            Run.run_distributed(model)

    def test_the_flood_model_names_the_river_geometry_it_lacks(
        self, coello_fixtures: dict, coello_dist_parameters_muskingum: str
    ):
        """Test that the flood model reports which geometry rasters are unset.

        Test scenario:
            `read_river_geometry` sets four arrays together, so a missing one means it was
            never called. The check used to reach `np.shape(None)`; it now lists the names.
        """
        model = _build("coello", coello_dist_parameters_muskingum, **coello_fixtures)

        with pytest.raises(ValueError, match="read_river_geometry") as exc_info:
            Run.run_flood(model)

        assert "bankfull_depth" in str(exc_info.value), (
            f"the error should name the missing rasters, got: {exc_info.value}"
        )


class TestRoutingIsChosenByTheEntryPoint:
    """`routing_method` records intent; the entry point is what routes."""

    @pytest.mark.parametrize("declared", ["Muskingum", "MAXBAS"])
    def test_run_distributed_routes_with_muskingum_whatever_the_field_says(
        self,
        coello_fixtures: dict,
        coello_dist_parameters_muskingum: str,
        declared: str,
    ):
        """Test that `routing_method` does not change what `run_distributed` does.

        Test scenario:
            The field used to be compared inside the routing loop, so a catchment built with
            anything but `"Muskingum"` reached `bankfull_depth[x, y]` -- None outside the
            flood model -- and died with `TypeError: 'NoneType' object is not subscriptable`
            partway through routing. Skipping river cells is now an explicit argument to the
            flood entry point, so the declared method cannot reach the loop at all.

        Args:
            declared: The routing method the catchment is built with.
        """
        model = _build("coello", coello_dist_parameters_muskingum, **coello_fixtures)
        model.routing_method = declared

        results = Run.run_distributed(model)

        assert results.routing is RoutingKind.MUSKINGUM, (
            f"run_distributed must route with Muskingum whatever routing_method says; "
            f"declared {declared!r}, got {results.routing}"
        )
        assert results.q_total is not None, "every cell must have been routed"

    def test_a_kinematic_catchment_still_runs_distributed_without_crashing(
        self, coello_fixtures: dict, coello_dist_parameters_muskingum: str
    ):
        """Test that declaring Kinematic no longer breaks a plain distributed run.

        Test scenario:
            `"Kinematic"` is a real routing method -- the wave model the flood path applies
            to river cells -- and stays accepted. But the routing loop used to compare
            against it directly, so a catchment declaring it and calling `run_distributed`
            reached `bankfull_depth[x, y]`, which is None outside the flood model, and died
            with `TypeError: 'NoneType' object is not subscriptable` partway through routing.
            The skip is read by `run_flood` now, so `run_distributed` routes every cell.
        """
        model = _build("coello", coello_dist_parameters_muskingum, **coello_fixtures)
        model.routing_method = "Kinematic"

        results = Run.run_distributed(model)

        assert results.routing is RoutingKind.MUSKINGUM, (
            "run_distributed routes with Muskingum regardless of the declared method"
        )
        assert results.q_total is not None, (
            "every cell must be routed; the flood skip belongs to run_flood"
        )
