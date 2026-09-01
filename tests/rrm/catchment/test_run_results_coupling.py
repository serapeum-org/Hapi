"""Tests for how `Run` and `Catchment` are coupled, and for the results object between them.

`Run` used to subclass `Catchment` and be called unbound (`Run.RunHapi(model)`, with the
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
    "Qtot",
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
            "RunHapi",
            "RunFloodModel",
            "runHAPIwithLake",
            "runFW1",
            "RunFW1withLake",
            "runLumped",
            "from_yaml",
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

    def test_from_yaml_refuses_and_names_the_pattern(self):
        """Test that `Run.from_yaml` explains itself rather than raising AttributeError.

        Test scenario:
            `Run` no longer inherits `Catchment.from_yaml`, so the call would fail with a
            bare AttributeError. The explicit refusal is kept because it names what to do.
        """
        with pytest.raises(TypeError, match="cannot be built from a configuration"):
            Run.from_yaml("anything.yaml")


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

        results = Run.RunHapi(model)

        assert isinstance(results, SimulationResults), (
            f"RunHapi must return SimulationResults, got {type(results).__name__}"
        )
        assert results is model.results, (
            "the returned results must be the same object assigned to model.results"
        )
        assert results.routing is RoutingKind.MUSKINGUM, (
            f"a RunHapi run is Muskingum-routed, got {results.routing}"
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

        results = Run.runFW1(model)

        assert results.routing is RoutingKind.MAXBAS, (
            f"a runFW1 run is MAXBAS-routed, got {results.routing}"
        )
        assert not results.outlet_shortcut_valid, (
            "MAXBAS sends every cell to the outlet, so the outlet-cell shortcut is invalid"
        )


class TestResultAttributesAreReadOnlyViews:
    """The historical attribute names still read, but the results object owns the arrays."""

    @pytest.mark.parametrize("field", RESULT_FIELDS)
    def test_field_reads_through_to_the_results_object(
        self, coello_fixtures: dict, coello_dist_parameters_muskingum: str, field: str
    ):
        """Test that each historical name returns exactly what the results object holds.

        Test scenario:
            Existing scripts and notebooks read `model.Qtot` and friends after a run. The
            properties exist so that keeps working; identity is asserted rather than
            equality so a copy cannot pass.

        Args:
            field: The result attribute being checked.
        """
        model = _build("coello", coello_dist_parameters_muskingum, **coello_fixtures)
        Run.RunHapi(model)

        assert getattr(model, field) is getattr(model.results, field), (
            f"model.{field} must read through to results.{field}, not copy it"
        )

    @pytest.mark.parametrize("field", RESULT_FIELDS)
    def test_field_is_none_before_a_run(self, field: str):
        """Test that the result names read as None on a catchment that has not run.

        Test scenario:
            They used to be None-initialised attributes. Reading one before a run must stay
            a None rather than becoming an AttributeError.

        Args:
            field: The result attribute being checked.
        """
        model = Catchment("empty", "2009-01-01", "2009-01-10")

        assert getattr(model, field) is None, (
            f"model.{field} must be None before a run, got {type(getattr(model, field))}"
        )

    @pytest.mark.parametrize("field", RESULT_FIELDS)
    def test_field_cannot_be_assigned(self, field: str):
        """Test that a result field rejects assignment.

        Test scenario:
            These are outputs. A run that can be half-overwritten by hand is exactly what
            the results object exists to prevent, so the properties have no setter and
            staging a state goes through `model.results` instead.

        Args:
            field: The result attribute being checked.
        """
        model = Catchment("empty", "2009-01-01", "2009-01-10")

        with pytest.raises(AttributeError):
            setattr(model, field, np.zeros((2, 2, 2)))


class TestRoutingProvenanceReplacesTheFlag:
    """`_maxbas_routed` is derived from the results, so it cannot outlive the run."""

    def test_a_muskingum_run_after_a_maxbas_run_clears_the_maxbas_reading(
        self,
        coello_fixtures: dict,
        coello_dist_parameters_maxbas: str,
        coello_dist_parameters_muskingum: str,
    ):
        """Test that running MAXBAS then Muskingum leaves the model reading as Muskingum.

        Test scenario:
            This is the case the old boolean needed hand-clearing for: `_maxbas_routed` was
            set by the MAXBAS path and had to be reset by every Muskingum path, with a
            comment saying so. Deriving it from the results makes that impossible to forget,
            because a new run replaces the object the reading comes from.
        """
        model = _build(
            "coello", coello_dist_parameters_maxbas, maxbas=True, **coello_fixtures
        )
        Run.runFW1(model)
        assert model._maxbas_routed, "the FW1 run should read as MAXBAS-routed"

        # Re-read the parameters the Muskingum path needs, then run it on the same model.
        model.read_parameters(coello_dist_parameters_muskingum, False)
        Run.RunHapi(model)

        assert not model._maxbas_routed, (
            "after a Muskingum run the model must no longer read as MAXBAS-routed"
        )
        assert model.results.outlet_shortcut_valid, (
            "the outlet-cell shortcut is valid again once Muskingum has routed the results"
        )

    def test_a_fresh_catchment_does_not_read_as_maxbas_routed(self):
        """Test that a model that has never run does not claim MAXBAS routing.

        Test scenario:
            The derived reading has to answer for the no-results case too, since
            `extract_discharge` consults it before checking anything else.
        """
        model = Catchment("empty", "2009-01-01", "2009-01-10")

        assert not model._maxbas_routed, (
            "a catchment with no results must not read as MAXBAS-routed"
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
            Run.RunHapi(model)

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
            Run.RunFloodModel(model)

        assert "bankfull_depth" in str(exc_info.value), (
            f"the error should name the missing rasters, got: {exc_info.value}"
        )
