"""Tests for the YAML run-configuration schema and the model it builds.

`hapi.config` is pure data: pydantic models that validate a parsed YAML mapping. The rules that
matter are the cross-field ones, because which blocks a configuration needs depends on
`catchment.spatial_resolution` and on `meteo.source` -- a distributed run needs a routing
network and all three drivers, a lumped one needs the single averaged-driver CSV. Those rules
exist so `Catchment.from_yaml` can consume a validated config without re-checking anything, so
they are pinned here per rule rather than in aggregate.

The second half covers `Catchment.from_yaml` itself: that it makes the `read_*` calls in the
order the build-then-mutate pattern requires, that it skips the two optional blocks when they
are absent, and that it builds `cls` so the subclasses return their own type.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from hapi.calibration import Calibration
from hapi.catchment import Catchment
from hapi.config import (
    CatchmentConfig,
    ConceptualModelConfig,
    FlowNetworkConfig,
    GaugesConfig,
    MeteoConfig,
    OutputsConfig,
    ParametersConfig,
    RunConfig,
)
from hapi.run import Run

COMBINED_NC = "tests/rrm/data/coello/meteo.nc"


@pytest.fixture(scope="function")
def distributed_mapping(
    coello_start_date: str,
    coello_end_date: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_muskingum: str,
    coello_cat_area: int,
    coello_initial_cond: list,
    coello_gauges_table: str,
    coello_gauges_path: str,
) -> dict:
    """A complete distributed configuration, as a plain mapping.

    Returns:
        dict: A mapping that validates, for tests to mutate one field at a time.
    """
    return {
        "catchment": {
            "name": "Coello",
            "start": coello_start_date,
            "end": coello_end_date,
            "spatial_resolution": "distributed",
        },
        "meteo": {
            "source": "netcdf",
            "path": COMBINED_NC,
            "precipitation": "precipitation",
            "temperature": "temperature",
            "evapotranspiration": "evapotranspiration",
        },
        "conceptual_model": {
            "model_class": "HBVBergestrom92",
            "catchment_area": coello_cat_area,
            "initial_condition": coello_initial_cond,
        },
        "parameters": {"path": coello_dist_parameters_muskingum, "snow": False},
        "gauges": {"table": coello_gauges_table, "discharge": coello_gauges_path},
        "flow_network": {
            "flow_accumulation": coello_acc_path,
            "flow_direction": coello_fd_path,
        },
    }


@pytest.fixture(scope="function")
def lumped_mapping(
    coello_start_date: str,
    coello_end_date: str,
    lumped_meteo_data_path: str,
    lumped_parameters_path: str,
    lumped_gauges_path: str,
) -> dict:
    """A complete lumped configuration, as a plain mapping.

    Points at the bundled lumped fixtures, so the mapping both validates and builds.

    Returns:
        dict: A mapping that validates and can be handed to `from_yaml`.
    """
    return {
        "catchment": {
            "name": "Coello",
            "start": coello_start_date,
            "end": coello_end_date,
            "spatial_resolution": "lumped",
        },
        "meteo": {"path": lumped_meteo_data_path},
        "conceptual_model": {
            "model_class": "HBVBergestrom92",
            "catchment_area": 1530,
            "initial_condition": [0, 10, 10, 10, 0],
        },
        "parameters": {"path": lumped_parameters_path},
        "gauges": {"discharge": lumped_gauges_path},
    }


def write_yaml(mapping: dict, tmp_path) -> str:
    """Dump a mapping to a YAML file.

    Args:
        mapping: The configuration to write.
        tmp_path: pytest temporary directory.

    Returns:
        str: Path to the written file.
    """
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(path)


class TestCatchmentConfig:
    """Tests for the `catchment` block."""

    def test_defaults_fill_the_optional_fields(self):
        """Test that only name and the two dates are required.

        Test scenario:
            The remaining fields describe the common case -- a daily lumped run parsed with
            ISO dates -- so a caller should not have to restate them.
        """
        config = CatchmentConfig(name="Coello", start="2009-01-01", end="2009-01-10")

        assert config.fmt == "%Y-%m-%d", f"unexpected default fmt: {config.fmt}"
        assert config.spatial_resolution == "lumped", (
            f"expected 'lumped' by default, got {config.spatial_resolution}"
        )
        assert config.temporal_resolution == "daily", (
            f"expected 'daily' by default, got {config.temporal_resolution}"
        )
        assert config.routing_method == "muskingum", (
            f"expected 'muskingum' by default, got {config.routing_method}"
        )

    @pytest.mark.parametrize(
        "field, value, allowed",
        [
            ("spatial_resolution", "semi", "'lumped' or 'distributed'"),
            ("temporal_resolution", "weekly", "'daily' or 'hourly'"),
            ("routing_method", "kinematic", "'muskingum' or 'maxbas'"),
        ],
        ids=["spatial", "temporal", "routing"],
    )
    def test_an_unknown_value_names_the_accepted_ones(self, field, value, allowed):
        """Test that each enumerated field rejects an unknown value and lists the valid set.

        Args:
            field: The field under test.
            value: An unrecognised value for it.
            allowed: The wording the error is expected to carry.

        Test scenario:
            These three select whole code paths downstream, so a typo has to fail at parse
            time naming what was expected, rather than selecting a branch by accident.
        """
        kwargs = {
            "name": "Coello",
            "start": "2009-01-01",
            "end": "2009-01-10",
            field: value,
        }

        with pytest.raises(ValidationError, match="Input should be") as exc:
            CatchmentConfig(**kwargs)

        assert allowed in str(exc.value), (
            f"the error should list the accepted values {allowed}: {exc.value}"
        )

    def test_dates_stay_strings(self):
        """Test that a date is kept as text rather than coerced to a date object.

        Test scenario:
            `Catchment.__init__` parses the dates itself with `fmt`. If the schema coerced
            them, `strptime` would be handed a `date` and raise, which is exactly why the
            YAML quotes them.
        """
        config = CatchmentConfig(name="Coello", start="2009-01-01", end="2009-01-10")

        assert isinstance(config.start, str), (
            f"start should be str, got {type(config.start)}"
        )
        assert config.start == "2009-01-01", f"start was altered: {config.start}"


class TestMeteoConfig:
    """Tests for the `meteo` block."""

    def test_raster_defaults_match_the_reader(self):
        """Test that the raster-reading defaults are the ones `from_rasters` uses.

        Test scenario:
            A configuration that names three folders and nothing else must read them the way
            the loader would by default, or the YAML would silently change the date parsing.
        """
        config = MeteoConfig()

        assert config.source == "rasters", f"expected 'rasters', got {config.source}"
        assert config.glob == "*.tif", f"unexpected glob default: {config.glob}"
        assert config.regex_string == r"\d{4}.\d{2}.\d{2}", (
            f"unexpected regex default: {config.regex_string}"
        )
        assert config.file_name_data_fmt is None, (
            "the date format should be inferred by default"
        )

    def test_an_unknown_source_is_refused(self):
        """Test that `source` accepts only the three loaders that exist.

        Test scenario:
            `source` selects which `MeteoInputs` constructor runs, so an unknown value has no
            loader to dispatch to and must fail here rather than fall through.
        """
        with pytest.raises(ValidationError, match="Input should be") as exc:
            MeteoConfig(source="zarr")

        assert "'rasters', 'netcdf' or 'netcdf_files'" in str(exc.value), (
            f"the error should name the three loaders: {exc.value}"
        )


class TestConceptualModelConfig:
    """Tests for the `conceptual_model` block."""

    def test_initial_condition_must_hold_five_states(self):
        """Test that the state vector is required to be exactly five long.

        Test scenario:
            HBV indexes `[sp, sm, uz, lz, wc]` by position, so a shorter vector does not
            raise where it is built -- it reads the wrong state, or runs off the end deep in
            the per-cell loop.
        """
        with pytest.raises(ValidationError, match="at least 5 items") as exc:
            ConceptualModelConfig(
                model_class="HBVBergestrom92",
                catchment_area=1530,
                initial_condition=[0, 5, 5, 5],
            )

        assert "initial_condition" in str(exc.value), (
            f"the error should name the field: {exc.value}"
        )

    def test_catchment_area_must_be_positive(self):
        """Test that a non-positive catchment area is refused.

        Test scenario:
            The area scales depth to discharge, so zero or a negative value produces a
            hydrograph that is silently zero or sign-flipped rather than an error.
        """
        with pytest.raises(ValidationError, match="greater than 0"):
            ConceptualModelConfig(
                model_class="HBVBergestrom92",
                catchment_area=-5,
                initial_condition=[0, 5, 5, 5, 0],
            )

    def test_model_class_keeps_its_name(self):
        """Test that the `model_class` field survives pydantic's protected namespace.

        Test scenario:
            Pydantic reserves the `model_` prefix. The YAML key is `model_class`, so the
            namespace is cleared rather than the field renamed -- this pins that it stayed.
        """
        config = ConceptualModelConfig(
            model_class="HBVBergestrom92",
            catchment_area=1530,
            initial_condition=[0, 5, 5, 5, 0],
        )

        assert config.model_class == "HBVBergestrom92", (
            f"model_class did not round-trip: {config.model_class}"
        )
        assert config.q_init is None, "q_init should default to None"


class TestStrictKeys:
    """Tests for the `extra=forbid` setting shared by every block."""

    @pytest.mark.parametrize(
        "model, kwargs",
        [
            (
                CatchmentConfig,
                {"name": "c", "start": "2009-01-01", "end": "2009-01-10"},
            ),
            (MeteoConfig, {}),
            (FlowNetworkConfig, {"flow_accumulation": "acc.tif"}),
            (ParametersConfig, {"path": "p"}),
            (GaugesConfig, {"discharge": "q.csv"}),
            (OutputsConfig, {}),
        ],
        ids=["catchment", "meteo", "flow_network", "parameters", "gauges", "outputs"],
    )
    def test_an_unknown_key_is_refused(self, model, kwargs):
        """Test that every block rejects a key it does not define.

        Args:
            model: The block under test.
            kwargs: The minimum valid arguments for it.

        Test scenario:
            A misspelled key would otherwise be dropped silently, and the input it was meant
            to supply would go missing far from the typo that caused it.
        """
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            model(**{**kwargs, "definitely_not_a_field": 1})


class TestRunConfigCrossFieldRules:
    """Tests for the rules tying the blocks to `catchment.spatial_resolution`."""

    def test_a_complete_distributed_configuration_validates(self, distributed_mapping):
        """Test that the fixture itself is accepted, so the negative cases mean something.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            Every rejection test below mutates one field of this mapping, so it has to be
            valid to begin with or those tests would pass for the wrong reason.
        """
        config = RunConfig.model_validate(distributed_mapping)

        assert config.catchment.spatial_resolution == "distributed"
        assert config.flow_network is not None, "the flow network should be parsed"
        assert config.gauges is not None, "the gauges block should be parsed"

    def test_a_complete_lumped_configuration_validates(self, lumped_mapping):
        """Test that a lumped configuration with no grid blocks is accepted.

        Args:
            lumped_mapping: A complete lumped configuration.

        Test scenario:
            The lumped shape is the other half of the schema: one CSV of averaged drivers,
            no flow network, and no gauge table.
        """
        config = RunConfig.model_validate(lumped_mapping)

        assert config.flow_network is None, "a lumped run carries no flow network"
        assert config.meteo.path == lumped_mapping["meteo"]["path"], (
            f"the lumped meteo CSV was not kept: {config.meteo.path}"
        )
        assert config.gauges.table is None, "a lumped run needs no gauge table"

    def test_distributed_requires_a_flow_network(self, distributed_mapping):
        """Test that a distributed run without a routing network is refused.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            The network supplies the grid every cube is checked against, so without it the
            run fails later on a shape mismatch that says nothing about the missing block.
        """
        del distributed_mapping["flow_network"]

        with pytest.raises(ValidationError, match="needs a flow_network block") as exc:
            RunConfig.model_validate(distributed_mapping)

        assert "distributed" in str(exc.value), (
            f"the error should name the resolution that requires it: {exc.value}"
        )

    def test_muskingum_requires_a_flow_direction_raster(self, distributed_mapping):
        """Test that a Muskingum run without a direction raster is refused at parse time.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            `flow_direction` is optional on the block because MAXBAS never reads one. Muskingum
            does, and it is the default routing method -- so a config copied from the MAXBAS
            example builds fine and then dereferences a None array inside the routing loop,
            after every raster has been read. The rule has to be stated here instead.
        """
        del distributed_mapping["flow_network"]["flow_direction"]

        with pytest.raises(ValidationError, match="flow_network.flow_direction is required"):
            RunConfig.model_validate(distributed_mapping)

    def test_maxbas_does_not_require_a_flow_direction_raster(self, distributed_mapping):
        """Test that the triangular path still accepts a network without a direction raster.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            The counterpart to the rule above: MAXBAS routes every cell straight to the outlet,
            so requiring the raster there would reject the configuration the shipped MAXBAS
            example uses.
        """
        distributed_mapping["catchment"]["routing_method"] = "maxbas"
        distributed_mapping["parameters"]["maxbas"] = True
        del distributed_mapping["flow_network"]["flow_direction"]

        config = RunConfig.model_validate(distributed_mapping)

        assert config.flow_network.flow_direction is None, (
            "MAXBAS should be allowed to omit the direction raster"
        )

    @pytest.mark.parametrize(
        "routing_method, maxbas",
        [("maxbas", False), ("muskingum", True)],
        ids=["maxbas-run-muskingum-set", "muskingum-run-maxbas-set"],
    )
    def test_the_routing_method_and_the_parameter_set_must_agree(
        self, distributed_mapping, routing_method, maxbas
    ):
        """Test that a routing method mismatched to its parameter set is refused.

        Args:
            distributed_mapping: A complete distributed configuration.
            routing_method: The routing method declared on the catchment.
            maxbas: The flag declared on the parameter set, deliberately disagreeing.

        Test scenario:
            The parameter-count check cannot catch this: a MAXBAS set holds 11 parameters and
            a Muskingum set 12, and `parameters.maxbas` selects which count is expected, so a
            disagreeing pair counts correctly. The run then completes and reads the Muskingum
            X as the MAXBAS value -- a hydrograph that is quietly wrong, the worst failure
            mode for a modelling tool.
        """
        distributed_mapping["catchment"]["routing_method"] = routing_method
        distributed_mapping["parameters"]["maxbas"] = maxbas
        if routing_method == "maxbas":
            del distributed_mapping["flow_network"]["flow_direction"]

        with pytest.raises(ValidationError, match="must agree"):
            RunConfig.model_validate(distributed_mapping)

    def test_a_configuration_without_parameters_skips_the_routing_cross_check(
        self, distributed_mapping
    ):
        """Test that the cross-check does not fire when no parameter set is configured.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            A calibration declares no `parameters` block -- its parameters come from the
            bounds -- so there is nothing to disagree with the routing method, and requiring
            agreement would reject every calibration configuration.
        """
        del distributed_mapping["parameters"]

        config = RunConfig.model_validate(distributed_mapping)

        assert config.parameters is None

    def test_distributed_requires_a_gauge_table_when_gauges_are_given(
        self, distributed_mapping
    ):
        """Test that a distributed `gauges` block without a table is refused.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            The table is what locates each gauge on the grid. Supplying discharge without it
            is a half-configured block, distinct from omitting gauges altogether.
        """
        del distributed_mapping["gauges"]["table"]

        with pytest.raises(ValidationError, match="needs gauges.table"):
            RunConfig.model_validate(distributed_mapping)

    @pytest.mark.parametrize(
        "driver", ["precipitation", "temperature", "evapotranspiration"]
    )
    def test_distributed_requires_all_three_drivers(self, distributed_mapping, driver):
        """Test that each missing driver is reported by name.

        Args:
            distributed_mapping: A complete distributed configuration.
            driver: The driver removed from the meteo block.

        Test scenario:
            The conceptual model reads all three every step, so a configuration missing one
            cannot run -- and the error has to say which, since the three are interchangeable
            in shape.
        """
        del distributed_mapping["meteo"][driver]

        with pytest.raises(ValidationError, match="missing") as exc:
            RunConfig.model_validate(distributed_mapping)

        assert driver in str(exc.value), (
            f"the error should name the missing driver {driver}: {exc.value}"
        )

    def test_netcdf_source_requires_a_path(self, distributed_mapping):
        """Test that `source: netcdf` without `meteo.path` is refused.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            For this source the three driver fields are variable *names* inside one file, so
            without the file there is nothing to read them from.
        """
        del distributed_mapping["meteo"]["path"]

        with pytest.raises(ValidationError, match="needs meteo.path"):
            RunConfig.model_validate(distributed_mapping)

    def test_lumped_requires_the_meteo_csv(self, lumped_mapping):
        """Test that a lumped run without `meteo.path` is refused.

        Args:
            lumped_mapping: A complete lumped configuration.

        Test scenario:
            Lumped mode has no grid to fall back on: `read_lumped_inputs` needs the single
            CSV of catchment-average drivers, and nothing else supplies them.
        """
        del lumped_mapping["meteo"]["path"]

        with pytest.raises(ValidationError, match="needs meteo.path"):
            RunConfig.model_validate(lumped_mapping)

    def test_parameters_and_gauges_may_both_be_omitted(self, distributed_mapping):
        """Test that a configuration carrying neither optional block still validates.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            A calibration derives its parameters from the bounds it is given rather than
            reading a fitted set, and a run that is not scored against observations has no
            gauges. Both blocks are therefore optional.
        """
        del distributed_mapping["parameters"]
        del distributed_mapping["gauges"]

        config = RunConfig.model_validate(distributed_mapping)

        assert config.parameters is None, "parameters should be absent, not defaulted"
        assert config.gauges is None, "gauges should be absent, not defaulted"


class TestRoutingMethodNormalisation:
    """Tests for the `routing_method` canonicalisation in `Catchment.__init__`."""

    @pytest.mark.parametrize(
        "given, stored",
        [
            ("muskingum", "Muskingum"),
            ("Muskingum", "Muskingum"),
            ("MUSKINGUM", "Muskingum"),
            ("maxbas", "MAXBAS"),
            ("MAXBAS", "MAXBAS"),
            ("kinematic", "Kinematic"),
        ],
    )
    def test_any_casing_is_stored_canonically(self, given, stored):
        """Test that the constructor stores one spelling whatever casing it is handed.

        Args:
            given: The spelling passed to the constructor.
            stored: The canonical spelling expected on the model.

        Test scenario:
            `distrrm.SpatialRouting` compares `routing_method != "Muskingum"` exactly, and its
            false branch reads `bankfull_depth`, which is None outside the flood model. A
            lower-case "muskingum" stored verbatim therefore routed every cell down the MAXBAS
            branch and raised `TypeError: 'NoneType' object is not subscriptable`.
        """
        model = Catchment(
            "coello", "2009-01-01", "2009-01-10", routing_method=given
        )

        assert model.routing_method == stored, (
            f"{given!r} should be stored as {stored!r}, got {model.routing_method!r}"
        )

    def test_kinematic_is_accepted_for_the_flood_model(self):
        """Test that the flood model's routing method is still a legal value.

        Test scenario:
            `Run.RunFloodModel` relies on the same `!= "Muskingum"` comparison to skip cells
            with a real `bankfull_depth`, so "Kinematic" is a working value and must not be
            rejected by the new validation.
        """
        model = Catchment(
            "coello", "2009-01-01", "2009-01-10", routing_method="Kinematic"
        )

        assert model.routing_method == "Kinematic"

    def test_an_unknown_routing_method_is_refused(self):
        """Test that an unrecognised routing method fails at construction.

        Test scenario:
            Any unknown spelling silently selects the non-Muskingum branch downstream, so it
            has to be caught here rather than surface as a `TypeError` on `bankfull_depth`.
        """
        with pytest.raises(ValueError, match="available routing methods") as exc:
            Catchment("coello", "2009-01-01", "2009-01-10", routing_method="diffusive")

        assert "diffusive" in str(exc.value), (
            f"the error should echo the value given: {exc.value}"
        )


class TestCatchmentFromYaml:
    """Tests for `Catchment.from_yaml`, which turns a configuration into a built model."""

    def test_a_distributed_configuration_populates_every_input(
        self, distributed_mapping, tmp_path, coello_cat_area, coello_initial_cond
    ):
        """Test that the builder makes each `read_*` call the configuration asks for.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.
            coello_cat_area: Expected catchment area.
            coello_initial_cond: Expected initial state.

        Test scenario:
            This is the whole point of the alternate constructor: the attributes a
            hand-written script assembled by calling the readers in order must all be
            populated by one call.
        """
        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.name == "Coello", f"name not set: {model.name}"
        assert model.spatial_resolution == "distributed"
        assert model.meteo is not None, "meteo was not assigned"
        assert model.flow_network is not None, "flow_network was not assigned"
        assert model.parameters is not None, "parameters were not read"
        assert model.lumped_model is not None, "the conceptual model was not read"
        assert model.GaugesTable is not None, "the gauge table was not read"
        assert model.QGauges is not None, "the discharge was not read"
        assert model.area == coello_cat_area, f"area not set: {model.area}"
        assert model.initial_cond == coello_initial_cond, (
            f"initial condition not set: {model.initial_cond}"
        )

    def test_the_drivers_cover_the_model_period(self, distributed_mapping, tmp_path):
        """Test that the meteo window defaults to the catchment's own dates.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The drivers pair with the model's date index by position, so a file spanning a
            longer record has to be trimmed to the run. Omitting `meteo.start` / `meteo.end`
            should fall back to the catchment block rather than read the file whole.
        """
        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.meteo.time_steps == len(model.date_index), (
            f"drivers hold {model.meteo.time_steps} steps, model spans "
            f"{len(model.date_index)}"
        )
        assert model.meteo.time[0] == model.date_index[0], (
            f"drivers start at {model.meteo.time[0]}, model at {model.date_index[0]}"
        )

    def test_an_explicit_meteo_window_overrides_the_catchment_dates(
        self, distributed_mapping, tmp_path
    ):
        """Test that `meteo.start` / `meteo.end` win when given.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The fallback is a convenience, not a rule: a caller who states the window
            explicitly is asking for that slice of the record.
        """
        distributed_mapping["meteo"]["start"] = "2009-01-03"
        distributed_mapping["meteo"]["end"] = "2009-01-07"

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.meteo.time_steps == 5, (
            f"03 to 07 inclusive is five steps, got {model.meteo.time_steps}"
        )

    def test_omitting_parameters_leaves_them_unread(
        self, distributed_mapping, tmp_path
    ):
        """Test that no parameter set is read when the block is absent.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The calibration shape. Reading a fitted set here would overwrite what the
            optimiser is about to supply, so the builder must skip the call entirely.
        """
        del distributed_mapping["parameters"]

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.parameters is None, (
            f"parameters should be unread, got {type(model.parameters)}"
        )
        assert model.meteo is not None, "the rest of the build should still have run"

    def test_omitting_gauges_leaves_them_unread(self, distributed_mapping, tmp_path):
        """Test that no gauge data is read when the block is absent.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            A run that is only inspected, never scored, carries no observations -- and the
            gauge readers would otherwise fail on a path that was never configured.
        """
        del distributed_mapping["gauges"]

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.GaugesTable is None, "no gauge table should have been read"
        assert model.QGauges is None, "no discharge should have been read"

    def test_the_routing_label_is_the_literal_the_router_compares(
        self, distributed_mapping, tmp_path
    ):
        """Test that lower-case `muskingum` reaches the model as `"Muskingum"`.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            `distrrm.SpatialRouting` tests `routing_method != "Muskingum"` case-sensitively,
            and `Catchment.__init__` stores whatever it is given verbatim. A lower-case
            spelling would send every cell down the MAXBAS branch and read `bankfull_depth`,
            which is None outside the flood model.
        """
        distributed_mapping["catchment"]["routing_method"] = "muskingum"

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.routing_method == "Muskingum", (
            f"the router compares against 'Muskingum' exactly, got {model.routing_method!r}"
        )

    def test_an_unregistered_model_class_is_refused_by_name(
        self, distributed_mapping, tmp_path
    ):
        """Test that a conceptual model the registry does not know is rejected.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The YAML names the model as a string, so the builder has to resolve it. An
            unknown name must say what is available rather than fail on a None class later.
        """
        distributed_mapping["conceptual_model"]["model_class"] = "HBV97"

        with pytest.raises(ValueError, match="not.*registered") as exc:
            Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert "HBVBergestrom92" in str(exc.value), (
            f"the error should list the known models: {exc.value}"
        )

    @pytest.mark.parametrize("cls", [Catchment, Calibration])
    def test_the_builder_returns_the_class_it_was_called_on(
        self, distributed_mapping, tmp_path, cls
    ):
        """Test that a subclass taking the same constructor arguments builds its own type.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.
            cls: The class the classmethod is called on.

        Test scenario:
            `Calibration` extends `Catchment` with the same constructor signature, so
            `Calibration.from_yaml(...)` should hand back a `Calibration` the calibration
            methods can be called on, not a bare `Catchment`.
        """
        model = cls.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert isinstance(model, cls), (
            f"expected a {cls.__name__}, got {type(model).__name__}"
        )

    def test_run_cannot_be_built_because_it_takes_no_constructor_arguments(
        self, distributed_mapping, tmp_path
    ):
        """Test that `Run.from_yaml` fails loudly rather than building something unusable.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            `Run` inherits the classmethod but overrides `__init__` to take only `self`, and
            its entry points are called unbound on a catchment (`Run.RunHapi(model)`). Pins
            that the mismatch surfaces as a `TypeError` at the constructor rather than as a
            half-built model, so the docstring's warning stays true.
        """
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Run.from_yaml(write_yaml(distributed_mapping, tmp_path))

    def test_a_lumped_configuration_reads_the_averaged_driver_csv(
        self, lumped_mapping, tmp_path
    ):
        """Test that the lumped branch reads one CSV instead of a grid.

        Args:
            lumped_mapping: A complete lumped configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            Lumped mode is the other half of the builder: `read_lumped_inputs` fills `data`
            from a single file of catchment-average drivers, and no `MeteoInputs` grid or
            flow network is built at all.
        """
        model = Catchment.from_yaml(write_yaml(lumped_mapping, tmp_path))

        assert model.spatial_resolution == "lumped"
        assert model.data is not None, "the averaged drivers were not read"
        assert model.meteo is None, "a lumped run should build no driver grid"
        assert model.flow_network is None, "a lumped run should build no flow network"
        assert model.parameters is not None, "the lumped parameter file was not read"

    def test_a_lumped_run_reads_discharge_without_a_gauge_table(
        self, lumped_mapping, tmp_path
    ):
        """Test that lumped gauges are read from one file, with no table lookup.

        Args:
            lumped_mapping: A complete lumped configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The gauge table exists to locate gauges on a grid, which lumped mode has none
            of, so the builder must skip `read_gauge_table` and still read the discharge.
        """
        model = Catchment.from_yaml(write_yaml(lumped_mapping, tmp_path))

        assert model.QGauges is not None, "the observed discharge was not read"
        assert model.GaugesTable is None, "a lumped run should read no gauge table"

    def test_the_inherited_window_is_parsed_with_the_catchment_format(
        self, distributed_mapping, tmp_path
    ):
        """Test that fallback dates are read with `catchment.fmt`, not `meteo.fmt`.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            `catchment.fmt` and `meteo.fmt` are independent fields. When the meteo block
            states no window it inherits the catchment's dates, which are written in the
            catchment's format -- so parsing them with the meteo format either fails on a
            date the user never wrote that way, or, between two mutually parseable layouts,
            silently windows the drivers to the wrong period.
        """
        distributed_mapping["catchment"]["fmt"] = "%d/%m/%Y"
        distributed_mapping["catchment"]["start"] = "01/01/2009"
        distributed_mapping["catchment"]["end"] = "10/01/2009"

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.meteo.time_steps == len(model.date_index), (
            f"drivers hold {model.meteo.time_steps} steps, model spans "
            f"{len(model.date_index)}"
        )
        assert model.meteo.time[0] == model.date_index[0], (
            f"window start {model.meteo.time[0]} does not match the model's "
            f"{model.date_index[0]}"
        )

    def test_a_stated_meteo_window_uses_the_meteo_format(
        self, distributed_mapping, tmp_path
    ):
        """Test that a window the block states is parsed with the block's own format.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The other half of the rule: `meteo.fmt` governs `meteo.start` / `meteo.end`, so a
            block may describe its window in a different layout from the catchment's without
            either being reinterpreted.
        """
        distributed_mapping["meteo"]["fmt"] = "%d/%m/%Y"
        distributed_mapping["meteo"]["start"] = "03/01/2009"
        distributed_mapping["meteo"]["end"] = "07/01/2009"

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.meteo.time_steps == 5, (
            f"03 to 07 January inclusive is five steps, got {model.meteo.time_steps}"
        )

    def test_a_non_ascii_name_survives_the_read(self, distributed_mapping, tmp_path):
        """Test that the configuration is decoded as UTF-8 whatever the platform default is.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            Without an explicit encoding the file is decoded with the locale codec, so a
            non-ASCII name mojibakes on a machine that does not default to UTF-8 -- silently,
            because the corrupted text is still valid YAML. The name reaches result filenames
            and plot titles, and the same risk applies to every path field.
        """
        distributed_mapping["catchment"]["name"] = "Río Coello"
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(distributed_mapping, allow_unicode=True), encoding="utf-8"
        )

        model = Catchment.from_yaml(str(path))

        assert model.name == "Río Coello", f"the name was corrupted on read: {model.name!r}"

    def test_the_configuration_stays_reachable_on_the_model(
        self, distributed_mapping, tmp_path
    ):
        """Test that blocks the build does not consume survive on `model.config`.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            `outputs` describes where results go, so nothing in the build reads it. Discarding
            the config would leave it parsed but unreachable, forcing a caller to restate in
            Python a path the file already gives -- which is what the shipped example used to
            do.
        """
        distributed_mapping["outputs"] = {"results_dir": "somewhere/else/"}

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.config is not None, "the configuration should be kept on the model"
        assert model.config.outputs.results_dir == "somewhere/else/", (
            f"outputs did not survive: {model.config.outputs}"
        )
        assert model.config.flow_network.flow_accumulation is not None, (
            "the flow-accumulation path should stay reachable for save_results"
        )

    def test_a_hand_built_model_has_no_configuration(self, coello_start_date, coello_end_date):
        """Test that `config` is None on a model that was not built from a file.

        Args:
            coello_start_date: Simulation start date.
            coello_end_date: Simulation end date.

        Test scenario:
            The attribute has to be safe to check on any catchment, so a hand-assembled one
            reports no configuration rather than raising `AttributeError`.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)

        assert model.config is None, f"expected no configuration, got {model.config}"

    def test_an_invalid_configuration_fails_before_anything_is_read(
        self, distributed_mapping, tmp_path
    ):
        """Test that validation runs ahead of the first reader.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            Reading rasters is the slow part of a build. A configuration that cannot work
            should be rejected on the parsed mapping, not after minutes of I/O.
        """
        distributed_mapping["catchment"]["spatial_resolution"] = "semi"

        with pytest.raises(ValidationError, match="Input should be"):
            Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

    def test_the_configuration_is_read_from_the_given_path(
        self, distributed_mapping, tmp_path
    ):
        """Test that two different files build two differently-named models.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            Guards against the path being ignored in favour of something cached or
            hard-coded -- the name has to come from the file that was named.
        """
        first = write_yaml(distributed_mapping, tmp_path)
        renamed = copy.deepcopy(distributed_mapping)
        renamed["catchment"]["name"] = "Elsewhere"
        second = tmp_path / "other.yaml"
        second.write_text(yaml.safe_dump(renamed), encoding="utf-8")

        assert Catchment.from_yaml(first).name == "Coello"
        assert Catchment.from_yaml(str(second)).name == "Elsewhere"
