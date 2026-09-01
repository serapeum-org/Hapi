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
import datetime as dt
import os
from pathlib import Path

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
from hapi.inputs import MeteoInputs
from hapi.run import Run

COMBINED_NC = "tests/rrm/data/coello/meteo.nc"


@pytest.fixture
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


@pytest.fixture
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


PATH_FIELDS = (
    ("meteo", "path"),
    ("parameters", "path"),
    ("gauges", "table"),
    ("gauges", "discharge"),
    ("flow_network", "flow_accumulation"),
    ("flow_network", "flow_direction"),
)


def write_yaml(mapping: dict, tmp_path) -> str:
    """Dump a mapping to a YAML file, with its input paths made absolute.

    The fixtures name their inputs relative to the repository root, but `from_yaml` resolves a
    relative path against the configuration's own directory -- and these configurations are
    written to a temporary one. Absolutising here keeps each test about the field it is
    exercising; the resolution rule itself is covered by its own test.

    Args:
        mapping: The configuration to write.
        tmp_path: pytest temporary directory.

    Returns:
        str: Path to the written file.
    """
    mapping = copy.deepcopy(mapping)
    for block, field in PATH_FIELDS:
        value = mapping.get(block, {}).get(field)
        if isinstance(value, str) and not Path(value).is_absolute():
            mapping[block][field] = str(Path(value).resolve())

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

    @pytest.mark.parametrize(
        "value, fmt, expected",
        [
            (dt.date(2009, 1, 1), "%Y-%m-%d", "2009-01-01"),
            (dt.date(2009, 1, 1), "%d/%m/%Y", "01/01/2009"),
            (dt.datetime(2009, 1, 1, 6, 0), "%Y-%m-%d", "2009-01-01"),
        ],
        ids=["date", "custom-fmt", "timestamp"],
    )
    def test_a_date_yaml_already_parsed_is_written_back_in_fmt(
        self, value, fmt, expected
    ):
        """Test that an unquoted YAML date is accepted and rendered in the block's format.

        Args:
            value: What YAML hands pydantic for an unquoted date or timestamp.
            fmt: The block's date format.
            expected: The string the field should end up holding.

        Test scenario:
            `start: 2009-01-01` written without quotes is a `datetime.date` by the time the
            schema sees it, and the field is a string because the constructor parses it with
            `fmt`. Rejecting it would reject the spelling a YAML author writes first, over a
            difference that carries no information.
        """
        config = CatchmentConfig(name="Coello", start=value, end=value, fmt=fmt)

        assert config.start == expected, f"expected {expected!r}, got {config.start!r}"
        assert isinstance(config.start, str), (
            f"the field must still hold a string, got {type(config.start)}"
        )

    def test_a_quoted_date_is_left_exactly_as_written(self):
        """Test that the normaliser does not touch a date that is already text.

        Test scenario:
            A string may be in any format the author declares, including ones no `date`
            round-trips through, so it has to pass through untouched.
        """
        config = CatchmentConfig(
            name="Coello", start="01/01/2009", end="31/12/2011", fmt="%d/%m/%Y"
        )

        assert config.start == "01/01/2009", f"start was rewritten: {config.start}"

    @pytest.mark.parametrize(
        "values, why",
        [
            ("not-a-mapping", "a scalar block"),
            (
                {
                    "name": "Coello",
                    "start": "2009-01-01",
                    "end": "2009-01-10",
                    "fmt": 5,
                },
                "a non-string fmt",
            ),
        ],
        ids=["scalar", "non-string-fmt"],
    )
    def test_the_normaliser_defers_to_pydantic_for_what_it_cannot_render(
        self, values, why
    ):
        """Test that unrenderable input is passed through for pydantic to report.

        Args:
            values: Raw input the normaliser cannot act on.
            why: What is wrong with it, for the failure message.

        Test scenario:
            The normaliser runs before validation, so it sees raw input. Raising its own
            error for a malformed block would replace pydantic's precise, field-located
            message with a vaguer one.
        """
        with pytest.raises(ValidationError):
            CatchmentConfig.model_validate(values)


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

        with pytest.raises(
            ValidationError, match="flow_network.flow_direction is required"
        ):
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

        with pytest.raises(
            ValidationError, match="all three meteorological drivers"
        ) as exc:
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

        with pytest.raises(ValidationError, match="meteo.path must be set"):
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

    @pytest.mark.parametrize(
        "block, field",
        [("catchment", "start"), ("catchment", "end"), ("meteo", "start")],
    )
    def test_a_date_that_does_not_match_its_format_is_refused(
        self, distributed_mapping, block, field
    ):
        """Test that every date is checked against the format it is written in.

        Args:
            distributed_mapping: A complete distributed configuration.
            block: The block carrying the date.
            field: The date field to corrupt.

        Test scenario:
            The module promises a validated config is consumable without re-checking, but an
            unparseable date slipped through to fail later inside `Catchment.__init__` -- or,
            for the meteo bounds, deep in the loader.
        """
        distributed_mapping[block][field] = "not-a-date"

        with pytest.raises(ValidationError, match="does not match its format") as exc:
            RunConfig.model_validate(distributed_mapping)

        assert f"{block}.{field}" in str(exc.value), (
            f"the error should name the offending field: {exc.value}"
        )

    def test_a_period_that_ends_before_it_starts_is_refused(self, distributed_mapping):
        """Test that a reversed period is caught at parse time.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            Nothing checked the order, so a reversed period produced an empty date index and
            failed far downstream on a shape mismatch that said nothing about the dates.
        """
        distributed_mapping["catchment"]["start"] = "2009-01-10"
        distributed_mapping["catchment"]["end"] = "2009-01-01"

        with pytest.raises(ValidationError, match="is after"):
            RunConfig.model_validate(distributed_mapping)

    def test_a_lumped_configuration_may_not_carry_a_flow_network(self, lumped_mapping):
        """Test that a grid block on a lumped run is refused rather than ignored.

        Args:
            lumped_mapping: A complete lumped configuration.

        Test scenario:
            `extra="forbid"` exists so a misspelled key fails loudly rather than being
            dropped. Silently discarding a correctly spelled but inapplicable block is the
            same silence by another route -- the user's block simply never took effect.
        """
        lumped_mapping["flow_network"] = {"flow_accumulation": "acc.tif"}

        with pytest.raises(ValidationError, match="has no grid"):
            RunConfig.model_validate(lumped_mapping)

    def test_a_lumped_configuration_may_not_name_a_grid_meteo_source(
        self, lumped_mapping
    ):
        """Test that a grid loader on a lumped run is refused.

        Args:
            lumped_mapping: A complete lumped configuration.

        Test scenario:
            Lumped mode reads `meteo.path` with `pd.read_csv` regardless of `source`, so a
            `source: netcdf` config would hand a `.nc` file to the CSV reader.
        """
        lumped_mapping["meteo"]["source"] = "netcdf"

        with pytest.raises(ValidationError, match="does not apply"):
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

    @pytest.mark.parametrize(
        "block, field, value",
        [
            ("gauges", "table", "gauges.csv"),
            ("gauges", "column", "name"),
            ("meteo", "precipitation", "prec"),
            ("meteo", "start", "2009-01-01"),
            ("meteo", "end", "2011-12-31"),
            ("meteo", "glob", "*.tif"),
        ],
        ids=["table", "column", "driver", "window-start", "window-end", "glob"],
    )
    def test_a_lumped_configuration_refuses_a_field_it_would_not_read(
        self, lumped_mapping, block, field, value
    ):
        """Test that each field a lumped run never reads is named and refused.

        Args:
            lumped_mapping: A complete lumped configuration.
            block: Which block the field lives in.
            field: The inapplicable field.
            value: A plausible value for it.

        Test scenario:
            A lumped run reads one CSV of catchment-average drivers and locates no gauges,
            so a grid driver, a reader knob, a window or a gauge table is a line with no
            effect. Dropping it silently is the failure `extra="forbid"` exists to prevent,
            arrived at by another route.
        """
        lumped_mapping[block][field] = value

        with pytest.raises(ValidationError, match="read by nothing") as exc:
            RunConfig.model_validate(lumped_mapping)

        assert f"{block}.{field}" in str(exc.value), (
            f"the error should name the offending field: {exc.value}"
        )

    @pytest.mark.parametrize(
        "source, patch, refused",
        [
            ("netcdf", {"path": "m.nc", "glob": "*.tif"}, "meteo.glob"),
            (
                "netcdf",
                {"path": "m.nc", "per_variable": {"p": {}}},
                "meteo.per_variable",
            ),
            ("netcdf", {"path": "m.nc", "variable": "pre"}, "meteo.variable"),
            ("netcdf_files", {"path": "m.nc"}, "meteo.path"),
            ("rasters", {"path": "m.nc"}, "meteo.path"),
            ("rasters", {"variable": "pre"}, "meteo.variable"),
        ],
        ids=[
            "netcdf-glob",
            "netcdf-per-variable",
            "netcdf-variable",
            "netcdf-files-path",
            "rasters-path",
            "rasters-variable",
        ],
    )
    def test_a_source_refuses_the_fields_its_own_branch_never_reads(
        self, distributed_mapping, source, patch, refused
    ):
        """Test that each `meteo.source` refuses the knobs belonging to the other two.

        Args:
            distributed_mapping: A complete distributed configuration.
            source: The source under test.
            patch: Fields to set on the `meteo` block, including the inapplicable one.
            refused: The field the error must name.

        Test scenario:
            The three branches of `MeteoInputs.from_config` read different fields: the
            raster reader takes `glob` and `per_variable` and no `path`, `netcdf` takes a
            `path` and no reader knobs, `netcdf_files` takes a `variable`. Setting one
            outside its source's set says something the run will not do.
        """
        distributed_mapping["meteo"]["source"] = source
        distributed_mapping["meteo"].update(patch)

        with pytest.raises(ValidationError, match="read by nothing") as exc:
            RunConfig.model_validate(distributed_mapping)

        assert refused in str(exc.value), (
            f"the error should name {refused}: {exc.value}"
        )

    def test_a_default_the_author_never_wrote_is_not_refused(self, lumped_mapping):
        """Test that only explicitly written fields count as inapplicable.

        Args:
            lumped_mapping: A complete lumped configuration.

        Test scenario:
            Every refused field has a default, so testing the value rather than whether it
            was set would reject every lumped configuration ever written -- `meteo.glob`
            alone defaults to `"*.tif"`. The check reads `model_fields_set`.
        """
        config = RunConfig.model_validate(lumped_mapping)

        assert config.meteo.glob == "*.tif", (
            f"the default should still be there, got {config.meteo.glob}"
        )

    @pytest.mark.parametrize(
        "maxbas, expected",
        [(True, "maxbas"), (False, "muskingum")],
        ids=["maxbas-set", "muskingum-set"],
    )
    def test_an_unstated_routing_method_is_derived_from_the_parameter_set(
        self, lumped_mapping, maxbas, expected
    ):
        """Test that `routing_method` follows `parameters.maxbas` when it is not written.

        Args:
            lumped_mapping: A complete lumped configuration.
            maxbas: What the parameter set carries.
            expected: The routing method that should be derived from it.

        Test scenario:
            The two describe the same choice from opposite sides. Left at its `muskingum`
            default, a MAXBAS run carried a `routing_method` contradicting what it does --
            and that attribute is what `distrrm.route_muskingum` keys off.
        """
        lumped_mapping["parameters"]["maxbas"] = maxbas
        assert "routing_method" not in lumped_mapping["catchment"], (
            "this test is about the unstated case"
        )

        config = RunConfig.model_validate(lumped_mapping)

        assert config.catchment.routing_method == expected, (
            f"expected {expected!r} derived from maxbas={maxbas}, got "
            f"{config.catchment.routing_method!r}"
        )

    def test_a_lumped_routing_method_must_still_agree_with_the_parameter_set(
        self, lumped_mapping
    ):
        """Test that the agreement check is no longer scoped to distributed runs.

        Args:
            lumped_mapping: A complete lumped configuration.

        Test scenario:
            The parameter-count check cannot catch this: the two counts differ by one and
            `maxbas` is what selects which is expected, so a contradicting pair still counts
            correctly and then reads the wrong parameter as the routing one.
        """
        lumped_mapping["catchment"]["routing_method"] = "muskingum"
        lumped_mapping["parameters"]["maxbas"] = True

        with pytest.raises(ValidationError, match="must agree"):
            RunConfig.model_validate(lumped_mapping)

    def test_the_derivation_runs_before_the_flow_direction_check(
        self, distributed_mapping
    ):
        """Test that a derived MAXBAS run may omit the flow-direction raster.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            MAXBAS sends every cell straight to the outlet and never reads a flow direction,
            but that requirement is keyed off `routing_method` -- so if the block checks ran
            before the derivation, the still-defaulted `muskingum` would demand a raster the
            run has no use for.
        """
        distributed_mapping["parameters"]["maxbas"] = True
        del distributed_mapping["flow_network"]["flow_direction"]
        distributed_mapping["catchment"].pop("routing_method", None)

        config = RunConfig.model_validate(distributed_mapping)

        assert config.catchment.routing_method == "maxbas", (
            f"expected the derived method, got {config.catchment.routing_method!r}"
        )

    @pytest.mark.parametrize(
        "window",
        [
            {"start": "2012-06-01"},
            {"end": "2008-01-01"},
            {"start": "2010-01-01", "end": "2009-01-01"},
        ],
        ids=[
            "start-after-catchment-end",
            "end-before-catchment-start",
            "both-inverted",
        ],
    )
    def test_the_resolved_meteorological_window_must_run_forwards(
        self, distributed_mapping, window
    ):
        """Test that the window the run will use is checked, not the two literal pairs.

        Args:
            distributed_mapping: A complete distributed configuration.
            window: A `meteo` window that inverts the effective period.

        Test scenario:
            `MeteoInputs.from_config` takes each bound from `meteo` when stated and falls
            back to `catchment` otherwise, so a block stating only one half can invert the
            effective window while neither pair is inverted on its own.
        """
        distributed_mapping["meteo"].update(window)

        with pytest.raises(ValidationError, match="ends before it starts"):
            RunConfig.model_validate(distributed_mapping)

    def test_a_meteo_window_inside_the_catchment_period_is_accepted(
        self, distributed_mapping
    ):
        """Test that a genuine sub-period is not caught by the window check.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            Narrowing the drivers to part of the catchment's period is the reason the two
            `meteo` bounds exist, so the stricter check must not cost that.
        """
        distributed_mapping["meteo"]["start"] = "2009-01-02"
        distributed_mapping["meteo"]["end"] = "2009-01-05"

        config = RunConfig.model_validate(distributed_mapping)

        assert config.meteo.start == "2009-01-02", (
            f"the stated window should survive: {config.meteo}"
        )

    def test_the_gauge_table_format_falls_back_to_the_discharge_format(
        self, distributed_mapping
    ):
        """Test that `table_fmt` defaults to `fmt` rather than to a format of its own.

        Args:
            distributed_mapping: A complete distributed configuration.

        Test scenario:
            The two parse different files -- the table's validity-period columns and the
            discharge CSV index -- but one hand usually writes both, so the common case
            should stay a single field.
        """
        distributed_mapping["gauges"]["fmt"] = "%d/%m/%Y"

        config = RunConfig.model_validate(distributed_mapping)

        assert config.gauges.table_fmt is None, (
            f"table_fmt should stay unset so the builder can fall back: "
            f"{config.gauges.table_fmt}"
        )
        assert config.gauges.fmt == "%d/%m/%Y", "the discharge format should be kept"

    def test_a_lumped_configuration_may_omit_gauges(self, lumped_mapping):
        """Test that a lumped run with no gauges block validates.

        Args:
            lumped_mapping: A complete lumped configuration.

        Test scenario:
            The inapplicable-field check for `gauges` only runs when the block is present, so
            a lumped run that is not scored against observations has to pass through it
            untouched rather than tripping on a block that is not there.
        """
        del lumped_mapping["gauges"]

        config = RunConfig.model_validate(lumped_mapping)

        assert config.gauges is None, "gauges should be absent, not defaulted"


class TestMeteoInputsFromConfig:
    """Tests for the `meteo.source` dispatch in `MeteoInputs.from_config`."""

    def test_the_raster_source_reads_the_three_folders(
        self,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_evap_path: str,
        coello_start_date: str,
        coello_end_date: str,
    ):
        """Test that `source: rasters` reads the folders and dates them.

        Args:
            coello_prec_path: Rainfall raster folder.
            coello_temp_path: Temperature raster folder.
            coello_evap_path: Evapotranspiration raster folder.
            coello_start_date: Simulation start date.
            coello_end_date: Simulation end date.

        Test scenario:
            This is the branch the shipped MAXBAS example depends on, and the one with the
            most argument plumbing -- seven forwarded keywords plus two conditional ones --
            so a mis-forwarded argument would be invisible until a user's file names stopped
            parsing.
        """
        meteo = MeteoInputs.from_config(
            MeteoConfig(
                source="rasters",
                precipitation=coello_prec_path,
                temperature=coello_temp_path,
                evapotranspiration=coello_evap_path,
                file_name_data_fmt="%Y.%m.%d",
            ),
            start=coello_start_date,
            end=coello_end_date,
        )

        assert meteo.shape == (13, 14, 10), (
            f"unexpected grid or step count: {meteo.shape}"
        )
        assert meteo.time is not None, "the calendar should come from the file names"
        assert meteo.time[0].strftime("%Y-%m-%d") == coello_start_date

    def test_the_raster_source_forwards_the_reader_arguments(
        self,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_evap_path: str,
    ):
        """Test that `glob`, `per_variable` and `gdal_env` reach the loader.

        Args:
            coello_prec_path: Rainfall raster folder.
            coello_temp_path: Temperature raster folder.
            coello_evap_path: Evapotranspiration raster folder.

        Test scenario:
            The conditional pass-throughs are the easiest to drop silently. A `glob` that
            matches nothing must surface as a read error, which proves it was forwarded
            rather than ignored.
        """
        config = MeteoConfig(
            source="rasters",
            precipitation=coello_prec_path,
            temperature=coello_temp_path,
            evapotranspiration=coello_evap_path,
            file_name_data_fmt="%Y.%m.%d",
            glob="*.nothing",
            gdal_env={"GDAL_PAM_ENABLED": "NO"},
            per_variable={"temperature": {"glob": "*.tif"}},
        )

        with pytest.raises((FileNotFoundError, ValueError)):
            MeteoInputs.from_config(config)

    def test_the_netcdf_files_source_reads_one_file_per_driver(
        self, coello_start_date: str, coello_end_date: str
    ):
        """Test that `source: netcdf_files` reads a separate file for each driver.

        Args:
            coello_start_date: Simulation start date.
            coello_end_date: Simulation end date.

        Test scenario:
            The third branch, and the only one whose driver fields are per-driver *paths*
            rather than folders or variable names -- so a branch that confused them would
            still find files but read the wrong ones.
        """
        meteo = MeteoInputs.from_config(
            MeteoConfig(
                source="netcdf_files",
                precipitation="tests/rrm/data/coello/prec.nc",
                temperature="tests/rrm/data/coello/temp.nc",
                evapotranspiration="tests/rrm/data/coello/evap.nc",
            ),
            start=coello_start_date,
            end=coello_end_date,
        )

        assert meteo.shape == (13, 14, 10), (
            f"unexpected grid or step count: {meteo.shape}"
        )

    def test_a_config_missing_a_driver_is_refused(self):
        """Test that the defensive guard fires for a hand-built config.

        Test scenario:
            `RunConfig` rejects such a configuration, so this only fires for a `MeteoConfig`
            built directly -- which is exactly when the message naming the missing drivers is
            the only thing the caller has to go on.
        """
        config = MeteoConfig(source="rasters", precipitation="p")

        with pytest.raises(ValueError, match="all three meteorological drivers") as exc:
            MeteoInputs.from_config(config)

        assert "temperature" in str(exc.value), (
            f"the error should name what is unset: {exc.value}"
        )

    def test_the_netcdf_source_needs_a_path(self):
        """Test that `source: netcdf` without a path is refused.

        Test scenario:
            For this source the driver fields are variable names inside one file, so without
            the file there is nothing to read them from.
        """
        config = MeteoConfig(
            source="netcdf",
            precipitation="precipitation",
            temperature="temperature",
            evapotranspiration="evapotranspiration",
        )

        with pytest.raises(ValueError, match="meteo.path must be set"):
            MeteoInputs.from_config(config)

    def test_the_netcdf_source_reads_the_three_variables_from_one_file(
        self, coello_start_date: str, coello_end_date: str
    ):
        """Test that `source="netcdf"` builds a grid from one file's three variables.

        Args:
            coello_start_date: Simulation start date.
            coello_end_date: Simulation end date.

        Test scenario:
            The other two sources have direct success tests; this branch was covered only
            transitively through `from_yaml`, so a change to how `meteo.path` or the variable
            names are forwarded would have surfaced only there.
        """
        config = MeteoConfig(
            source="netcdf",
            path=COMBINED_NC,
            precipitation="precipitation",
            temperature="temperature",
            evapotranspiration="evapotranspiration",
        )

        meteo = MeteoInputs.from_config(
            config, start=coello_start_date, end="2009-01-10"
        )

        assert meteo.precipitation.shape == meteo.temperature.shape, (
            f"the three cubes must agree: {meteo.shape}"
        )
        assert meteo.time_steps == 10, (
            f"the window should hold ten steps, got {meteo.time_steps}"
        )


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
            `distrrm.route_muskingum` compares `routing_method != "Muskingum"` exactly, and its
            false branch reads `bankfull_depth`, which is None outside the flood model. A
            lower-case "muskingum" stored verbatim therefore routed every cell down the MAXBAS
            branch and raised `TypeError: 'NoneType' object is not subscriptable`.
        """
        model = Catchment("coello", "2009-01-01", "2009-01-10", routing_method=given)

        assert model.routing_method == stored, (
            f"{given!r} should be stored as {stored!r}, got {model.routing_method!r}"
        )

    def test_kinematic_is_accepted_for_the_flood_model(self):
        """Test that the kinematic-wave routing method stays a legal value.

        Test scenario:
            "Kinematic" names the routing the flood model applies to river cells. It is read
            by `Run.run_flood`, which uses it to decide whether the Muskingum pass leaves
            those cells to the hydraulic model -- so it carries meaning and must be accepted.
            What changed is only where it is read: no longer inside the routing loop, where
            it ran for every distributed model.
        """
        model = Catchment(
            "coello", "2009-01-01", "2009-01-10", routing_method="Kinematic"
        )

        assert model.routing_method == "Kinematic"

    def test_an_unknown_spatial_resolution_is_refused(self):
        """Test that the sibling enumerated argument is validated the same way.

        Test scenario:
            `spatial_resolution` selects which half of the build runs, so an unrecognised
            value has no branch to take and must fail at construction rather than silently
            choosing the lumped one.
        """
        with pytest.raises(ValueError, match="'lumped' and 'distributed'"):
            Catchment("coello", "2009-01-01", "2009-01-10", spatial_resolution="semi")

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

    @pytest.mark.parametrize(
        "argument",
        ["spatial_resolution", "temporal_resolution", "routing_method"],
    )
    def test_a_mode_argument_that_is_not_a_string_names_itself(self, argument):
        """Test that each mode argument reports its own name when handed a non-string.

        Args:
            argument: The constructor argument under test.

        Test scenario:
            All three are lower-cased, so a non-string used to reach `.lower()` and raise an
            `AttributeError` naming neither the argument nor the class. `Calibration` made
            that reachable through its own signature, which declared all three `str | None`.
        """
        with pytest.raises(TypeError, match=f"{argument} must be a string") as exc:
            Catchment("Coello", "2009-01-01", "2009-01-10", **{argument: None})

        assert "NoneType" in str(exc.value), (
            f"the error should name what it got: {exc.value}"
        )

    def test_calibration_accepts_the_same_three_arguments(self):
        """Test that the guard reaches `Calibration`, which passes the arguments down.

        Test scenario:
            `Calibration.__init__` forwards all three to `Catchment.__init__` unchanged, and
            its annotations used to advertise a `None` that crashed.
        """
        with pytest.raises(TypeError, match="routing_method must be a string"):
            Calibration("Coello", "2009-01-01", "2009-01-10", routing_method=None)


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
        assert model.parameters.values is not None, "parameters were not read"
        assert model.model_setup.model is not None, "the conceptual model was not read"
        assert model.GaugesTable is not None, "the gauge table was not read"
        assert model.QGauges is not None, "the discharge was not read"
        assert model.model_setup.area == coello_cat_area, (
            f"area not set: {model.model_setup.area}"
        )
        assert model.model_setup.initial_cond == coello_initial_cond, (
            f"initial condition not set: {model.model_setup.initial_cond}"
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

        assert model.meteo.time_steps == len(model.period.date_index), (
            f"drivers hold {model.meteo.time_steps} steps, model spans "
            f"{len(model.period.date_index)}"
        )
        assert model.meteo.time[0] == model.period.date_index[0], (
            f"drivers start at {model.meteo.time[0]}, model at {model.period.date_index[0]}"
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
            f"parameters should be unread, got {type(model.parameters.values)}"
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
            `distrrm.route_muskingum` tests `routing_method != "Muskingum"` case-sensitively,
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

        path = write_yaml(distributed_mapping, tmp_path)

        with pytest.raises(ValueError, match="not.*registered") as exc:
            Catchment.from_yaml(path)

        assert "HBVBergestrom92" in str(exc.value), (
            f"the error should list the known models: {exc.value}"
        )

    def test_an_unregistered_model_class_is_refused_before_the_readers_run(
        self, distributed_mapping, tmp_path
    ):
        """Test that the registry lookup happens before any input is read.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The lookup needs nothing but the config, so running it after the parameter folder
            read charged a typo the full cost of that I/O and left a partly-populated model
            behind. Pointing every input at a path that does not exist isolates the ordering:
            if the readers ran first the failure would name a missing file instead.
        """
        distributed_mapping["conceptual_model"]["model_class"] = "HBV97"
        distributed_mapping["parameters"]["path"] = "no/such/parameters"
        distributed_mapping["meteo"]["path"] = "no/such/meteo.nc"
        distributed_mapping["flow_network"]["flow_accumulation"] = "no/such/acc.tif"
        distributed_mapping["flow_network"]["flow_direction"] = "no/such/fd.tif"

        path = write_yaml(distributed_mapping, tmp_path)

        with pytest.raises(ValueError, match="not.*registered"):
            Catchment.from_yaml(path)

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

    def test_run_has_no_constructor_and_nothing_to_build_from_a_configuration(self):
        """Test that `Run` is a namespace of entry points, not something a config builds.

        Test scenario:
            `Run` used to inherit `Catchment.from_yaml` through the subclassing, so the call
            resolved and had to be overridden to refuse. It no longer inherits anything, so
            the name is simply absent -- which is the honest answer for a class that holds
            no state and models nothing.
        """
        assert not hasattr(Run, "from_yaml"), (
            "Run must not offer from_yaml; a configuration builds a Catchment, which is "
            "then passed to an entry point"
        )

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
        assert model.parameters.values is not None, (
            "the lumped parameter file was not read"
        )

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

        assert model.meteo.time_steps == len(model.period.date_index), (
            f"drivers hold {model.meteo.time_steps} steps, model spans "
            f"{len(model.period.date_index)}"
        )
        assert model.meteo.time[0] == model.period.date_index[0], (
            f"window start {model.meteo.time[0]} does not match the model's "
            f"{model.period.date_index[0]}"
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

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.name == "Río Coello", (
            f"the name was corrupted on read: {model.name!r}"
        )

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
        assert model.config.outputs.results_dir.replace("\\", "/").endswith(
            "somewhere/else"
        ), f"outputs did not survive: {model.config.outputs}"
        assert model.config.flow_network.flow_accumulation is not None, (
            "the flow-accumulation path should stay reachable for save_results"
        )

    def test_a_hand_built_model_has_no_configuration(
        self, coello_start_date, coello_end_date
    ):
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

    def test_relative_paths_resolve_against_the_configuration_file(
        self, distributed_mapping, tmp_path, monkeypatch
    ):
        """Test that a configuration runs from a working directory that is not its own.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.
            monkeypatch: Used to move the process out of the repository root.

        Test scenario:
            Resolving against the process's working directory would make a file valid only
            from the one place it happened to be written for. Rewriting the paths relative to
            the config and then running from elsewhere is the case that distinguishes the two.
        """
        repo_root = Path.cwd()
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        for block, field in (
            ("meteo", "path"),
            ("parameters", "path"),
            ("gauges", "table"),
            ("gauges", "discharge"),
        ):
            distributed_mapping[block][field] = os.path.relpath(
                repo_root / distributed_mapping[block][field], config_dir
            )
        for field in ("flow_accumulation", "flow_direction"):
            distributed_mapping["flow_network"][field] = os.path.relpath(
                repo_root / distributed_mapping["flow_network"][field], config_dir
            )
        path = config_dir / "config.yaml"
        path.write_text(yaml.safe_dump(distributed_mapping), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        model = Catchment.from_yaml(str(path))

        assert model.meteo is not None, (
            "the drivers should resolve from the config's own dir"
        )
        assert model.flow_network is not None, "the network should resolve too"

    def test_a_netcdf_variable_name_is_not_treated_as_a_path(
        self, distributed_mapping, tmp_path
    ):
        """Test that the driver fields survive path resolution when they name variables.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            Under `source: netcdf` the three driver fields are variable names inside
            `meteo.path`, not paths. Resolving them would turn `precipitation` into an
            absolute directory and the read would fail looking for a variable of that name.
        """
        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.config.meteo.precipitation == "precipitation", (
            f"the variable name was rewritten as a path: {model.config.meteo.precipitation}"
        )

    def test_a_path_object_is_accepted(self, distributed_mapping, tmp_path):
        """Test that the path may be a `Path`, not only a string.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The body immediately wraps the argument in `Path`, and the rest of the package
            annotates such arguments `str | Path`, so a caller holding a `Path` should not
            have to stringify it.
        """
        write_yaml(distributed_mapping, tmp_path)

        model = Catchment.from_yaml(tmp_path / "config.yaml")

        assert model.name == "Coello", (
            f"the config was not read from the Path: {model.name}"
        )

    def test_an_empty_file_names_itself(self, tmp_path):
        """Test that an empty configuration file is reported as such.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            An empty file parses to `None`, which pydantic reports as "Input should be a valid
            dictionary" without naming the file -- unhelpful when a run names several.
        """
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="is empty") as exc:
            Catchment.from_yaml(str(path))

        assert "empty.yaml" in str(exc.value), (
            f"the error should name the file: {exc.value}"
        )

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

        path = write_yaml(distributed_mapping, tmp_path)

        with pytest.raises(ValidationError, match="Input should be"):
            Catchment.from_yaml(path)

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
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        second = write_yaml(renamed, other_dir)

        assert Catchment.from_yaml(first).name == "Coello"
        assert Catchment.from_yaml(second).name == "Elsewhere"

    def test_a_missing_file_names_the_path(self, tmp_path):
        """Test that a configuration path that does not exist raises `FileNotFoundError`.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            Documented in `Raises` but untested. The path is opened directly, so the error
            comes from `read_text` and carries the name.
        """
        missing = tmp_path / "not-here.yaml"

        with pytest.raises(FileNotFoundError):
            Catchment.from_yaml(str(missing))

    def test_malformed_yaml_is_reported_as_malformed(self, tmp_path):
        """Test that a file YAML cannot parse raises `yaml.YAMLError`.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            Also documented and untested. An unclosed bracket is a parse error, which has to
            surface as one rather than as a validation error about a missing block.
        """
        path = tmp_path / "broken.yaml"
        path.write_text("catchment: {name: Coello\n", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            Catchment.from_yaml(str(path))

    def test_a_top_level_scalar_is_refused(self, tmp_path):
        """Test that a file holding a bare scalar is refused rather than indexed.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A one-word file parses to a string, not to None, so it bypasses the empty-file
            message and reaches pydantic. Pins that it fails there rather than raising an
            `AttributeError` somewhere in the build.
        """
        path = tmp_path / "scalar.yaml"
        path.write_text("hello\n", encoding="utf-8")

        with pytest.raises(ValidationError, match="valid dictionary"):
            Catchment.from_yaml(str(path))

    def test_every_missing_input_path_is_named_at_once(
        self, distributed_mapping, tmp_path
    ):
        """Test that the pre-flight check reports all the missing paths together.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The readers fail one at a time and in the order the build calls them, so a typo
            in the gauge table used to be reported only after the whole meteorological cube
            and the parameter folder had been read. Two typos should now be reported in one
            message, before anything is opened.
        """
        distributed_mapping["parameters"]["path"] = "no/such/parameters"
        distributed_mapping["gauges"]["table"] = "no/such/gauges.csv"

        path = write_yaml(distributed_mapping, tmp_path)

        with pytest.raises(FileNotFoundError) as exc:
            Catchment.from_yaml(path)

        message = str(exc.value)
        assert "parameters.path" in message, (
            f"the missing parameter folder should be named: {message}"
        )
        assert "gauges.table" in message, (
            f"the missing gauge table should be named in the same message: {message}"
        )

    def test_a_netcdf_variable_name_is_not_checked_for_existence(
        self, distributed_mapping, tmp_path
    ):
        """Test that the pre-flight check does not treat a variable name as a path.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            Under `source: netcdf` the three driver fields name variables inside
            `meteo.path`, so checking them as paths would fail every NetCDF configuration.
            The fixture already uses that source.
        """
        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.meteo is not None, "the build should have reached the readers"

    def test_the_gauge_columns_come_from_the_configured_column(
        self, distributed_mapping, tmp_path
    ):
        """Test that `gauges.column` labels the hydrograph frame and every column is filled.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            The frame is labelled from `column` but each file is named after `id`. Filling by
            `int(name)` instead of by the label left a `column != "id"` table with the
            requested columns all-NaN and a second, id-named set beside them -- silently, and
            the metrics were then computed over both.
        """
        distributed_mapping["gauges"]["column"] = "name"

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        columns = list(model.QGauges.columns)
        assert all(isinstance(name, str) for name in columns), (
            f"the frame should carry the table's names, got {columns}"
        )
        all_nan = [name for name in columns if model.QGauges[name].isna().all()]
        assert not all_nan, f"no column should be left unfilled, got {all_nan}"

    def test_the_gauge_table_format_can_differ_from_the_discharge_one(
        self, distributed_mapping, tmp_path
    ):
        """Test that `table_fmt` is what reaches `read_gauge_table`.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            One field used to feed both readers, so the two files could not disagree. The
            Coello table carries no `start` / `end` columns, so an unused `table_fmt` must
            simply be accepted and forwarded rather than applied to the discharge index.
        """
        distributed_mapping["gauges"]["table_fmt"] = "%d/%m/%Y"

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.config.gauges.table_fmt == "%d/%m/%Y", (
            f"the table format should be kept on the config: {model.config.gauges}"
        )
        assert not model.QGauges.isna().all().all(), (
            "the discharge index must still be parsed with gauges.fmt"
        )

    def test_an_unquoted_date_builds_the_same_model_as_a_quoted_one(
        self, distributed_mapping, tmp_path
    ):
        """Test the unquoted-date path end to end, not only at the schema.

        Args:
            distributed_mapping: A complete distributed configuration.
            tmp_path: pytest temporary directory.

        Test scenario:
            `start: 2009-01-01` without quotes is what a YAML author writes first, and it
            reaches pydantic as a `date`. The model built from it must be identical to the
            one built from the quoted spelling.
        """
        quoted = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        unquoted_mapping = copy.deepcopy(distributed_mapping)
        unquoted_mapping["catchment"]["start"] = dt.date.fromisoformat(
            distributed_mapping["catchment"]["start"]
        )
        other = tmp_path / "unquoted"
        other.mkdir()
        unquoted = Catchment.from_yaml(write_yaml(unquoted_mapping, other))

        assert unquoted.period.start == quoted.period.start, (
            f"expected {quoted.period.start}, got {unquoted.period.start}"
        )
        assert len(unquoted.period.date_index) == len(quoted.period.date_index), (
            "both spellings should span the same period"
        )

    def test_a_raster_source_configuration_reads_the_three_folders(
        self,
        distributed_mapping,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_evap_path: str,
        tmp_path,
    ):
        """Test the raster branch of the build, including its path check.

        Args:
            distributed_mapping: A complete distributed configuration.
            coello_prec_path: Folder of precipitation rasters.
            coello_temp_path: Folder of temperature rasters.
            coello_evap_path: Folder of evapotranspiration rasters.
            tmp_path: pytest temporary directory.

        Test scenario:
            The fixture drives every other `from_yaml` test from one combined NetCDF, where
            the three driver fields are variable names. Under `rasters` they are folders that
            the pre-flight check does look for, and `MeteoInputs.from_rasters` is a different
            loader -- so both are covered only here.
        """
        distributed_mapping["meteo"] = {
            "source": "rasters",
            "precipitation": str(Path(coello_prec_path).resolve()),
            "temperature": str(Path(coello_temp_path).resolve()),
            "evapotranspiration": str(Path(coello_evap_path).resolve()),
            "file_name_data_fmt": "%Y.%m.%d",
        }

        model = Catchment.from_yaml(write_yaml(distributed_mapping, tmp_path))

        assert model.meteo.time_steps == len(model.period.date_index), (
            f"the drivers must span the model period: {model.meteo.time_steps} against "
            f"{len(model.period.date_index)}"
        )

    def test_a_missing_driver_folder_is_reported_before_anything_is_read(
        self,
        distributed_mapping,
        coello_temp_path: str,
        coello_evap_path: str,
        tmp_path,
    ):
        """Test that a raster driver folder is checked for existence like any other path.

        Args:
            distributed_mapping: A complete distributed configuration.
            coello_temp_path: Folder of temperature rasters.
            coello_evap_path: Folder of evapotranspiration rasters.
            tmp_path: pytest temporary directory.

        Test scenario:
            A misspelled folder used to be reported from inside pyramids, after the reader
            had opened whatever it could. Under a NetCDF source the driver fields are
            variable names and must not be checked; under this one they are paths and must
            be.
        """
        distributed_mapping["meteo"] = {
            "source": "rasters",
            "precipitation": str(tmp_path / "no-such-folder"),
            "temperature": str(Path(coello_temp_path).resolve()),
            "evapotranspiration": str(Path(coello_evap_path).resolve()),
        }

        path = write_yaml(distributed_mapping, tmp_path)

        with pytest.raises(FileNotFoundError, match="meteo.precipitation"):
            Catchment.from_yaml(path)
