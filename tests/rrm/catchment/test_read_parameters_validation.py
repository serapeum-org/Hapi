"""Tests for the parameter-count validation and the temporal-resolution branches of Catchment.

`read_parameters` accepts a directory of rasters (or a CSV in lumped mode) and then checks the
count against the snow/maxbas combination the caller declared. The count is what the conceptual
model indexes by position, so a wrong one does not raise inside HBV — it reads the wrong
parameter — which is why the guard is worth pinning per combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hapi.catchment import Catchment
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped

MAXBAS_BANDS = 11
MUSKINGUM_BANDS = 12


@pytest.fixture
def distributed(coello_start_date: str, coello_end_date: str) -> Catchment:
    """Provide an empty distributed daily catchment.

    Returns:
        Catchment: Instance with no inputs read yet.
    """
    return Catchment(
        "coello",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
        fmt="%Y-%m-%d",
    )


class TestTemporalResolution:
    """Tests for the temporal-resolution branch of `Catchment.__init__`."""

    @pytest.mark.parametrize(
        "resolution, expected_steps, expected_freq",
        [("Daily", 366, "D"), ("Hourly", 8761, "h")],
    )
    def test_known_resolutions_build_the_date_index(
        self, resolution: str, expected_steps: int, expected_freq: str
    ):
        """Test that a recognised resolution sets `dt`, the factor and the date index.

        Args:
            resolution: The temporal resolution to construct with.
            expected_steps: Number of steps a full year spans at that resolution.
            expected_freq: Pandas frequency alias the index should carry.

        Test scenario:
            The date index is what `MeteoInputs.validate_against` is checked against, so the
            two recognised resolutions must produce an index of the right length and step.
        """
        model = Catchment(
            "coello", "2009-01-01", "2010-01-01", temporal_resolution=resolution
        )

        assert len(model.date_index) == expected_steps, (
            f"Expected {expected_steps} steps, got {len(model.date_index)}"
        )
        assert model.date_index.freqstr.lower() == expected_freq.lower(), (
            f"Expected frequency {expected_freq}, got {model.date_index.freqstr}"
        )
        # `dt` is hard-coded to 1 in both branches, so it carries no resolution information;
        # what distinguishes them is the conversion factor, asserted below.
        assert model.dt == 1, f"Expected dt of 1, got {model.dt}"
        assert model.temporal_resolution == resolution.lower(), (
            f"the resolution must be stored lowercased, got {model.temporal_resolution}"
        )

    def test_unknown_resolution_is_rejected_at_construction(self):
        """Test that a resolution other than daily or hourly is refused up front.

        Test scenario:
            Every downstream check pairs the meteorological cubes with `date_index`
            positionally, so a resolution the constructor cannot build an index for has to
            fail at construction rather than leave the model half-built.
        """
        with pytest.raises(ValueError, match="'daily' and 'hourly'"):
            Catchment("coello", "2009-01-01", "2009-01-10", temporal_resolution="15min")

    def test_hourly_resolution_scales_the_conversion_factor(self):
        """Test that the hourly branch divides the daily conversion factor by 24.

        Test scenario:
            The conversion factor turns depth per step into discharge, so it has to follow
            the step length. The hourly value must be exactly a twenty-fourth of the daily
            one, not a rounded constant.
        """
        daily = Catchment("coello", "2009-01-01", "2009-01-10")
        hourly = Catchment(
            "coello", "2009-01-01", "2009-01-10", temporal_resolution="Hourly"
        )

        assert hourly.conversion_factor == pytest.approx(
            daily.conversion_factor / 24
        ), f"Expected {daily.conversion_factor / 24}, got {hourly.conversion_factor}"


class TestReadParametersDistributed:
    """Tests for the distributed branch of `Catchment.read_parameters`."""

    @pytest.mark.parametrize(
        "fixture_name, snow, maxbas, expected_count",
        [
            ("coello_dist_parameters_maxbas", True, True, 16),
            ("coello_dist_parameters_muskingum", False, True, 11),
            ("coello_dist_parameters_muskingum", True, False, 17),
            ("coello_dist_parameters_maxbas", False, False, 12),
        ],
    )
    def test_rejects_a_parameter_count_the_configuration_does_not_expect(
        self,
        distributed: Catchment,
        request,
        fixture_name: str,
        snow: bool,
        maxbas: bool,
        expected_count: int,
    ):
        """Test that each snow/maxbas combination enforces its own parameter count.

        Args:
            distributed: Empty distributed catchment.
            request: Used to resolve the parameter-directory fixture by name.
            fixture_name: Which bundled parameter set to read.
            snow: Whether the snow routine is declared.
            maxbas: Whether triangular routing is declared.
            expected_count: The count that combination requires.

        Test scenario:
            The bundled sets hold 11 (maxbas) and 12 (muskingum) parameters. Every
            combination that expects a different number must raise, and the message must
            name the count it wanted so the caller can tell which flag is wrong.
        """
        path = request.getfixturevalue(fixture_name)

        with pytest.raises(ValueError, match=f"takes {expected_count} parameters"):
            distributed.read_parameters(path, snow, maxbas=maxbas)

    @pytest.mark.parametrize(
        "fixture_name, snow, maxbas, expected_bands",
        [
            ("coello_dist_parameters_maxbas", False, True, MAXBAS_BANDS),
            ("coello_dist_parameters_muskingum", False, False, MUSKINGUM_BANDS),
        ],
    )
    def test_accepts_the_matching_configuration(
        self,
        distributed: Catchment,
        request,
        fixture_name: str,
        snow: bool,
        maxbas: bool,
        expected_bands: int,
    ):
        """Test that the two bundled sets load under the configuration they were made for.

        Args:
            distributed: Empty distributed catchment.
            request: Used to resolve the parameter-directory fixture by name.
            fixture_name: Which bundled parameter set to read.
            snow: Whether the snow routine is declared.
            maxbas: Whether triangular routing is declared.
            expected_bands: Number of parameter bands the set holds.

        Test scenario:
            The counterpart to the rejection cases — the guard must not fire on the
            combinations the fixtures were built for, and the cube must come back with the
            parameter axis last.
        """
        path = request.getfixturevalue(fixture_name)

        distributed.read_parameters(path, snow, maxbas=maxbas)

        assert distributed.parameters.shape[2] == expected_bands, (
            f"Expected {expected_bands} parameter bands, "
            f"got {distributed.parameters.shape[2]}"
        )
        assert distributed.snow is snow, f"snow flag not stored: {distributed.snow}"
        assert distributed.maxbas is maxbas, (
            f"maxbas flag not stored: {distributed.maxbas}"
        )

    def test_missing_directory_names_the_path_it_could_not_read(
        self, distributed: Catchment, tmp_path
    ):
        """Test that a missing parameter directory raises naming the path.

        Test scenario:
            Path validation is delegated to pyramids, whose message does not name the
            offending directory. Hapi re-raises with it, which is the difference between a
            usable error and one the user has to bisect.
        """
        missing = tmp_path / "no-such-parameters"

        with pytest.raises(FileNotFoundError, match="no-such-parameters"):
            distributed.read_parameters(str(missing), False, maxbas=True)


class TestReadParametersLumped:
    """Tests for the lumped branch of `Catchment.read_parameters`."""

    @pytest.mark.parametrize(
        "snow, maxbas, expected_count",
        [
            (True, True, 16),
            (False, True, 11),
            (True, False, 17),
            (False, False, 12),
        ],
    )
    def test_rejects_a_parameter_file_of_the_wrong_length(
        self,
        coello_start_date: str,
        coello_end_date: str,
        tmp_path,
        snow: bool,
        maxbas: bool,
        expected_count: int,
    ):
        """Test that the lumped branch counts rows of the CSV, not raster bands.

        Args:
            coello_start_date: Simulation start date.
            coello_end_date: Simulation end date.
            tmp_path: Directory the parameter file is written into.
            snow: Whether the snow routine is declared.
            maxbas: Whether triangular routing is declared.
            expected_count: The count that combination requires.

        Test scenario:
            In lumped mode the parameters arrive as a two-column CSV read into a list, so the
            same guard has to measure length rather than shape. A ten-row file is the wrong
            length for every combination, so each must raise naming its own count.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)
        path = tmp_path / "parameters.csv"
        pd.DataFrame({"name": [f"p{i}" for i in range(10)], "value": range(10)}).to_csv(
            path, header=False, index=False
        )

        with pytest.raises(ValueError, match=f"takes {expected_count} parameters"):
            model.read_parameters(str(path), snow, maxbas=maxbas)

    def test_missing_file_raises_before_reading(
        self, coello_start_date: str, coello_end_date: str, tmp_path
    ):
        """Test that a missing lumped parameter file raises a clear error.

        Test scenario:
            The lumped branch checks the path itself rather than delegating to pyramids, so
            it needs its own coverage.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            model.read_parameters(str(tmp_path / "absent.csv"), False)


class TestReadLumpedModelQInit:
    """Tests for the `q_init` guard in `Catchment.read_lumped_model`."""

    def test_a_float_initial_discharge_is_accepted(
        self, coello_start_date: str, coello_end_date: str, coello_initial_cond: list
    ):
        """Test that the documented type is allowed through.

        Test scenario:
            `q_init` is annotated `float | None` and the conceptual model divides it in two
            (`q_uz[0] = q_init / 2`), so a float is the only thing it can be. The guard used
            to assert `not isinstance(q_init, float)` under the message "q_init should be of
            type float", rejecting exactly the valid input -- so nobody could pass one.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)

        model.read_lumped_model(HBVLumped, 1530.0, coello_initial_cond, q_init=5.0)

        assert model.q_init == pytest.approx(5.0), (
            f"the initial discharge must be stored, got {model.q_init}"
        )

    def test_omitting_the_initial_discharge_leaves_it_unset(
        self, coello_start_date: str, coello_end_date: str, coello_initial_cond: list
    ):
        """Test that the guard does not fire when no initial discharge is given.

        Test scenario:
            `q_init=None` is the default and means "derive it from the initial state", so it
            must skip the type check rather than fail it.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)

        model.read_lumped_model(HBVLumped, 1530.0, coello_initial_cond)

        assert model.q_init is None, (
            f"expected no initial discharge, got {model.q_init}"
        )

    @pytest.mark.parametrize("bad", [5, "5.0", [5.0]], ids=["int", "str", "list"])
    def test_a_non_float_initial_discharge_is_refused(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_initial_cond: list,
        bad,
    ):
        """Test that anything other than a float is rejected, naming the type wanted.

        Args:
            coello_start_date: Simulation start date.
            coello_end_date: Simulation end date.
            coello_initial_cond: Initial HBV state.
            bad: A value of the wrong type.

        Test scenario:
            The value is divided in two inside the conceptual model, so a string or a list
            fails there rather than here -- far from the call that supplied it. A `TypeError`
            rather than an `AssertionError`, so the check survives `python -O`.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)

        with pytest.raises(TypeError, match="q_init should be of type float"):
            model.read_lumped_model(HBVLumped, 1530.0, coello_initial_cond, q_init=bad)


class TestReadLumpedInputs:
    """Tests for the column handling in `Catchment.read_lumped_inputs`."""

    @pytest.fixture
    def three_column_csv(self, tmp_path) -> str:
        """Write a lumped input file without the long-term average column.

        Returns:
            str: Path to a CSV of [date, prec, ET, temp].
        """
        path = tmp_path / "meteo-3col.csv"
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2009-01-01", periods=10, freq="D"),
                "prec": np.linspace(0.0, 9.0, 10),
                "et": np.linspace(1.0, 2.0, 10),
                "temp": np.linspace(15.0, 25.0, 10),
            }
        )
        frame.to_csv(path, index=False)
        return str(path)

    def test_a_three_column_file_gains_the_long_term_average(
        self, coello_start_date: str, coello_end_date: str, three_column_csv: str
    ):
        """Test that the fourth column is derived rather than left missing.

        Test scenario:
            The method documents 3 or 4 columns, but `Wrapper.Lumped` reads `data[:, 3]`
            unconditionally. A three-column file was therefore accepted here and then raised
            `IndexError` in the middle of the run. The derived column is the record's mean
            temperature, which is what the reader this replaced computed.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)

        model.read_lumped_inputs(three_column_csv)

        assert model.data.shape[1] == 4, (
            f"the run reads four columns, got {model.data.shape[1]}"
        )
        assert model.data[:, 3] == pytest.approx(model.data[:, 2].mean()), (
            "the fourth column must hold the record's mean temperature"
        )

    def test_a_four_column_file_is_left_alone(
        self, coello_start_date: str, coello_end_date: str, tmp_path
    ):
        """Test that a caller-supplied long-term average is not overwritten.

        Test scenario:
            A caller who supplies the fourth column has chosen a reference the snow routine
            should use; deriving one on top of it would silently discard that choice.
        """
        path = tmp_path / "meteo-4col.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2009-01-01", periods=10, freq="D"),
                "prec": np.linspace(0.0, 9.0, 10),
                "et": np.linspace(1.0, 2.0, 10),
                "temp": np.linspace(15.0, 25.0, 10),
                "tm": np.full(10, 7.5),
            }
        ).to_csv(path, index=False)
        model = Catchment("coello", coello_start_date, coello_end_date)

        model.read_lumped_inputs(str(path))

        assert model.data.shape[1] == 4, (
            f"expected 4 columns, got {model.data.shape[1]}"
        )
        assert model.data[:, 3] == pytest.approx(7.5), (
            "the supplied long-term average must survive untouched"
        )

    @pytest.mark.parametrize("columns", [2, 5])
    def test_any_other_column_count_is_refused(
        self, coello_start_date: str, coello_end_date: str, tmp_path, columns: int
    ):
        """Test that a file with the wrong number of columns raises.

        Args:
            coello_start_date: Simulation start date.
            coello_end_date: Simulation end date.
            tmp_path: Directory the file is written into.
            columns: How many data columns to write.

        Test scenario:
            The columns are read by position, so a file with a different count cannot be
            interpreted and must be refused rather than silently mapped.
        """
        path = tmp_path / f"meteo-{columns}col.csv"
        data = {"date": pd.date_range("2009-01-01", periods=10, freq="D")}
        data.update({f"c{i}": np.arange(10.0) for i in range(columns)})
        pd.DataFrame(data).to_csv(path, index=False)
        model = Catchment("coello", coello_start_date, coello_end_date)

        with pytest.raises(ValueError, match="should be of length at least 3"):
            model.read_lumped_inputs(str(path))
