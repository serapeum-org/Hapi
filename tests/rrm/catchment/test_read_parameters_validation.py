"""Tests for the parameter-count validation and the temporal-resolution branches of Catchment.

`read_parameters` accepts a directory of rasters (or a CSV in lumped mode) and then checks the
count against the snow/maxbas combination the caller declared. The count is what the conceptual
model indexes by position, so a wrong one does not raise inside HBV — it reads the wrong
parameter — which is why the guard is worth pinning per combination.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hapi.catchment import Catchment

MAXBAS_BANDS = 11
MUSKINGUM_BANDS = 12


@pytest.fixture(scope="function")
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
        assert model.dt == 1, f"Expected dt of 1, got {model.dt}"

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

    def test_rejects_a_parameter_file_of_the_wrong_length(
        self, coello_start_date: str, coello_end_date: str, tmp_path
    ):
        """Test that the lumped branch counts rows of the CSV, not raster bands.

        Test scenario:
            In lumped mode the parameters arrive as a two-column CSV read into a list, so the
            same guard has to measure length rather than shape. A ten-row file under the
            no-snow/no-maxbas combination (which wants 12) must raise.
        """
        model = Catchment("coello", coello_start_date, coello_end_date)
        path = tmp_path / "parameters.csv"
        pd.DataFrame({"name": [f"p{i}" for i in range(10)], "value": range(10)}).to_csv(
            path, header=False, index=False
        )

        with pytest.raises(ValueError, match="takes 12 parameters"):
            model.read_parameters(str(path), False, maxbas=False)

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
