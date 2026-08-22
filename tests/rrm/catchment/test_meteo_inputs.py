"""Tests for ``MeteoInputs``, which owns the model's meteorological drivers.

The three loaders must be interchangeable: a distributed Muskingum run driven from folders of
rasters, from one NetCDF per variable, or from a single NetCDF holding all three, has to produce
the same discharge. That equivalence is the whole point of the structure, so it is asserted
against the raster-driven run that ``test_extract_discharge_distributed`` already exercises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.netcdf import NetCDF

from hapi.catchment import Catchment
from hapi.inputs import METEO_VARIABLES, FlowNetwork, MeteoInputs, read_rasters
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run

NC_DIR = "tests/rrm/data/coello"


@pytest.fixture(scope="module")
def raster_kwargs(coello_start_date: str, coello_end_date: str) -> dict:
    """Reader arguments matching the Coello raster file names."""
    return dict(
        start=coello_start_date,
        end=coello_end_date,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )


@pytest.fixture(scope="module")
def from_rasters(
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    raster_kwargs: dict,
) -> MeteoInputs:
    """The three drivers loaded from the folders of dated GeoTIFFs."""
    return MeteoInputs.from_rasters(
        coello_prec_path, coello_temp_path, coello_evap_path, **raster_kwargs
    )


@pytest.fixture(scope="module")
def from_netcdf_files() -> MeteoInputs:
    """The three drivers loaded from one NetCDF per variable."""
    return MeteoInputs.from_netcdf_files(
        f"{NC_DIR}/prec.nc", f"{NC_DIR}/temp.nc", f"{NC_DIR}/evap.nc"
    )


#: One NetCDF holding all three drivers, its variables named after them. Regenerate with
#: `tests/rrm/data/coello/combine_netcdf.py`.
COMBINED_NC = f"{NC_DIR}/meteo.nc"


@pytest.fixture(scope="module")
def from_combined_netcdf() -> MeteoInputs:
    """The three drivers loaded from the single multi-variable NetCDF."""
    return MeteoInputs.from_netcdf(
        COMBINED_NC,
        precipitation="precipitation",
        temperature="temperature",
        evapotranspiration="evapotranspiration",
    )


def _run(model_name: str, inputs: MeteoInputs, fixtures: dict) -> Catchment:
    """Build a distributed Coello model on `inputs` and run Muskingum routing."""
    coello = Catchment(
        model_name,
        fixtures["start"],
        fixtures["end"],
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    coello.meteo = inputs
    coello.flow_network = FlowNetwork.from_rasters(fixtures["acc"], fixtures["fd"])
    coello.read_parameters(fixtures["parameters"], False)
    coello.read_lumped_model(HBVLumped, fixtures["area"], fixtures["initial"])
    coello.read_gauge_table(fixtures["gauges_table"], fixtures["acc"])
    coello.read_discharge_gauges(fixtures["gauges"], column="id", fmt="%Y-%m-%d")
    Run.RunHapi(coello)
    return coello


@pytest.fixture(scope="module")
def fixtures(
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
    """The non-meteorological inputs the run needs, gathered once."""
    return dict(
        start=coello_start_date,
        end=coello_end_date,
        acc=coello_acc_path,
        fd=coello_fd_path,
        parameters=coello_dist_parameters_muskingum,
        area=coello_cat_area,
        initial=coello_initial_cond,
        gauges_table=coello_gauges_table,
        gauges=coello_gauges_path,
    )


class TestLoaders:
    """Tests that the three sources yield the same cubes."""

    def test_rasters_and_netcdf_files_agree(
        self, from_rasters: MeteoInputs, from_netcdf_files: MeteoInputs
    ):
        """Test that the raster folders and the per-variable NetCDFs hold the same data.

        Test scenario:
            The NetCDFs were produced from those very rasters, so every cube must match
            element-for-element, no-data cells included -- the loaders must not quietly
            re-scale, transpose, or mask anything.
        """
        for name in METEO_VARIABLES:
            np.testing.assert_array_equal(
                getattr(from_rasters, name),
                getattr(from_netcdf_files, name),
                err_msg=f"{name} differs between the raster and NetCDF loaders",
            )

    def test_combined_netcdf_agrees(
        self, from_netcdf_files: MeteoInputs, from_combined_netcdf: MeteoInputs
    ):
        """Test that one file holding all three variables loads the same cubes.

        Test scenario:
            Same values, different packaging: the caller names which variable is which, and the
            result must be indistinguishable from the one-file-per-variable load.
        """
        for name in METEO_VARIABLES:
            np.testing.assert_array_equal(
                getattr(from_combined_netcdf, name),
                getattr(from_netcdf_files, name),
                err_msg=f"{name} differs when read from the combined NetCDF",
            )

    def test_combined_netcdf_names_its_variables(
        self, from_combined_netcdf: MeteoInputs
    ):
        """Test that the shipped file names its variables after the drivers.

        Test scenario:
            The file is a committed fixture, so its contract is fixed: a caller must be able to
            ask for "precipitation" rather than guess at "Band_1". Also pins the calendar, which
            `NetCDF.time_stamp` does not decode for a multi-variable file.
        """
        nc = NetCDF.read_file(COMBINED_NC)
        assert sorted(nc.variable_names) == sorted(METEO_VARIABLES), (
            f"expected the drivers as variable names, got {nc.variable_names}"
        )
        assert from_combined_netcdf.time is not None, (
            "the calendar must survive the load"
        )
        assert str(from_combined_netcdf.time[0].date()) == "2009-01-01"
        assert str(from_combined_netcdf.time[-1].date()) == "2009-01-10"

    def test_shape_and_calendar(self, from_netcdf_files: MeteoInputs):
        """Test the reported geometry and the decoded calendar.

        Test scenario:
            The Coello fixture is a 13x14 grid over 10 daily steps; `time` carries one stamp per
            step so a caller can cross-check it against the model's own date index.
        """
        assert from_netcdf_files.shape == (13, 14, 10), (
            f"expected a 13x14 grid over 10 steps, got {from_netcdf_files.shape}"
        )
        assert (from_netcdf_files.rows, from_netcdf_files.cols) == (13, 14)
        assert from_netcdf_files.time_steps == 10
        assert from_netcdf_files.time is not None, "the NetCDF carries a calendar"
        assert str(from_netcdf_files.time[0].date()) == "2009-01-01"
        assert str(from_netcdf_files.time[-1].date()) == "2009-01-10"


class TestValidation:
    """Tests for the construction-time checks."""

    def test_mismatched_shapes_rejected(self, from_netcdf_files: MeteoInputs):
        """Test that cubes of different shapes are refused at construction.

        Test scenario:
            A shape mismatch is otherwise invisible until the run loop indexes past the end of
            the shorter cube, far from the cause.
        """
        with pytest.raises(ValueError, match="share one shape"):
            MeteoInputs(
                precipitation=from_netcdf_files.precipitation,
                temperature=from_netcdf_files.temperature[:, :, :5],
                evapotranspiration=from_netcdf_files.evapotranspiration,
            )

    def test_non_array_cube_rejected(self, from_netcdf_files: MeteoInputs):
        """Test that a cube which is not a numpy array is refused.

        Args:
            from_netcdf_files: A valid set of cubes to fill the other two fields.

        Test scenario:
            Handing in a nested list is an easy slip, and it would only fail much later at the
            first `[x, y, :]` index inside the run loop.
        """
        with pytest.raises(TypeError, match="must be a numpy array"):
            MeteoInputs(
                precipitation=[[1.0, 2.0], [3.0, 4.0]],
                temperature=from_netcdf_files.temperature,
                evapotranspiration=from_netcdf_files.evapotranspiration,
            )

    def test_non_3d_cube_rejected(self, from_netcdf_files: MeteoInputs):
        """Test that a 2D array is refused.

        Test scenario:
            Passing a single timestep instead of a cube is an easy slip; it must fail loudly.
        """
        with pytest.raises(ValueError, match="3D"):
            MeteoInputs(
                precipitation=from_netcdf_files.precipitation[:, :, 0],
                temperature=from_netcdf_files.temperature,
                evapotranspiration=from_netcdf_files.evapotranspiration,
            )

    def test_time_length_must_match(self, from_netcdf_files: MeteoInputs):
        """Test that a calendar of the wrong length is refused.

        Test scenario:
            A calendar shorter than the cubes means the caller mixed up two periods.
        """
        with pytest.raises(ValueError, match="time has"):
            MeteoInputs(
                precipitation=from_netcdf_files.precipitation,
                temperature=from_netcdf_files.temperature,
                evapotranspiration=from_netcdf_files.evapotranspiration,
                time=from_netcdf_files.time[:3],
            )

    def test_unknown_variable_names_the_available_ones(self):
        """Test that asking for a missing variable reports what the file holds.

        Test scenario:
            Variable names are caller-supplied strings, so the error has to be self-correcting.
        """
        with pytest.raises(KeyError, match="nope"):
            MeteoInputs.from_netcdf(
                COMBINED_NC,
                precipitation="nope",
                temperature="temperature",
                evapotranspiration="evapotranspiration",
            )

    def test_multi_variable_file_needs_an_explicit_variable(self):
        """Test that from_netcdf_files refuses an ambiguous file.

        Test scenario:
            Handing a three-variable file to the per-variable loader is a mistake; silently
            taking the first variable would load precipitation three times.
        """
        with pytest.raises(ValueError, match="from_netcdf"):
            MeteoInputs.from_netcdf_files(COMBINED_NC, COMBINED_NC, COMBINED_NC)


class TestRunEquivalence:
    """Tests that the model runs identically whichever source fed it."""

    def test_ll_temp_is_the_per_cell_mean_broadcast(self, from_rasters: MeteoInputs):
        """Test the long-term average temperature derived on MeteoInputs.

        Test scenario:
            `ll_temp` is the reference the snow routine compares each step against, and it
            moved off Catchment when MeteoInputs took ownership. It is each cell's mean over
            the whole record, repeated across the time axis, so every step of a cell holds the
            same number.
        """
        expected = from_rasters.temperature.mean(axis=2)

        ll_temp = from_rasters.ll_temp

        assert ll_temp.shape == from_rasters.shape, (
            f"ll_temp must match the cubes {from_rasters.shape}, got {ll_temp.shape}"
        )
        np.testing.assert_allclose(
            ll_temp[:, :, 0],
            expected,
            rtol=1e-6,
            err_msg="ll_temp must hold each cell's mean over time",
        )
        np.testing.assert_allclose(
            ll_temp[:, :, -1],
            ll_temp[:, :, 0],
            rtol=1e-12,
            err_msg="ll_temp must be constant along the time axis",
        )
        assert from_rasters.ll_temp is ll_temp, (
            "ll_temp should be cached, not recomputed"
        )

    def test_ll_temp_can_be_overridden(self, from_netcdf_files: MeteoInputs):
        """Test that a caller-supplied long-term average replaces the derived one.

        Test scenario:
            The reader this replaced accepted an `ll_temp` argument and then silently ignored
            it. Assigning to the property must actually take effect, and must reject an array
            that does not match the cubes.
        """
        # a private copy: the fixture is module-scoped and later tests run on it
        inputs = MeteoInputs(
            precipitation=from_netcdf_files.precipitation,
            temperature=from_netcdf_files.temperature,
            evapotranspiration=from_netcdf_files.evapotranspiration,
        )
        override = np.full(inputs.shape, 7.5, dtype="float32")

        inputs.ll_temp = override

        np.testing.assert_array_equal(inputs.ll_temp, override)
        with pytest.raises(ValueError, match="ll_temp must match"):
            inputs.ll_temp = override[:, :, :3]

    def test_replacing_temperature_drops_the_cached_ll_temp(
        self, from_netcdf_files: MeteoInputs
    ):
        """Test that reassigning `temperature` recomputes the long-term average.

        Test scenario:
            `ll_temp` is derived from `temperature` and cached on first read. Swapping the
            temperature cube afterwards must invalidate that cache, otherwise the snow
            routine keeps comparing against the mean of the array that was thrown away and
            the melt threshold silently belongs to the wrong record.
        """
        inputs = MeteoInputs(
            precipitation=from_netcdf_files.precipitation,
            temperature=from_netcdf_files.temperature,
            evapotranspiration=from_netcdf_files.evapotranspiration,
        )
        stale = inputs.ll_temp
        shifted = from_netcdf_files.temperature + 10.0

        inputs.temperature = shifted

        assert inputs.ll_temp is not stale, (
            "the cached ll_temp must not survive a temperature replacement"
        )
        np.testing.assert_allclose(
            inputs.ll_temp[:, :, 0],
            shifted.mean(axis=2),
            rtol=1e-6,
            err_msg="ll_temp must be recomputed from the replacement cube",
        )

    def test_replacing_temperature_before_any_read_leaves_the_cache_empty(
        self, from_netcdf_files: MeteoInputs
    ):
        """Test that the invalidation hook is inert when nothing was cached yet.

        Test scenario:
            The invalidation hook only has work to do once `ll_temp` has been materialised.
            On a fresh instance it must take the no-op path -- which is only observable by
            inspecting the cache itself: asserting on `ll_temp` alone passes with the hook
            deleted, because a never-populated cache and a correctly-cleared one look the
            same from outside.
        """
        inputs = MeteoInputs(
            precipitation=from_netcdf_files.precipitation,
            temperature=from_netcdf_files.temperature,
            evapotranspiration=from_netcdf_files.evapotranspiration,
        )
        shifted = from_netcdf_files.temperature - 4.0
        assert inputs._ll_temp is None, "nothing should be cached before the first read"

        inputs.temperature = shifted

        assert inputs._ll_temp is None, (
            "the cache must still be empty; the hook has nothing to clear yet"
        )
        np.testing.assert_allclose(
            inputs.ll_temp[:, :, 0],
            shifted.mean(axis=2),
            rtol=1e-6,
            err_msg="the first ll_temp read must describe the assigned cube",
        )
        assert inputs._ll_temp is not None, "reading it must populate the cache"

    def test_replacing_a_cube_with_a_different_shape_is_refused(
        self, from_netcdf_files: MeteoInputs
    ):
        """Test that the three cubes cannot be pulled out of agreement after construction.

        Test scenario:
            The class's central promise is that the cubes share a shape, and `__post_init__`
            alone cannot hold it: the fields are plain attributes. `shape`, `rows` and
            `time_steps` all report precipitation's, so a short temperature cube passes
            `validate_against` and the run then reads the wrong grid without raising.
        """
        inputs = MeteoInputs(
            precipitation=from_netcdf_files.precipitation,
            temperature=from_netcdf_files.temperature,
            evapotranspiration=from_netcdf_files.evapotranspiration,
        )

        with pytest.raises(ValueError, match="must stay") as exc_info:
            inputs.temperature = from_netcdf_files.temperature[:, :-1, :]

        assert "MeteoInputs" in str(exc_info.value), (
            f"the error should say how to change the grid, got: {exc_info.value}"
        )
        assert inputs.temperature.shape == inputs.precipitation.shape, (
            "the rejected assignment must leave the cube untouched"
        )

    @pytest.mark.parametrize(
        "name", ["precipitation", "temperature", "evapotranspiration"]
    )
    def test_every_cube_is_guarded_not_just_temperature(
        self, from_netcdf_files: MeteoInputs, name: str
    ):
        """Test that all three fields re-check, not only the one with a cache behind it.

        Args:
            from_netcdf_files: Loaded drivers to copy.
            name: The cube being replaced.

        Test scenario:
            The hook already existed to invalidate `ll_temp`, which only `temperature` needs.
            The shape guarantee is about all three, so the check must not be attached to that
            one field by accident.
        """
        inputs = MeteoInputs(
            precipitation=from_netcdf_files.precipitation,
            temperature=from_netcdf_files.temperature,
            evapotranspiration=from_netcdf_files.evapotranspiration,
        )

        with pytest.raises(ValueError, match=name):
            setattr(inputs, name, getattr(inputs, name)[:, :, :-1])

    def test_a_same_shaped_replacement_is_accepted(
        self, from_netcdf_files: MeteoInputs
    ):
        """Test that the guard does not block a legitimate swap.

        Test scenario:
            Replacing a cube with one of the same shape is how a caller applies a correction
            or a unit conversion, so it must still work -- and for temperature it must still
            drop the derived long-term average.
        """
        inputs = MeteoInputs(
            precipitation=from_netcdf_files.precipitation,
            temperature=from_netcdf_files.temperature,
            evapotranspiration=from_netcdf_files.evapotranspiration,
        )
        stale = inputs.ll_temp
        shifted = from_netcdf_files.temperature + 2.0

        inputs.temperature = shifted

        np.testing.assert_allclose(inputs.temperature, shifted)
        assert inputs.ll_temp is not stale, "the derived average must be recomputed"

    def test_netcdf_run_reproduces_the_raster_run(
        self, from_rasters: MeteoInputs, from_netcdf_files: MeteoInputs, fixtures: dict
    ):
        """Test that a full Muskingum run gives the same discharge from either source.

        Test scenario:
            The end-to-end claim: swapping folders of rasters for NetCDFs must not move the
            hydrograph by a single cell. Compares the routed `Qtot` field and the per-gauge
            `Qsim` extracted from it.
        """
        raster_run = _run("coello-rasters", from_rasters, fixtures)
        netcdf_run = _run("coello-netcdf", from_netcdf_files, fixtures)

        np.testing.assert_allclose(
            netcdf_run.Qtot,
            raster_run.Qtot,
            rtol=1e-9,
            err_msg="the routed discharge field differs between the two sources",
        )
        raster_run.extract_discharge(calculate_metrics=False)
        netcdf_run.extract_discharge(calculate_metrics=False)
        np.testing.assert_allclose(
            netcdf_run.Qsim.to_numpy(dtype=float),
            raster_run.Qsim.to_numpy(dtype=float),
            rtol=1e-9,
            err_msg="the per-gauge hydrographs differ between the two sources",
        )


@pytest.fixture(scope="module")
def coello_muskingum_from_netcdf(
    from_combined_netcdf: MeteoInputs, fixtures: dict
) -> Catchment:
    """Distributed Coello with a completed Muskingum run, driven from one NetCDF.

    The `coello_muskingum_run` fixture of `test_extract_discharge_distributed`, with its three
    raster reader calls replaced by a single `MeteoInputs.from_netcdf` load of `meteo.nc`.

    Returns:
        Catchment: Model with `Qtot` populated by the spatial routing.
    """
    return _run("coello-combined-netcdf", from_combined_netcdf, fixtures)


@pytest.mark.e2e
class TestMuskingumFromCombinedNetcdf:
    """End-to-end Muskingum run whose drivers came from the single multi-variable NetCDF."""

    def test_metrics(self, coello_muskingum_from_netcdf: Catchment):
        """Test that the run yields finite metrics for every gauge.

        Test scenario:
            The same claim `test_extract_discharge_distributed_metrics` makes of the
            raster-driven run: after routing, `extract_discharge` walks the gauge table and
            fills all seven metrics with finite numbers. Driving the model from one NetCDF has
            to reach the same place, which is what makes the loader usable rather than merely
            correct at the array level.
        """
        coello = coello_muskingum_from_netcdf
        coello.extract_discharge(calculate_metrics=True)

        assert isinstance(coello.metrics, DataFrame), (
            f"metrics should be a DataFrame, got {type(coello.metrics)}"
        )
        assert list(coello.metrics.index) == [
            "RMSE",
            "NSE",
            "NSEhf",
            "KGE",
            "WB",
            "Pearson-CC",
            "R2",
        ], f"metrics rows mismatch: {list(coello.metrics.index)}"

        n_gauges = len(coello.GaugesTable)
        assert coello.metrics.shape[1] == n_gauges, (
            f"expected one metrics column per gauge ({n_gauges}), "
            f"got {coello.metrics.shape[1]}"
        )
        assert np.isfinite(coello.metrics.to_numpy(dtype=float)).all(), (
            "all metric values should be finite"
        )
        assert coello.Qsim.shape == (len(coello.date_index), n_gauges), (
            f"Qsim shape mismatch: {coello.Qsim.shape}"
        )

    def test_reproduces_the_raster_run(
        self,
        coello_muskingum_from_netcdf: Catchment,
        from_rasters: MeteoInputs,
        fixtures: dict,
    ):
        """Test that the one-file run matches the raster-driven run exactly.

        Test scenario:
            Packing all three drivers into one file, with the caller naming which variable is
            which, must not move the hydrograph. Compares the routed `Qtot` field and the
            per-gauge `Qsim` against a run fed from the raster folders.
        """
        raster_run = _run("coello-rasters-vs-combined", from_rasters, fixtures)
        combined = coello_muskingum_from_netcdf

        np.testing.assert_allclose(
            combined.Qtot,
            raster_run.Qtot,
            rtol=1e-9,
            err_msg="the routed discharge field differs from the raster-driven run",
        )
        raster_run.extract_discharge(calculate_metrics=False)
        combined.extract_discharge(calculate_metrics=False)
        np.testing.assert_allclose(
            combined.Qsim.to_numpy(dtype=float),
            raster_run.Qsim.to_numpy(dtype=float),
            rtol=1e-9,
            err_msg="the per-gauge hydrographs differ from the raster-driven run",
        )


@pytest.fixture(scope="function")
def numbered_rasters(tmp_path) -> Path:
    """Write four rasters whose names carry a plain index, not a date.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path: Folder holding ``0_par.tif`` ... ``3_par.tif``, each filled with its own index.
    """
    for i in range(4):
        Dataset.create_from_array(
            np.full((3, 3), float(i), dtype="float32"),
            geo=(0.0, 4000.0, 0.0, 12000.0, 0.0, -4000.0),
            epsg=32618,
            no_data_value=-9999.0,
        ).to_file(str(tmp_path / f"{i}_par.tif"))
    return tmp_path


class TestReadRasters:
    """Tests for ``read_rasters``, the adapter over ``DatasetCollection.from_files``."""

    def test_numeric_mode_rejects_a_name_without_a_number(self, tmp_path):
        """Test that numeric ordering fails loudly when a name carries no index.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            The numeric mode sorts on an integer pulled out of each file name. A file matching
            the glob but carrying no number cannot be placed in that order, and silently
            dropping or appending it would scramble the cube.
        """
        Dataset.create_from_array(
            np.zeros((2, 2), dtype="float32"),
            geo=(0.0, 4000.0, 0.0, 8000.0, 0.0, -4000.0),
            epsg=32618,
            no_data_value=-9999.0,
        ).to_file(str(tmp_path / "no_index_here.tif"))

        with pytest.raises(ValueError, match="matched no number"):
            read_rasters(tmp_path, regex_string=r"\d+", date=False)

    def test_numeric_range_that_excludes_everything_raises(self, numbered_rasters):
        """Test that an index window matching no file is reported, not returned empty.

        Args:
            numbered_rasters: Folder of index-named rasters.

        Test scenario:
            The Rhine model selects a season with integer bounds. Bounds outside the available
            indices must fail here rather than hand back an empty collection that only breaks
            later, inside the run loop.
        """
        with pytest.raises(FileNotFoundError, match=r"index within \[90, 99\]"):
            read_rasters(
                numbered_rasters, regex_string=r"\d+", date=False, start=90, end=99
            )

    def test_numeric_range_selects_an_inclusive_window(self, numbered_rasters):
        """Test that integer bounds keep both endpoints.

        Args:
            numbered_rasters: Folder of index-named rasters.

        Test scenario:
            Each raster holds its own index as its pixel value, so the recovered values name
            exactly which files survived the filter.
        """
        collection = read_rasters(
            numbered_rasters, regex_string=r"\d+", date=False, start=1, end=2
        )

        recovered = [
            int(collection.values[i].flat[0]) for i in range(collection.time_length)
        ]
        assert recovered == [1, 2], (
            f"expected indices 1 and 2 inclusive, got {recovered}"
        )

    def test_unordered_read_when_no_date_format_is_given(self, numbered_rasters):
        """Test the fallback branch that reads a folder without ordering it.

        Args:
            numbered_rasters: Folder of index-named rasters.

        Test scenario:
            With ``date=True`` but no ``file_name_data_fmt`` there is nothing to sort on, and
            pyramids rejects an ordered read in that state. The reader falls back to a plain
            unordered one, which is what ``Inputs.prepare_inputs`` relies on.
        """
        collection = read_rasters(numbered_rasters, date=True, file_name_data_fmt=None)

        assert collection.time_length == 4, (
            f"expected all 4 rasters, got {collection.time_length}"
        )


class TestValidateAgainst:
    """Tests for ``MeteoInputs.validate_against``."""

    def test_accepts_the_matching_grid(self, from_netcdf_files: MeteoInputs):
        """Test that the cubes' own grid passes and the call is a pure check.

        Args:
            from_netcdf_files: The Coello drivers, a 13x14 grid.

        Test scenario:
            The check must be silent when the meteorology covers the catchment, which is the
            normal case on every run -- and it must only check: a validator that reshaped or
            replaced a cube would change the run it was supposed to be guarding.
        """
        before = {name: getattr(from_netcdf_files, name) for name in METEO_VARIABLES}

        assert from_netcdf_files.validate_against(13, 14) is None, (
            "validate_against reports by raising; it must not return a verdict to ignore"
        )

        for name, cube in before.items():
            assert getattr(from_netcdf_files, name) is cube, (
                f"{name} must be left exactly as it was"
            )

    def test_accepts_a_matching_calendar(self, from_netcdf_files: MeteoInputs):
        """Test that a date_index the drivers span passes.

        Args:
            from_netcdf_files: The Coello drivers, 2009-01-01 to 2009-01-10.

        Test scenario:
            The normal case -- the model was built for exactly the period the rasters cover.
        """
        import pandas as pd

        from_netcdf_files.validate_against(
            13, 14, pd.date_range("2009-01-01", "2009-01-10", freq="D")
        )

    def test_rejects_a_calendar_of_a_different_length(
        self, from_netcdf_files: MeteoInputs
    ):
        """Test that a model spanning more dates than the drivers is refused.

        Args:
            from_netcdf_files: The Coello drivers, 10 daily steps.

        Test scenario:
            The run is positional, so 10 driver steps against a 32-day model silently pairs
            each step with the wrong date instead of failing.
        """
        import pandas as pd

        with pytest.raises(ValueError, match="10 steps but the model spans 32"):
            from_netcdf_files.validate_against(
                13, 14, pd.date_range("2009-01-01", "2009-02-01", freq="D")
            )

    def test_rejects_a_calendar_covering_a_different_period(
        self, from_netcdf_files: MeteoInputs
    ):
        """Test that a same-length window over different dates is refused.

        Args:
            from_netcdf_files: The Coello drivers, 2009-01-01 to 2009-01-10.

        Test scenario:
            A length check alone would pass a model shifted by a year, which would score the
            simulation against the wrong observations.
        """
        import pandas as pd

        with pytest.raises(ValueError, match="2010-01-01"):
            from_netcdf_files.validate_against(
                13, 14, pd.date_range("2010-01-01", "2010-01-10", freq="D")
            )

    def test_calendar_check_is_skipped_when_no_dates_are_given(
        self, from_netcdf_files: MeteoInputs
    ):
        """Test that omitting date_index checks the grid and nothing else.

        Args:
            from_netcdf_files: The Coello drivers, which carry a 2009 calendar.

        Test scenario:
            A caller with no calendar of its own must still be able to check the grid. This
            is only meaningful if a calendar that *would* be rejected passes when omitted --
            otherwise it is the same test as the one above with a different name.
        """
        wrong_period = pd.date_range("1999-01-01", periods=10, freq="D")
        with pytest.raises(ValueError, match="cover"):
            from_netcdf_files.validate_against(13, 14, wrong_period)

        from_netcdf_files.validate_against(13, 14)

    @pytest.mark.parametrize(
        "rows, cols", [(12, 14), (13, 15), (1, 1)], ids=["rows", "cols", "both"]
    )
    def test_rejects_a_grid_the_cubes_do_not_cover(
        self, from_netcdf_files: MeteoInputs, rows: int, cols: int
    ):
        """Test that a grid mismatch is refused and both shapes are named.

        Args:
            from_netcdf_files: The Coello drivers, a 13x14 grid.
            rows: Row count to check against.
            cols: Column count to check against.

        Test scenario:
            This replaced seven copies of a bare assert in ``run.py`` and ``calibration.py``.
            The message has to name the meteorology's grid *and* the model's, since the point
            is telling the user which input is the odd one out.
        """
        with pytest.raises(ValueError, match="13x14") as exc:
            from_netcdf_files.validate_against(rows, cols)

        assert f"{rows}x{cols}" in str(exc.value), (
            f"the error should name the model grid {rows}x{cols}, got: {exc.value}"
        )


class TestVariableSelectionAndCalendar:
    """Tests for the NetCDF loaders' variable picking and calendar decoding."""

    def test_explicit_variable_name_is_used_for_every_file(
        self, from_combined_netcdf: MeteoInputs
    ):
        """Test that `variable=` picks the named variable out of a multi-variable file.

        Test scenario:
            Passing a single-variable file cannot tell the two branches apart -- the explicit
            name and the "take the only one" default read the same array. The combined file
            holds three, so naming one is the only way it can be read at all, and the cube
            that comes back identifies which name was honoured.
        """
        inputs = MeteoInputs.from_netcdf_files(
            COMBINED_NC, COMBINED_NC, COMBINED_NC, variable="temperature"
        )

        for name in METEO_VARIABLES:
            np.testing.assert_array_equal(
                getattr(inputs, name),
                from_combined_netcdf.temperature,
                err_msg=f"{name} should hold the named variable, not the file's first",
            )

    def test_a_multi_variable_file_without_a_name_is_refused(self):
        """Test that the default branch reports the ambiguity instead of guessing.

        Test scenario:
            The counterpart to naming a variable: with three in the file there is no "only
            one" to take, so the loader must say so and list what it found rather than pick
            the first.
        """
        with pytest.raises(ValueError, match="pass variable=") as exc_info:
            MeteoInputs.from_netcdf_files(COMBINED_NC, COMBINED_NC, COMBINED_NC)

        assert "temperature" in str(exc_info.value), (
            f"the error should list the variables it found, got: {exc_info.value}"
        )

    def test_an_unknown_variable_name_names_what_is_available(self):
        """Test that asking for a variable the file lacks lists the ones it has.

        Test scenario:
            A typo in the variable name is the likely cause, so the message has to show the
            real names rather than only echoing the one that failed.
        """
        with pytest.raises(KeyError) as exc_info:
            MeteoInputs.from_netcdf_files(
                COMBINED_NC, COMBINED_NC, COMBINED_NC, variable="nope"
            )

        message = str(exc_info.value)
        assert "nope" in message, (
            f"the error should name the missing variable: {message}"
        )
        assert "temperature" in message, (
            f"the error should list the available variables: {message}"
        )

    def test_file_without_a_calendar_loads_with_time_none(self, tmp_path):
        """Test that a positional time axis is left alone rather than read as 1970.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            ``to_netcdf`` writes a positional index for an undated collection. Those values are
            small integers; decoding them as nanoseconds since the epoch would silently date the
            run to 1970, so the loader must report no calendar instead.
        """
        for i in range(2):
            Dataset.create_from_array(
                np.full((3, 3), float(i), dtype="float32"),
                geo=(0.0, 4000.0, 0.0, 12000.0, 0.0, -4000.0),
                epsg=32618,
                no_data_value=-9999.0,
            ).to_file(str(tmp_path / f"plain_{i}.tif"))
        undated = str(tmp_path / "undated.nc")
        DatasetCollection.from_files(tmp_path, glob="*.tif").to_netcdf(undated)

        inputs = MeteoInputs.from_netcdf_files(
            undated, undated, undated, variable="Band_1"
        )

        assert inputs.time is None, (
            f"a positional axis carries no calendar, got {inputs.time}"
        )


class TestCalendarFallbacks:
    """Tests for ``MeteoInputs._calendar``, which decodes a file's time axis."""

    class _Stub:
        """Minimal stand-in for a NetCDF, exposing only what ``_calendar`` reads."""

        def __init__(self, time_stamp, time_values):
            """Store the two accessors' behaviour.

            Args:
                time_stamp: Value to return, or an exception instance to raise.
                time_values: Value to return, or an exception instance to raise.
            """
            self._time_stamp = time_stamp
            self._time_values = time_values

        @property
        def time_stamp(self):
            """Return the decoded stamps, or raise what the test asked for."""
            if isinstance(self._time_stamp, Exception):
                raise self._time_stamp
            return self._time_stamp

        def get_time_values(self):
            """Return the raw time values, or raise what the test asked for."""
            if isinstance(self._time_values, Exception):
                raise self._time_values
            return self._time_values

    def test_decoded_stamps_are_preferred(self):
        """Test that a file whose ``time_stamp`` decodes is used as-is.

        Test scenario:
            The single-variable case, where pyramids already hands back date strings; the raw
            values should not be consulted at all.
        """
        stub = self._Stub(["2009-01-01", "2009-01-02"], np.array([0, 1]))

        calendar = MeteoInputs._calendar(stub)

        assert calendar is not None, "decoded stamps should produce a calendar"
        assert str(calendar[0].date()) == "2009-01-01", f"got {calendar[0]}"

    @pytest.mark.parametrize(
        "raised",
        [AttributeError("no such attribute"), KeyError("time"), ValueError("bad axis")],
        ids=["attribute", "key", "value"],
    )
    def test_falls_back_to_raw_values_when_time_stamp_raises(self, raised: Exception):
        """Test that a failing ``time_stamp`` falls through to the raw values.

        Args:
            raised: The exception ``time_stamp`` raises.

        Test scenario:
            A multi-variable file returns None from ``time_stamp``, but older or odder files
            raise instead. Either way the nanosecond stamps are still readable, so the calendar
            must survive rather than being lost.
        """
        epoch_ns = np.array([1230768000000000000, 1230854400000000000], dtype="int64")
        stub = self._Stub(raised, epoch_ns)

        calendar = MeteoInputs._calendar(stub)

        assert calendar is not None, (
            f"{type(raised).__name__} should fall back, not give up"
        )
        assert str(calendar[0].date()) == "2009-01-01", f"got {calendar[0]}"

    def test_returns_none_when_both_accessors_fail(self):
        """Test that a file with no readable time axis yields no calendar.

        Test scenario:
            The calendar is optional on ``MeteoInputs``, so an unreadable axis must degrade to
            None rather than propagate an error out of a loader.
        """
        stub = self._Stub(AttributeError("none"), ValueError("none"))

        assert MeteoInputs._calendar(stub) is None, (
            "an unreadable axis should give no calendar"
        )

    def test_positional_index_is_not_read_as_a_date(self):
        """Test that small integers are treated as an index, not as epoch nanoseconds.

        Test scenario:
            ``to_netcdf`` writes 0..n-1 for an undated collection. Passing those to a datetime
            decoder would silently date every run to 1970-01-01.
        """
        stub = self._Stub(None, np.arange(5, dtype="int64"))

        assert MeteoInputs._calendar(stub) is None, (
            "a positional index carries no calendar"
        )
