"""Tests for ``MeteoInputs`` and the ``Catchment.read_meteo`` entry point.

The three loaders must be interchangeable: a distributed Muskingum run driven from folders of
rasters, from one NetCDF per variable, or from a single NetCDF holding all three, has to produce
the same discharge. That equivalence is the whole point of the structure, so it is asserted
against the raster-driven run that ``test_extract_discharge_distributed`` already exercises.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, DatasetCollection

from hapi.catchment import Catchment
from hapi.inputs import METEO_VARIABLES, MeteoInputs
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


@pytest.fixture(scope="module")
def combined_netcdf(tmp_path_factory, from_netcdf_files: MeteoInputs) -> str:
    """Write one NetCDF holding all three drivers as separate variables.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.
        from_netcdf_files: Source cubes, so the combined file carries identical values.

    Returns:
        str: Path to a NetCDF whose ``Band_1`` / ``Band_2`` / ``Band_3`` variables are
            precipitation / temperature / evapotranspiration.

    Test scenario:
        Hapi has no writer for this layout, so it is built from three-band rasters: one band per
        driver, which ``to_netcdf`` turns into one variable per band.
    """
    folder = tmp_path_factory.mktemp("combined")
    cubes = [getattr(from_netcdf_files, name) for name in METEO_VARIABLES]
    for step in range(from_netcdf_files.time_steps):
        bands = np.stack([cube[:, :, step] for cube in cubes]).astype("float32")
        Dataset.create_from_array(
            bands,
            geo=(0.0, 4000.0, 0.0, bands.shape[1] * 4000.0, 0.0, -4000.0),
            epsg=32618,
            no_data_value=-9999.0,
        ).to_file(str(folder / f"{step}_all_2009_1_{step + 1}.tif"))

    out = folder / "all.nc"
    DatasetCollection.from_files(
        folder,
        glob="*.tif",
        date_format="%Y_%m_%d",
        date_regex=r"\d{4}_\d{1,2}_\d{1,2}",
    ).to_netcdf(str(out))
    return str(out)


def _run(model_name: str, inputs: MeteoInputs, fixtures: dict) -> Catchment:
    """Build a distributed Coello model on `inputs` and run Muskingum routing."""
    coello = Catchment(
        model_name,
        fixtures["start"],
        fixtures["end"],
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    coello.read_meteo(inputs)
    coello.read_flow_acc(fixtures["acc"])
    coello.read_flow_dir(fixtures["fd"])
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
        self, from_netcdf_files: MeteoInputs, combined_netcdf: str
    ):
        """Test that one file holding all three variables loads the same cubes.

        Test scenario:
            Same values, different packaging: the caller names which variable is which, and the
            result must be indistinguishable from the one-file-per-variable load.
        """
        combined = MeteoInputs.from_netcdf(
            combined_netcdf,
            precipitation="Band_1",
            temperature="Band_2",
            evapotranspiration="Band_3",
        )
        for name in METEO_VARIABLES:
            np.testing.assert_allclose(
                getattr(combined, name),
                getattr(from_netcdf_files, name),
                err_msg=f"{name} differs when read from the combined NetCDF",
            )

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
        with pytest.raises(KeyError, match="Band_1"):
            MeteoInputs.from_netcdf(
                f"{NC_DIR}/prec.nc",
                precipitation="nope",
                temperature="Band_1",
                evapotranspiration="Band_1",
            )

    def test_multi_variable_file_needs_an_explicit_variable(self, combined_netcdf: str):
        """Test that from_netcdf_files refuses an ambiguous file.

        Test scenario:
            Handing a three-variable file to the per-variable loader is a mistake; silently
            taking the first variable would load precipitation three times.
        """
        with pytest.raises(ValueError, match="from_netcdf"):
            MeteoInputs.from_netcdf_files(
                combined_netcdf, combined_netcdf, combined_netcdf
            )


class TestRunEquivalence:
    """Tests that the model runs identically whichever source fed it."""

    def test_read_meteo_matches_the_raster_readers(
        self,
        from_rasters: MeteoInputs,
        fixtures: dict,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_evap_path: str,
        raster_kwargs: dict,
    ):
        """Test that read_meteo sets the same state as the three raster readers.

        Test scenario:
            `read_meteo` has to be a drop-in for `read_rainfall` + `read_temperature` +
            `read_evapotranspiration`, including the derived `time_steps` and `ll_temp` that
            the run depends on.
        """
        via_readers = Catchment(
            "readers",
            fixtures["start"],
            fixtures["end"],
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
        )
        via_readers.read_rainfall(coello_prec_path, **raster_kwargs)
        via_readers.read_temperature(coello_temp_path, **raster_kwargs)
        via_readers.read_evapotranspiration(coello_evap_path, **raster_kwargs)

        via_structure = Catchment(
            "structure",
            fixtures["start"],
            fixtures["end"],
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
        )
        via_structure.read_meteo(from_rasters)

        for name in METEO_VARIABLES:
            np.testing.assert_array_equal(
                getattr(via_structure, name),
                getattr(via_readers, name),
                err_msg=f"{name} differs between read_meteo and the raster readers",
            )
        assert via_structure.time_steps == via_readers.time_steps
        np.testing.assert_allclose(
            via_structure.ll_temp,
            via_readers.ll_temp,
            err_msg="ll_temp must match what read_temperature derives",
        )

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
