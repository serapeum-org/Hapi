from pathlib import Path

import numpy as np
import pytest
from geopandas import GeoDataFrame
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube

from hapi.inputs import Inputs


def test_prepare_inputs(
    coello_prec_path: str, coello_acc_path: str, rrm_test_results: str
):
    """Test prepare_inputs function in Inputs class"""
    rpath = Path(f"{rrm_test_results}/prepare_inputs")
    # if rpath.exists():
    #     rpath.unlink()

    inputs = Inputs(coello_acc_path)
    inputs.prepare_inputs(coello_prec_path, rpath)
    assert rpath.exists()
    files = list(rpath.iterdir())
    assert len(files) == 10
    cube = Datacube.read_multiple_files(str(rpath), with_order=False)
    # if rpath.exists():
    #     rpath.unlink()


class TestExtractParameters:
    def test_as_raster(
        self,
        download_03_parameter,
        coello_prec_path: str,
        coello_acc_path: str,
        rrm_test_results: str,
    ):
        """Test extract_parameters function in Inputs class"""
        rpath = Path(f"{rrm_test_results}/extract_parameter")
        # if rpath.exists():
        #     rpath.unlink()

        inputs = Inputs(coello_acc_path)
        inputs.extract_parameters(None, "3", as_raster=True, save_to=str(rpath))
        assert rpath.exists()
        files = list(rpath.iterdir())
        assert len(files) == 19
        cube = Datacube.read_multiple_files(str(rpath), with_order=False)
        # if rpath.exists():
        #     rpath.unlink()

    # def test_as_raster_false(
    #     self,
    #     download_03_parameter,
    #     coello_acc_path: str,
    #     coello_basin: GeoDataFrame,
    # ):
    #     """Test extract_parameters function in Inputs class"""
    #     inputs = Inputs(coello_acc_path)
    #     par = inputs.extract_parameters(coello_basin, "03")
    #     par_vals = [
    #         0.8952,
    #         1.0,
    #         1.230,
    #         3.099,
    #         0.07358,
    #         0.05464,
    #         548.72,
    #         3.085,
    #         1.0,
    #         0.911,
    #         0.8657,
    #         0.5961,
    #         0.09381,
    #         38.313,
    #         3.919,
    #         1.873,
    #         1.0,
    #         0.20,
    #     ]
    #     assert np.isclose(
    #         par.loc[:, "max"].to_list(), par_vals, atol=0.001, rtol=0.001
    #     ).all()


def test_extract_parameters_boundaries(
    download_max_min_parameter, coello_basin: GeoDataFrame
):
    """Test extract_parameters function in Inputs class"""
    par = Inputs.extract_parameters_boundaries(coello_basin)
    upper_bound_valid = [
        2.262565,
        1,
        1.49494,
        4.502295,
        0.138411,
        0.079819,
        608.27124,
        3.669066,
        1,
        1,
        0.865717,
        0.8,
        0.107622,
        72.304459,
        5.275979,
        2.34628,
        1,
        0.2,
    ]
    lower_bound_valid = [
        -0.966476,
        1,
        1.044623,
        0.55258,
        0.035901,
        0.011214,
        50,
        1.148444,
        1,
        0.460137,
        0.227757,
        0.123802,
        0.005037,
        16.123743,
        1.657871,
        1.194185,
        1,
        0.2,
    ]
    assert np.isclose(
        par.loc[:, "ub"], upper_bound_valid, rtol=0.00001, atol=0.00001
    ).all()
    assert np.isclose(
        par.loc[:, "lb"], lower_bound_valid, rtol=0.00001, atol=0.00001
    ).all()


def test_create_lumped_parameter():
    path = "tests/rrm/data/coello/prec"
    lumped_data = Inputs.create_lumped_inputs(
        path,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    # independent expectation: numpy mean over the valid (non-nodata) cells of
    # each raster, bypassing the GDAL band-statistics path that
    # create_lumped_inputs relies on (pyramids >= 0.32 returns exact band
    # statistics instead of the sampled values returned by older versions)
    cube = Datacube.read_multiple_files(
        path,
        with_order=True,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    expected = []
    for i in range(cube.time_length):
        dataset = cube.iloc(i)
        arr = dataset.read_array(band=0).astype(np.float64)
        valid = arr[~np.isclose(arr, dataset.no_data_value[0], rtol=1e-5)]
        expected.append(valid.mean())
    assert len(lumped_data) == 10
    assert np.isclose(lumped_data, expected, atol=0.001, rtol=0.001).all()
    # anchor a couple of values against fixed references so a regression in the
    # raw raster reads cannot silently shift both sides of the comparison
    assert np.isclose(lumped_data[1], 0.2987198, atol=0.001)
    assert np.isclose(lumped_data[2], 44.0648258, atol=0.001)


class TestChronologicalReading:
    """Tests for the date-ordered reading that replaced ``Inputs.rename_files``."""

    def test_read_multiple_files_orders_by_parsed_date(self, tmp_path):
        """Test that rasters are read chronologically regardless of directory order.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            This is the capability that justified deleting ``Inputs.rename_files``, whose
            only purpose was to rewrite file names with an order prefix so a later
            directory read picked them up in sequence. Three rasters are written whose
            dates are deliberately not in creation order, each carrying a distinct pixel
            value identifying its date, and read back with ``with_order=True``. The values
            must come back chronologically, so no rename step is needed.
        """
        stamps = {"2020.02.01": 3, "2020.01.02": 1, "2020.01.10": 2}
        for stamp, value in stamps.items():
            Dataset.create_from_array(
                np.full((2, 2), value, dtype="int32"),
                top_left_corner=(0.0, 2.0),
                cell_size=1.0,
                epsg=4326,
                no_data_value=-9999,
                path=str(tmp_path / f"prec_{stamp}.tif"),
            ).close()

        cube = Datacube.read_multiple_files(
            str(tmp_path),
            with_order=True,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        first_pixel = [
            int(cube.iloc(i).read_array(band=0)[0, 0]) for i in range(cube.time_length)
        ]

        assert first_pixel == [1, 2, 3], (
            "rasters should be ordered by the date parsed from the file name, so no "
            f"rename step is needed; got {first_pixel}"
        )

    def test_rename_files_is_gone(self):
        """Test that the deleted helper is no longer on the public surface.

        Test scenario:
            ``Inputs.rename_files`` was removed because pyramids already orders by the
            parsed date. Pinning its absence keeps a merge from quietly reintroducing it.
        """
        assert not hasattr(Inputs, "rename_files"), (
            "Inputs.rename_files was deleted; read_multiple_files(with_order=True) "
            "supersedes it"
        )


class TestPrepareInputs:
    """Tests for ``Inputs.prepare_inputs``."""

    def test_missing_inputs_dir_raises_before_opening_the_dem(self, tmp_path):
        """Test that a missing input directory fails fast, without reading the DEM.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            The validation used to run *after* ``Dataset.read_file(self.source_dem)``, so
            a missing input directory surfaced only once the DEM had been opened — and a
            bogus DEM path raised the wrong error first. The DEM path here does not exist
            either; the ``FileNotFoundError`` must still name the input directory.
        """
        missing = tmp_path / "absent"

        with pytest.raises(FileNotFoundError) as exc_info:
            Inputs("dem-that-does-not-exist.tif").prepare_inputs(
                missing, tmp_path / "out"
            )

        assert str(missing) in str(exc_info.value), (
            "the error should name the missing input directory, not the DEM; got "
            f"{exc_info.value}"
        )

    def test_accepts_path_objects_for_both_directories(self, tmp_path):
        """Test that ``pathlib.Path`` arguments are accepted for input and output.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            The signature is annotated ``str | Path``, but the body used to warn (via a
            bare ``print``) whenever ``outputs_dir`` was not a ``str`` — contradicting the
            annotation while doing nothing. Passing ``Path`` for both arguments must work
            silently and write the aligned rasters.
        """
        dem_path = tmp_path / "dem.tif"
        Dataset.create_from_array(
            np.ones((4, 4), dtype="float32"),
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
            path=str(dem_path),
        ).close()

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        Dataset.create_from_array(
            np.full((4, 4), 5.0, dtype="float32"),
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
            path=str(src_dir / "prec_2020.01.01.tif"),
        ).close()

        out_dir = tmp_path / "out"
        Inputs(str(dem_path)).prepare_inputs(src_dir, out_dir)

        assert [f.name for f in sorted(out_dir.iterdir())] == ["prec_2020.01.01.tif"], (
            f"expected the source file name to be preserved, got {list(out_dir.iterdir())}"
        )
