import ast
import tomllib
import warnings
from math import ceil
from pathlib import Path

import numpy as np
import pytest
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube
from pyramids.feature import FeatureCollection
from shapely.geometry import Polygon

from hapi.inputs import PARAMETERS_LIST, Inputs

# UTM 18N covers the Coello basin, which the fixtures carry in geographic coordinates.
PARAMETER_EPSG = 32618
PARAMETER_CELL_SIZE = 5_000.0
PARAMETER_MARGIN = 10_000.0
# Added to the border ring so an unmasked read is far from the in-basin value.
PARAMETER_OUTSIDE_OFFSET = 1_000


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
    @pytest.mark.fig_share
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
    #     coello_basin: FeatureCollection,
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


@pytest.mark.fig_share
def test_extract_parameters_boundaries(
    download_max_min_parameter, coello_basin: FeatureCollection
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

    def test_create_lumped_inputs_orders_by_parsed_date(self, tmp_path):
        """Test that Hapi reads a raster folder chronologically without a rename step.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            This is the capability that justified deleting ``Inputs.rename_files``, whose
            only purpose was to rewrite file names with an order prefix. Three uniform
            rasters are written whose dates are deliberately not in name order, each with
            a distinct value, and read through ``Inputs.create_lumped_inputs`` — Hapi's
            own consumer of ordered reading. Its per-raster spatial means must come back
            chronologically. Exercising ``Inputs`` rather than ``Datacube`` directly is
            what makes this a test of Hapi: the previous version called pyramids straight
            and passed unchanged against ``main``.
        """
        for stamp, value in (
            ("2020.02.01", 3.0),
            ("2020.01.02", 1.0),
            ("2020.01.10", 2.0),
        ):
            Dataset.create_from_array(
                np.full((2, 2), value, dtype="float32"),
                top_left_corner=(0.0, 2.0),
                cell_size=1.0,
                epsg=4326,
                no_data_value=-9999.0,
                path=str(tmp_path / f"prec_{stamp}.tif"),
            ).close()

        averages = Inputs.create_lumped_inputs(
            str(tmp_path),
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )

        assert [float(v) for v in averages] == [1.0, 2.0, 3.0], (
            "Hapi must return the catchment averages in date order, so no rename step "
            f"is needed; got {[float(v) for v in averages]}"
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


class TestVectorTypes:
    """Tests that Hapi speaks pyramids' vector type and no longer needs geopandas."""

    @pytest.fixture(scope="function")
    def basin(self):
        """Return a single-polygon catchment as a plain GeoDataFrame.

        Returns:
            GeoDataFrame: One square polygon in WGS 84.
        """
        return FeatureCollection(
            {"id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )

    @pytest.fixture(scope="function")
    def utm_parameter_set(self, tmp_path, coello_basin, monkeypatch) -> str:
        """Write the 18 HBV parameter rasters in UTM 18N over the Coello basin.

        Each raster holds its own 1-based position in `PARAMETERS_LIST` over the basin
        bounding box, and `PARAMETER_OUTSIDE_OFFSET` above that in a border ring that the
        basin cannot reach. A statistic read back therefore says both which file it came
        from and whether the basin was applied as a mask: an unmasked read would report
        the ring value instead.

        The rasters are projected while the basin is geographic, matching the real Beck
        et al. sets, which are not necessarily in the catchment's CRS.

        Args:
            tmp_path: Pytest temporary directory.
            coello_basin: The Coello catchment polygon fixture.
            monkeypatch: Pytest monkeypatch fixture, used to point `HAPI_DATA_DIR` here.

        Returns:
            str: The scenario name to pass to `extract_parameters`.
        """
        scenario = "1"
        directory = tmp_path / scenario
        directory.mkdir()

        x_min, y_min, x_max, y_max = coello_basin.to_crs(
            crs=PARAMETER_EPSG
        ).total_bounds
        ring = int(PARAMETER_MARGIN / PARAMETER_CELL_SIZE)
        cols = 2 * ring + ceil((x_max - x_min) / PARAMETER_CELL_SIZE)
        rows = 2 * ring + ceil((y_max - y_min) / PARAMETER_CELL_SIZE)

        for position, name in enumerate(PARAMETERS_LIST, start=1):
            array = np.full(
                (rows, cols), position + PARAMETER_OUTSIDE_OFFSET, dtype=np.float32
            )
            array[ring:-ring, ring:-ring] = position
            Dataset.create_from_array(
                array,
                geo=(
                    x_min - PARAMETER_MARGIN,
                    PARAMETER_CELL_SIZE,
                    0.0,
                    y_max + PARAMETER_MARGIN,
                    0.0,
                    -PARAMETER_CELL_SIZE,
                ),
                epsg=PARAMETER_EPSG,
                no_data_value=-9999.0,
                path=str(directory / f"{name}.tif"),
            ).close()

        monkeypatch.setenv("HAPI_DATA_DIR", str(tmp_path))
        return scenario

    def test_extract_parameters_accepts_a_frame(self, coello_basin, utm_parameter_set):
        """Test the zonal-statistics branch of `extract_parameters`.

        Args:
            coello_basin: The Coello catchment polygon fixture.
            utm_parameter_set: Synthetic projected parameter set; its name is the scenario.

        Test scenario:
            `as_raster=False` had no live coverage — the only other test for the method
            passes `as_raster=True`, which takes the other branch, and it reaches its data
            through a live FigShare download besides. Exercising Hapi here rather than
            calling `FeatureCollection(...)` directly is the point: a test that only wraps
            a frame proves nothing about Hapi and passes against `main` too.

            The parameter set is built here rather than downloaded because the real one
            comes from FigShare into `HAPI_DATA_DIR`, which is empty on CI.

            Each raster carries its own position over the basin and a higher value in a
            border ring, so the statistics below pin three things: the rows come back one
            per parameter in `PARAMETERS_LIST` order, each row's statistic came from its
            own file, and the basin was applied as a mask — reading a whole raster would
            report the ring value instead.
        """
        inputs = Inputs("tests/rrm/data/coello/gis/acc4000.tif")

        stats = inputs.extract_parameters(
            coello_basin, utm_parameter_set, as_raster=False
        )

        positions = [float(position) for position in range(1, len(PARAMETERS_LIST) + 1)]
        assert list(stats.columns) == ["min", "max", "mean", "std"], (
            f"expected the four stat columns, got {list(stats.columns)}"
        )
        assert list(stats.index) == PARAMETERS_LIST, (
            f"expected one row per HBV parameter in order, got {list(stats.index)}"
        )
        assert stats["mean"].astype(float).tolist() == pytest.approx(positions), (
            "each raster should report its own position as the in-basin mean; got "
            f"{stats['mean'].tolist()}"
        )
        assert stats["max"].astype(float).tolist() == pytest.approx(positions), (
            "a max above the position means the out-of-basin ring was read, so the "
            f"basin was not applied as a mask; got {stats['max'].tolist()}"
        )

    def test_hapi_does_not_import_geopandas(self):
        """Test that no Hapi module imports geopandas directly.

        Test scenario:
            geopandas was dropped from ``[project].dependencies`` because nothing in
            ``src/hapi`` imports it any more — it arrives transitively through
            pyramids-gis. That matters beyond tidiness: pyramids excludes geopandas on
            win_arm64 (it vendors the vector stack there), and Hapi's unconditional pin
            overrode that exclusion. This scan keeps the import from creeping back and
            re-breaking that platform.
        """
        offenders = []
        # tests/ too: the runtime dependency was dropped, so a geopandas import anywhere
        # the suite loads breaks collection on win_arm64, where it cannot be installed.
        repo_root = Path(__file__).resolve().parents[2]
        roots = [repo_root / "src" / "hapi", repo_root / "tests"]
        for module in sorted(m for root in roots for m in root.rglob("*.py")):
            with warnings.catch_warnings():
                # Parsing every test module surfaces unrelated pre-existing escape
                # warnings (e.g. a Windows path in a calibration fixture); this guard is
                # about imports, not lexical hygiene.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name.split(".")[0] == "geopandas" for name in names):
                    offenders.append(f"{module}:{node.lineno}")

        assert not offenders, (
            "these modules import geopandas directly; use pyramids.feature."
            f"FeatureCollection instead: {offenders}"
        )

    def test_geopandas_is_not_a_declared_dependency(self):
        """Test that geopandas is absent from the project's runtime dependencies.

        Test scenario:
            Pairs with the import scan: the declaration and the imports must agree, so a
            future edit cannot reinstate the pin without also reinstating a use for it.
        """
        repo_root = Path(__file__).resolve().parents[2]
        manifest = tomllib.loads(
            (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = manifest["project"]["dependencies"]

        assert not any("geopandas" in spec for spec in declared), (
            "geopandas is declared again in [project].dependencies but nothing in "
            "src/hapi imports it; it arrives transitively via pyramids-gis, which "
            "deliberately excludes it on win_arm64"
        )
