"""Unit tests for the Catchment raster readers that delegate no-data masking to pyramids.

Covers ``Catchment.read_flow_acc``, ``Catchment.read_flow_dir`` and
``Catchment.read_flow_path_length`` — the three readers changed on
``refactor/delegate-gis-to-pyramids``, where a hand-rolled
``math.isclose(..., rel_tol=0.001)`` loop was replaced by
``Dataset.read_array(masked=True)``.

The behavioural centrepiece is the *near-sentinel* case: the old tolerance masked any
value within 0.1% of the no-data marker, so a legitimate ``-9990`` beside a ``-9999``
marker was silently rewritten to ``NaN``. pyramids compares integer bands for exact
equality, so such values now survive.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from pyramids.dataset import Dataset

from hapi.catchment import Catchment

CELL_SIZE = 4000.0
"""Cell size in metres used by every synthetic raster here (matches the Coello grid)."""

EPSG = 32618
"""Projected CRS (UTM 18N) used by every synthetic raster here (matches the Coello grid)."""

NO_DATA = -9999
"""No-data sentinel stamped on every synthetic raster here."""

NEAR_SENTINEL = -9990
"""Legitimate value within 0.1% of :data:`NO_DATA`.

``math.isclose(-9990, -9999, rel_tol=0.001)`` is ``True`` (the tolerance is
``0.001 * 9999 == 9.999`` and the gap is ``9``), so the pre-change loop masked this value.
Exact integer comparison keeps it.
"""


@pytest.fixture(scope="function")
def write_raster(tmp_path):
    """Return a factory that writes a small GeoTIFF and yields its path.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Callable[..., str]: A factory taking the pixel ``array`` and, optionally, a
            ``no_data_value``, file ``name``, and ``pixel_height`` (to build a
            non-square grid), returning the path of the written GeoTIFF.
    """

    def _write(
        array: np.ndarray,
        *,
        no_data_value: int | float | None = NO_DATA,
        name: str = "raster.tif",
        pixel_height: float | None = None,
    ) -> str:
        path = tmp_path / name
        height = CELL_SIZE if pixel_height is None else pixel_height
        Dataset.create_from_array(
            array,
            geo=(0.0, CELL_SIZE, 0.0, array.shape[0] * height, 0.0, -height),
            epsg=EPSG,
            no_data_value=no_data_value,
            path=str(path),
        ).close()
        return str(path)

    return _write


@pytest.fixture(scope="function")
def catchment() -> Catchment:
    """Return a minimal distributed Catchment with a two-day window.

    Returns:
        Catchment: An instance with no inputs read yet, ready for a ``read_*`` call.
    """
    return Catchment(
        "test", "2000-01-01", "2000-01-02", spatial_resolution="Distributed"
    )


class TestReadFlowAcc:
    """Tests for ``Catchment.read_flow_acc``."""

    def test_masks_no_data_cells_to_nan(self, catchment, write_raster):
        """Test that cells equal to the no-data sentinel become NaN.

        Test scenario:
            A 2x2 integer raster whose bottom-right cell holds the ``-9999`` sentinel.
            Only that cell must be NaN; the three real accumulation values are untouched.
        """
        path = write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert np.isnan(catchment.flow_acc_arr[1, 1]), (
            f"the no-data cell should be NaN, got {catchment.flow_acc_arr[1, 1]}"
        )
        assert not np.isnan(catchment.flow_acc_arr[:1, :]).any(), (
            f"real accumulation values must not be masked, got {catchment.flow_acc_arr}"
        )
        assert catchment.flow_acc_arr[0, 1] == 1.0, (
            f"expected the value 1 to survive unchanged, got {catchment.flow_acc_arr[0, 1]}"
        )

    def test_preserves_value_near_sentinel(self, catchment, write_raster):
        """Test that a real value within 0.1% of the sentinel is no longer destroyed.

        Test scenario:
            The regression guarding the behaviour change on this branch. A raster with a
            ``-9999`` marker also holds a legitimate ``-9990``. The pre-change
            ``math.isclose(..., rel_tol=0.001)`` loop masked it; exact integer comparison
            keeps it, so no cell is NaN and every cell counts toward the domain.
        """
        path = write_raster(np.array([[0, 1], [2, NEAR_SENTINEL]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert not np.isnan(catchment.flow_acc_arr).any(), (
            f"no cell holds the sentinel, so nothing should be masked, got {catchment.flow_acc_arr}"
        )
        assert catchment.flow_acc_arr[1, 1] == float(NEAR_SENTINEL), (
            f"expected {NEAR_SENTINEL} to survive masking, got {catchment.flow_acc_arr[1, 1]}"
        )
        assert catchment.no_elem == 4, (
            f"all 4 cells are real data, got no_elem={catchment.no_elem}"
        )

    def test_missing_sentinel_warns(self, catchment, write_raster):
        """Test that a raster with no no-data marker warns instead of failing silently.

        Test scenario:
            Before masking was delegated, such a raster raised ``TypeError`` from
            ``math.isclose(value, None)`` — accidental, but loud. pyramids masks nothing,
            which is the right reading but silently makes the whole grid the domain. A
            raster legitimately having no marker is valid input, so this warns rather
            than raising.
        """
        path = write_raster(
            np.array([[0, 1], [2, 3]], dtype="int32"),
            no_data_value=None,
            name="acc_nosentinel.tif",
        )

        with pytest.warns(UserWarning, match="declares no no-data value"):
            catchment.read_flow_acc(path)

        assert catchment.no_elem == 4, (
            f"with no sentinel every cell is domain, got {catchment.no_elem}"
        )

    def test_counts_domain_cells_excluding_no_data(self, catchment, write_raster):
        """Test that ``no_elem`` counts only cells inside the domain.

        Test scenario:
            A 2x3 raster with two sentinel cells leaves four real cells.
        """
        path = write_raster(np.array([[0, 1, 2], [3, NO_DATA, NO_DATA]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert catchment.no_elem == 4, (
            f"expected 4 domain cells out of 6, got {catchment.no_elem}"
        )

    def test_domain_count_does_not_use_fuzzy_tolerance(self, catchment, write_raster):
        """Test that ``no_elem`` is counted from the masked array, not by a fuzzy comparison.

        Test scenario:
            Guards against "simplifying" the count to ``Dataset.count_domain_cells()``.
            That helper re-reads the raster and compares with ``is_no_data``'s default
            relative tolerance, which masks values within 0.1% of the sentinel: on this
            raster it reports 3 domain cells where the masked array correctly has 4.
            ``no_elem`` must follow the masked array.
        """
        path = write_raster(np.array([[0, 1], [2, NEAR_SENTINEL]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert catchment.no_elem == 4, (
            "no_elem must be counted from the pyramids-masked array; a value 0.1% away "
            f"from the sentinel is real data, got no_elem={catchment.no_elem}"
        )
        # Deliberately not asserting count_domain_cells() == 3 here. That would pin an
        # upstream defect: a correct fix in pyramids would turn this red on a branch that
        # changed nothing. What matters is Hapi's own contract, asserted above.
        helper_count = Dataset.read_file(path).count_domain_cells()
        if helper_count == catchment.no_elem:
            pytest.skip(
                "pyramids' count_domain_cells now agrees with the masked array; the "
                "helper is safe to adopt and this guard can be removed"
            )

    def test_gdal_mask_band_shrinks_the_domain(self, catchment, write_raster):
        """Test that an internal GDAL mask band excludes cells from the domain.

        Test scenario:
            Documents a deliberate behaviour change. Masking used to compare against the
            no-data value only; delegating to pyramids also honours the band's GDAL mask,
            so a raster carrying one yields a *smaller* domain than before. That changes
            ``no_elem`` and, through it, the width of the parameter arrays — calibration
            vectors saved against the old domain will not fit. Keeping the behaviour is
            the correct reading of such a raster; pinning it is what stops the change
            being silent.
        """
        path = write_raster(
            np.array([[0, 1], [2, 3]], dtype="int32"), name="acc_maskband.tif"
        )
        ds = Dataset.read_file(path, read_only=False)
        ds.create_mask_band()
        band = ds.raster.GetRasterBand(1).GetMaskBand()
        band.WriteArray(np.array([[255, 255], [0, 255]], dtype="uint8"))
        ds.raster.FlushCache()
        ds.close()

        catchment.read_flow_acc(path)

        assert catchment.no_elem == 3, (
            "the cell zeroed in the mask band must be excluded from the domain, so 3 of "
            f"4 cells remain; got {catchment.no_elem}"
        )

    def test_returns_float_array_for_integer_raster(self, catchment, write_raster):
        """Test that an integer raster is promoted to float so NaN is representable.

        Test scenario:
            An ``int32`` source must come back as a floating array — an integer array
            cannot hold NaN, so the promotion is what makes masking expressible at all.
        """
        path = write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert catchment.flow_acc_arr.dtype.kind == "f", (
            f"expected a floating dtype so NaN can be stored, got {catchment.flow_acc_arr.dtype}"
        )

    def test_acc_val_is_sorted_unique_python_ints(self, catchment, write_raster):
        """Test that ``acc_val`` is a sorted, de-duplicated list of built-in ints.

        Test scenario:
            The raster repeats the value ``1`` and holds them out of order. ``acc_val``
            must collapse duplicates, sort ascending, and yield Python ``int`` rather
            than ``numpy.int64`` — downstream code compares it against plain lists.
        """
        path = write_raster(np.array([[2, 1], [1, 0]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert catchment.acc_val == [0, 1, 2], (
            f"expected the sorted distinct values [0, 1, 2], got {catchment.acc_val}"
        )
        assert all(type(value) is int for value in catchment.acc_val), (
            f"expected built-in ints, got {[type(v).__name__ for v in catchment.acc_val]}"
        )

    def test_acc_val_truncates_before_deduplicating(self, catchment, write_raster):
        """Test that a float raster yields distinct *integer* accumulation values.

        Test scenario:
            The regression guarding H1. ``np.unique`` on the float values keeps 1.2 and
            1.8 apart and only then truncates, producing a repeated ``1``; truncating
            first collapses them, which is what the per-cell ``set(int(...))`` this
            replaced did. The sibling test uses an int32 raster, where the ordering
            cannot matter, so it could never have caught this.
        """
        path = write_raster(
            np.array([[0.0, 1.2], [1.8, NO_DATA]], dtype="float32"),
            no_data_value=float(NO_DATA),
            name="acc_float.tif",
        )

        catchment.read_flow_acc(path)

        assert catchment.acc_val == [0, 1], (
            "float cells must be truncated before de-duplication, so 1.2 and 1.8 "
            f"collapse to a single 1; got {catchment.acc_val}"
        )

    def test_infinite_values_are_rejected(self, catchment, write_raster):
        """Test that a raster holding infinity fails loudly rather than saturating.

        Test scenario:
            ``astype(int)`` maps ``inf`` to ``INT64_MIN`` with only a ``RuntimeWarning``,
            so a corrupt raster would silently acquire a huge negative accumulation
            value. The conversion rejects it instead (L2).
        """
        path = write_raster(
            np.array([[0.0, np.inf], [1.0, NO_DATA]], dtype="float32"),
            no_data_value=float(NO_DATA),
            name="acc_inf.tif",
        )

        with pytest.raises(ValueError, match="infinite values"):
            catchment.read_flow_acc(path)

    def test_all_no_data_raster_raises(self, catchment, write_raster):
        """Test the degenerate all-no-data raster.

        Test scenario:
            Every cell is the sentinel, so no accumulation values survive and the
            max-value sanity check has nothing to take a maximum of. This documents
            pre-existing behaviour — the same ``ValueError`` was raised before the
            vectorisation — rather than endorsing it as a good error message.
        """
        path = write_raster(np.full((2, 2), NO_DATA, dtype="int32"))

        with pytest.raises(ValueError, match="empty"):
            catchment.read_flow_acc(path)

    def test_outlet_is_cell_of_maximum_accumulation(self, catchment, write_raster):
        """Test that the outlet is located at the maximum accumulation cell.

        Test scenario:
            Accumulation increases toward ``(1, 0)``, which must be reported as the outlet.
            The masked cell must not win despite its large magnitude sentinel.
        """
        path = write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert catchment.Outlet[0][0] == 1 and catchment.Outlet[1][0] == 0, (
            f"expected the outlet at row 1, column 0, got {catchment.Outlet}"
        )

    def test_derives_cell_size_and_pixel_area(self, catchment, write_raster):
        """Test that cell size and pixel areas are derived from the geotransform.

        Test scenario:
            A 4000 m grid gives a 16 km^2 pixel; three domain cells give 48 km^2 total.
        """
        path = write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert catchment.cell_size == pytest.approx(CELL_SIZE), (
            f"expected a {CELL_SIZE} m cell, got {catchment.cell_size}"
        )
        assert catchment.px_area == pytest.approx(16.0), (
            f"expected a 16 km2 pixel for a {CELL_SIZE} m cell, got {catchment.px_area}"
        )
        assert catchment.px_tot_area == pytest.approx(48.0), (
            f"expected 3 cells x 16 km2 = 48 km2, got {catchment.px_tot_area}"
        )

    def test_non_square_cells_use_both_axes(self, catchment, write_raster):
        """Test that pixel area uses width and height separately, not cell size squared.

        Test scenario:
            A grid of 4000 m wide by 2000 m tall cells. ``cell_size`` reports the pixel
            *width* only (that is what ``Dataset.cell_size`` means), while ``px_area``
            multiplies both axes — 4 km x 2 km = 8 km^2, not 16 km^2. Three domain cells
            give 24 km^2.

            This is not a regression test: ``main`` already read the two pixel dimensions
            separately and produced the same answer. It exists because nothing covered
            the non-square case at all, so the switch to named transform fields had no
            guard against getting it wrong.
        """
        path = write_raster(
            np.array([[0, 1], [2, NO_DATA]], dtype="int32"),
            name="acc_rect.tif",
            pixel_height=2000.0,
        )

        catchment.read_flow_acc(path)

        assert catchment.cell_size == pytest.approx(4000.0), (
            f"cell_size should report the pixel width, got {catchment.cell_size}"
        )
        assert catchment.px_area == pytest.approx(8.0), (
            f"expected 4 km x 2 km = 8 km2 for a non-square cell, got {catchment.px_area}"
        )
        assert catchment.px_tot_area == pytest.approx(24.0), (
            f"expected 3 cells x 8 km2 = 24 km2, got {catchment.px_tot_area}"
        )

    def test_cell_size_is_a_magnitude(self, catchment, write_raster):
        """Test that ``cell_size`` is positive even on a flipped grid.

        Test scenario:
            ``Dataset.cell_size`` returns the *signed* geotransform pixel width. A grid
            written west-to-east flipped therefore reports a negative width, while every
            consumer of ``cell_size`` treats it as a magnitude — as the value it replaced
            was, having been ``abs()``-ed.
        """
        path = str(Path(write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32"))))
        flipped = Dataset.read_file(path)
        gt = list(flipped.geotransform)
        gt[1] = -gt[1]
        flipped_path = path.replace(".tif", "_flipped.tif")
        Dataset.create_from_array(
            flipped.read_array(band=0),
            geo=tuple(gt),
            epsg=EPSG,
            no_data_value=NO_DATA,
            path=flipped_path,
        ).close()

        catchment.read_flow_acc(flipped_path)

        assert catchment.cell_size > 0, (
            f"cell_size must be a magnitude, got {catchment.cell_size}"
        )
        assert catchment.cell_size == pytest.approx(CELL_SIZE), (
            f"expected {CELL_SIZE}, got {catchment.cell_size}"
        )

    def test_missing_file_raises(self, catchment, tmp_path):
        """Test that a path which does not exist is rejected by pyramids.

        Test scenario:
            Validation is delegated to ``Dataset.read_file``, so a missing path raises
            ``FileNotFoundError`` rather than the ``AssertionError`` the hand-rolled
            guard used to produce.
        """
        with pytest.raises(FileNotFoundError, match="does not exist"):
            catchment.read_flow_acc(str(tmp_path / "absent.tif"))

    def test_unreadable_file_raises(self, catchment, tmp_path):
        """Test that a file GDAL cannot open is rejected.

        Test scenario:
            The guard no longer keys off the file extension — it lets GDAL decide. A file
            that is not a raster at all surfaces as a GDAL ``RuntimeError`` naming the
            path, instead of an extension complaint that could not tell a mislabelled
            GeoTIFF from genuine garbage.
        """
        other = tmp_path / "grid.asc"
        other.write_text("not a geotiff", encoding="utf-8")

        with pytest.raises(RuntimeError):
            catchment.read_flow_acc(str(other))

    def test_path_object_is_accepted(self, catchment, write_raster):
        """Test that a ``pathlib.Path`` is a valid argument.

        Test scenario:
            pyramids accepts ``str`` and ``Path`` alike. The removed
            ``assert isinstance(path, str)`` rejected ``Path`` outright, contradicting
            both pyramids and the surrounding type hints.
        """
        path = Path(write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32")))

        catchment.read_flow_acc(path)

        assert catchment.no_elem == 3, (
            f"a Path argument should read exactly as a str does, got {catchment.no_elem}"
        )

    def test_non_tif_raster_is_accepted(self, catchment, write_raster, tmp_path):
        """Test that a valid raster is read regardless of its extension.

        Test scenario:
            The old guard demanded ``.tif`` and so rejected every other GDAL format. An
            ESRI ASCII grid is a legitimate flow-accumulation raster and must now load.
        """
        source = Dataset.read_file(
            write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32"))
        )
        asc_path = str(tmp_path / "acc.asc")
        source.to_file(asc_path)

        catchment.read_flow_acc(asc_path)

        assert catchment.no_elem == 3, (
            f"a valid .asc raster should read like a .tif, got {catchment.no_elem}"
        )

    def test_reads_coello_fixture(
        self, catchment, coello_acc_path, coello_rows, coello_cols, coello_acc_values
    ):
        """Test that the real Coello flow-accumulation raster is unchanged by the refactor.

        Args:
            coello_acc_path: Path fixture for the Coello accumulation raster.
            coello_rows: Expected row count of the Coello grid.
            coello_cols: Expected column count of the Coello grid.
            coello_acc_values: The distinct sorted accumulation values in the domain.

        Test scenario:
            Guards the production fixture against a regression in grid shape, in the set of
            distinct accumulation values recovered after masking, or in the domain-cell
            count (89 cells, matching the ``Par2d`` width asserted in
            ``tests/rrm/test_dist_parameters.py``).
        """
        catchment.read_flow_acc(coello_acc_path)

        assert (catchment.rows, catchment.cols) == (coello_rows, coello_cols), (
            f"expected a {coello_rows}x{coello_cols} grid, got {catchment.rows}x{catchment.cols}"
        )
        assert catchment.acc_val == coello_acc_values, (
            "the distinct accumulation values recovered after masking changed; "
            f"expected {coello_acc_values}, got {catchment.acc_val}"
        )
        assert catchment.no_elem == 89, (
            f"expected the 89 Coello domain cells to survive masking, got {catchment.no_elem}"
        )


class TestDirectoryReaders:
    """Tests for the meteorological readers that consume a directory of rasters."""

    @pytest.mark.parametrize("method", ["read_rainfall", "read_temperature", "read_et"])
    def test_missing_directory_error_names_the_path(self, catchment, tmp_path, method):
        """Test that a missing input directory is reported with its path.

        Args:
            tmp_path: pytest's per-test temporary directory.
            method: The reader under test.

        Test scenario:
            pyramids reports "The path you have provided does not exist" without saying
            which path. The hand-rolled checks this replaced named it, and a model run
            reads several directories, so the bare message does not identify the culprit.
        """
        missing = str(tmp_path / "absent")

        with pytest.raises(FileNotFoundError, match=re.escape(missing)):
            getattr(catchment, method)(missing)

    def test_empty_directory_error_names_the_path(self, catchment, tmp_path):
        """Test that an empty input directory is also reported with its path.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            pyramids raises the same bare ``FileNotFoundError`` for an existing but empty
            directory; the path must survive into the message here too.
        """
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(FileNotFoundError, match=re.escape(str(empty))):
            catchment.read_rainfall(str(empty))


class TestReadFlowDir:
    """Tests for ``Catchment.read_flow_dir``."""

    def test_masks_no_data_cells_to_nan(self, catchment, write_raster):
        """Test that sentinel cells become NaN and valid D8 codes survive.

        Test scenario:
            A 2x2 raster of ESRI D8 codes with one sentinel cell. Only that cell is NaN.
        """
        path = write_raster(
            np.array([[2, 4], [1, NO_DATA]], dtype="int32"), name="fd.tif"
        )

        catchment.read_flow_dir(path)

        assert np.isnan(catchment.flow_dir_arr[1, 1]), (
            f"the no-data cell should be NaN, got {catchment.flow_dir_arr[1, 1]}"
        )
        assert catchment.flow_dir_arr[0, 0] == 2.0, (
            f"the D8 code 2 should survive unchanged, got {catchment.flow_dir_arr[0, 0]}"
        )

    def test_near_sentinel_value_now_surfaces_as_invalid_code(
        self, catchment, write_raster
    ):
        """Test that a near-sentinel value is no longer silently swallowed.

        Test scenario:
            The counterpart of the flow-accumulation regression. ``-9990`` is not a valid
            ESRI D8 code. The pre-change loop masked it to NaN and the validation never
            saw it; exact comparison keeps it, so the D8 validation now rejects the raster
            instead of modelling corrupt data as no-data.
        """
        path = write_raster(
            np.array([[2, 4], [1, NEAR_SENTINEL]], dtype="int32"), name="fd_bad.tif"
        )

        with pytest.raises(
            AssertionError, match="flow direction raster should contain values"
        ):
            catchment.read_flow_dir(path)

    def test_invalid_direction_code_raises(self, catchment, write_raster):
        """Test that a non-D8 code is rejected.

        Test scenario:
            ``3`` is not a power of two in the ESRI encoding, so validation fails.
        """
        path = write_raster(
            np.array([[2, 4], [1, 3]], dtype="int32"), name="fd_three.tif"
        )

        with pytest.raises(
            AssertionError, match="flow direction raster should contain values"
        ):
            catchment.read_flow_dir(path)

    def test_repeated_direction_codes_are_accepted(self, catchment, write_raster):
        """Test that duplicated D8 codes validate once de-duplicated.

        Test scenario:
            All four cells share the code ``1``. Validation works on the distinct values,
            so a raster with one repeated direction is legitimate.
        """
        path = write_raster(
            np.array([[1, 1], [1, 1]], dtype="int32"), name="fd_dup.tif"
        )

        catchment.read_flow_dir(path)

        assert np.unique(catchment.flow_dir_arr).tolist() == [1.0], (
            f"expected a single distinct code, got {np.unique(catchment.flow_dir_arr).tolist()}"
        )

    def test_builds_flow_direction_table(self, catchment, write_raster):
        """Test that the upstream lookup table is built for every domain cell.

        Test scenario:
            A 2x2 raster with one sentinel cell yields a table keyed ``"row,col"`` holding
            one entry per non-masked cell.
        """
        path = write_raster(
            np.array([[2, 4], [1, NO_DATA]], dtype="int32"), name="fd_table.tif"
        )

        catchment.read_flow_dir(path)

        assert set(catchment.FDT) == {"0,0", "0,1", "1,0"}, (
            f"expected one table entry per domain cell, got {sorted(catchment.FDT)}"
        )

    def test_unreadable_file_raises(self, catchment, tmp_path):
        """Test that a file GDAL cannot open is rejected.

        Test scenario:
            Validation is delegated to ``DEM.read_file``, so a non-raster file surfaces as
            a GDAL ``RuntimeError`` rather than the previous extension ``ValueError``.
        """
        other = tmp_path / "directions.asc"
        other.write_text("not a geotiff", encoding="utf-8")

        with pytest.raises(RuntimeError):
            catchment.read_flow_dir(str(other))

    def test_missing_file_raises(self, catchment, tmp_path):
        """Test that a missing flow-direction raster is rejected by pyramids.

        Test scenario:
            ``FileNotFoundError`` from ``DEM.read_file``, matching the other readers.
        """
        with pytest.raises(FileNotFoundError, match="does not exist"):
            catchment.read_flow_dir(str(tmp_path / "absent.tif"))

    def test_fdt_and_flow_dir_arr_use_different_masks(self, catchment, write_raster):
        """Test the documented divergence between ``FDT`` and ``flow_dir_arr``.

        Test scenario:
            Pins the warning in the docstring. ``flow_dir_arr`` is masked by pyramids,
            but ``FDT`` comes from ``DEM.flow_direction_table``, which re-reads the raster
            and applies its own ``np.isclose(rtol=1e-5)``. A raster whose no-data value is
            also a valid D8 code shows the split: the cell is ``NaN`` in the array yet
            still keyed in the table. Reconciling them means changing ``hapi.dem``, which
            is slated to move to digital-rivers; this test makes sure the divergence
            cannot drift further unnoticed in the meantime.
        """
        path = write_raster(
            np.array([[2, 4], [1, 8]], dtype="int32"),
            no_data_value=8,
            name="fd_split.tif",
        )

        catchment.read_flow_dir(path)

        assert np.isnan(catchment.flow_dir_arr[1, 1]), (
            "the no-data cell should be masked in the array, got "
            f"{catchment.flow_dir_arr[1, 1]}"
        )
        assert "1,1" in catchment.FDT, (
            "FDT is built from an independent unmasked read, so the masked cell is still "
            f"keyed; if this now fails the two masks were reconciled, got {sorted(catchment.FDT)}"
        )

    def test_reads_coello_fixture(self, catchment, coello_fd_path, coello_fdt):
        """Test that the real Coello flow-direction raster is unchanged by the refactor.

        Args:
            coello_fd_path: Path fixture for the Coello flow-direction raster.
            coello_fdt: Expected upstream lookup table for the Coello grid.

        Test scenario:
            Guards the production fixture: the derived flow-direction table must match the
            table recorded before the masking change.
        """
        catchment.read_flow_dir(coello_fd_path)

        assert catchment.FDT == coello_fdt, (
            "the flow-direction table changed after delegating masking to pyramids"
        )


class TestReadFlowPathLength:
    """Tests for ``Catchment.read_flow_path_length``."""

    def test_masks_no_data_cells_to_nan(self, catchment, write_raster):
        """Test that sentinel cells become NaN and real lengths survive.

        Test scenario:
            A 2x2 raster of path lengths with one sentinel cell.
        """
        path = write_raster(
            np.array([[10, 20], [30, NO_DATA]], dtype="int32"), name="fpl.tif"
        )

        catchment.read_flow_path_length(path)

        assert np.isnan(catchment.flow_path_length_arr[1, 1]), (
            f"the no-data cell should be NaN, got {catchment.flow_path_length_arr[1, 1]}"
        )
        assert catchment.flow_path_length_arr[0, 0] == 10.0, (
            f"the path length 10 should survive unchanged, got {catchment.flow_path_length_arr[0, 0]}"
        )

    def test_preserves_value_near_sentinel(self, catchment, write_raster):
        """Test that a real length within 0.1% of the sentinel is no longer destroyed.

        Test scenario:
            ``-9990`` beside a ``-9999`` marker survives, so all four cells count toward
            the domain.
        """
        path = write_raster(
            np.array([[10, 20], [30, NEAR_SENTINEL]], dtype="int32"),
            name="fpl_near.tif",
        )

        catchment.read_flow_path_length(path)

        assert not np.isnan(catchment.flow_path_length_arr).any(), (
            f"no cell holds the sentinel, so nothing should be masked, got {catchment.flow_path_length_arr}"
        )
        assert catchment.no_elem == 4, (
            f"all 4 cells are real data, got no_elem={catchment.no_elem}"
        )

    def test_counts_domain_cells_excluding_no_data(self, catchment, write_raster):
        """Test that ``no_elem`` counts only cells inside the domain.

        Test scenario:
            A 2x3 raster with two sentinel cells leaves four real cells.
        """
        path = write_raster(
            np.array([[10, 20, 30], [40, NO_DATA, NO_DATA]], dtype="int32"),
            name="fpl_count.tif",
        )

        catchment.read_flow_path_length(path)

        assert catchment.no_elem == 4, (
            f"expected 4 domain cells out of 6, got {catchment.no_elem}"
        )

    def test_unreadable_file_raises(self, catchment, tmp_path):
        """Test that a file GDAL cannot open is rejected.

        Test scenario:
            Validation is delegated to ``Dataset.read_file``, so a non-raster file
            surfaces as a GDAL ``RuntimeError`` rather than an extension ``ValueError``.
        """
        other = tmp_path / "lengths.asc"
        other.write_text("not a geotiff", encoding="utf-8")

        with pytest.raises(RuntimeError):
            catchment.read_flow_path_length(str(other))

    def test_missing_file_raises(self, catchment, tmp_path):
        """Test that a missing flow-path-length raster is rejected by pyramids.

        Test scenario:
            ``FileNotFoundError`` from ``Dataset.read_file``, matching the other readers.
        """
        with pytest.raises(FileNotFoundError, match="does not exist"):
            catchment.read_flow_path_length(str(tmp_path / "absent.tif"))
