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
            ``no_data_value`` and file ``name``, returning the path of the written
            GeoTIFF.
    """

    def _write(
        array: np.ndarray,
        *,
        no_data_value: int | float | None = NO_DATA,
        name: str = "raster.tif",
    ) -> str:
        path = tmp_path / name
        Dataset.create_from_array(
            array,
            top_left_corner=(0.0, array.shape[0] * CELL_SIZE),
            cell_size=CELL_SIZE,
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

        assert np.isnan(catchment.FlowAccArr[1, 1]), (
            f"the no-data cell should be NaN, got {catchment.FlowAccArr[1, 1]}"
        )
        assert not np.isnan(catchment.FlowAccArr[:1, :]).any(), (
            f"real accumulation values must not be masked, got {catchment.FlowAccArr}"
        )
        assert catchment.FlowAccArr[0, 1] == 1.0, (
            f"expected the value 1 to survive unchanged, got {catchment.FlowAccArr[0, 1]}"
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

        assert not np.isnan(catchment.FlowAccArr).any(), (
            f"no cell holds the sentinel, so nothing should be masked, got {catchment.FlowAccArr}"
        )
        assert catchment.FlowAccArr[1, 1] == float(NEAR_SENTINEL), (
            f"expected {NEAR_SENTINEL} to survive masking, got {catchment.FlowAccArr[1, 1]}"
        )
        assert catchment.no_elem == 4, (
            f"all 4 cells are real data, got no_elem={catchment.no_elem}"
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

    def test_returns_float_array_for_integer_raster(self, catchment, write_raster):
        """Test that an integer raster is promoted to float so NaN is representable.

        Test scenario:
            An ``int32`` source must come back as a floating array — an integer array
            cannot hold NaN, so the promotion is what makes masking expressible at all.
        """
        path = write_raster(np.array([[0, 1], [2, NO_DATA]], dtype="int32"))

        catchment.read_flow_acc(path)

        assert catchment.FlowAccArr.dtype.kind == "f", (
            f"expected a floating dtype so NaN can be stored, got {catchment.FlowAccArr.dtype}"
        )

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

        assert catchment.CellSize == pytest.approx(CELL_SIZE), (
            f"expected a {CELL_SIZE} m cell, got {catchment.CellSize}"
        )
        assert catchment.px_area == pytest.approx(16.0), (
            f"expected a 16 km2 pixel for a {CELL_SIZE} m cell, got {catchment.px_area}"
        )
        assert catchment.px_tot_area == pytest.approx(48.0), (
            f"expected 3 cells x 16 km2 = 48 km2, got {catchment.px_tot_area}"
        )

    def test_missing_file_raises(self, catchment, tmp_path):
        """Test that a path which does not exist is rejected.

        Test scenario:
            A ``.tif`` path inside an empty temp directory raises ``AssertionError``.
        """
        with pytest.raises(AssertionError, match="does not exist"):
            catchment.read_flow_acc(str(tmp_path / "absent.tif"))

    def test_non_tif_extension_raises(self, catchment, tmp_path):
        """Test that a non-GeoTIFF extension is rejected before the raster is opened.

        Test scenario:
            An existing file without a ``.tif`` extension raises ``AssertionError``.
        """
        other = tmp_path / "grid.asc"
        other.write_text("not a geotiff", encoding="utf-8")

        with pytest.raises(AssertionError, match="extension"):
            catchment.read_flow_acc(str(other))

    def test_non_string_path_raises_type_error(self, catchment, tmp_path):
        """Test that a non-string path is rejected with ``TypeError``.

        Test scenario:
            A ``pathlib.Path`` is not accepted by the current contract.
        """
        with pytest.raises(TypeError, match="string"):
            catchment.read_flow_acc(tmp_path / "acc.tif")

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

    def test_non_tif_extension_raises(self, catchment, tmp_path):
        """Test that a non-GeoTIFF extension is rejected before the raster is opened.

        Test scenario:
            An existing file without a ``.tif`` extension raises ``ValueError``.
        """
        other = tmp_path / "directions.asc"
        other.write_text("not a geotiff", encoding="utf-8")

        with pytest.raises(ValueError, match="extension"):
            catchment.read_flow_dir(str(other))

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

        assert np.isnan(catchment.fpl_arr[1, 1]), (
            f"the no-data cell should be NaN, got {catchment.fpl_arr[1, 1]}"
        )
        assert catchment.fpl_arr[0, 0] == 10.0, (
            f"the path length 10 should survive unchanged, got {catchment.fpl_arr[0, 0]}"
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

        assert not np.isnan(catchment.fpl_arr).any(), (
            f"no cell holds the sentinel, so nothing should be masked, got {catchment.fpl_arr}"
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

    def test_non_tif_extension_raises(self, catchment, tmp_path):
        """Test that a non-GeoTIFF extension is rejected.

        Test scenario:
            An existing file without a ``.tif`` extension raises ``ValueError``.
        """
        other = tmp_path / "lengths.asc"
        other.write_text("not a geotiff", encoding="utf-8")

        with pytest.raises(ValueError, match="extension"):
            catchment.read_flow_path_length(str(other))
