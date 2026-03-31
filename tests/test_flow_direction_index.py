"""Tests for DEM.flow_direction_index.

Covers encoding selection, direction offset correctness for all three
D8 encodings (ESRI, SAGA, GRASS), no-data handling, input validation,
output shape, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from hapi.dem import (
    DEM,
    D8_ENCODINGS,
    D8_OFFSETS_ESRI,
    D8_OFFSETS_GRASS,
    D8_OFFSETS_SAGA,
)

NO_DATA = -1.0


def _create_fd_raster(
    data: np.ndarray, no_data: float = NO_DATA
) -> gdal.Dataset:
    """Create an in-memory GDAL raster from a 2-D flow-direction array.

    Args:
        data: 2-D numpy array of flow direction codes.
        no_data: Sentinel value for no-data cells.

    Returns:
        An in-memory GDAL Dataset (MEM driver).
    """
    rows, cols = data.shape
    driver = gdal.GetDriverByName("MEM")
    ds = driver.Create("", cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform((0.0, 1.0, 0.0, float(rows), 0.0, -1.0))
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(no_data)
    band.WriteArray(data)
    band.FlushCache()
    return ds


def _single_code_raster(
    code: float, no_data: float = NO_DATA
) -> gdal.Dataset:
    """Create a 3x3 raster with one valid code at the center cell.

    Args:
        code: Flow direction code to place at cell (1, 1).
        no_data: Sentinel value for surrounding cells.

    Returns:
        A 3x3 in-memory GDAL Dataset.
    """
    data = np.full((3, 3), no_data)
    data[1, 1] = code
    return _create_fd_raster(data, no_data=no_data)


class TestFlowDirectionIndexEncoding:
    """Tests for the encoding parameter handling."""

    def test_default_encoding_is_esri(self):
        """Test that calling without encoding uses ESRI codes.

        Test scenario:
            A raster with ESRI code 1 (east) at center should
            produce a valid downstream index with no explicit
            encoding argument.
        """
        ds = _single_code_raster(1.0)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert not np.isnan(result[1, 1, 0]), (
            "Center cell should have a valid downstream index "
            "with default encoding"
        )

    @pytest.mark.parametrize("encoding", ["esri", "saga", "grass"])
    def test_supported_encoding_accepted(self, encoding):
        """Test that all supported encoding names are accepted.

        Args:
            encoding: One of the supported D8 encoding names.

        Test scenario:
            Place the first valid code for the encoding at center
            and verify no exception is raised.
        """
        first_code = min(D8_ENCODINGS[encoding])
        ds = _single_code_raster(float(first_code))
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding=encoding)
        assert result.shape == (3, 3, 2), (
            f"Output shape should be (3, 3, 2), got {result.shape}"
        )

    @pytest.mark.parametrize(
        "encoding_input, canonical",
        [("ESRI", "esri"), ("Saga", "saga"), ("GRASS", "grass")],
    )
    def test_encoding_case_insensitive(self, encoding_input, canonical):
        """Test that encoding matching is case-insensitive.

        Args:
            encoding_input: Mixed-case encoding name.
            canonical: Canonical lowercase name.

        Test scenario:
            Mixed-case variants like 'ESRI', 'Saga' should resolve
            to the canonical encoding without error.
        """
        first_code = min(D8_ENCODINGS[canonical])
        ds = _single_code_raster(float(first_code))
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding=encoding_input)
        assert result.shape == (3, 3, 2), (
            f"Encoding '{encoding_input}' should be accepted"
        )

    @pytest.mark.parametrize(
        "bad_encoding", ["arcgis", "qgis", "d8", "", "taudem"]
    )
    def test_invalid_encoding_raises_valueerror(self, bad_encoding):
        """Test that unsupported encoding names raise ValueError.

        Args:
            bad_encoding: An encoding name not in the supported set.

        Test scenario:
            Any name not in {'esri', 'saga', 'grass'} should raise
            ValueError mentioning 'Unsupported encoding'.
        """
        ds = _single_code_raster(1.0)
        dem = DEM(ds)
        with pytest.raises(
            ValueError, match="Unsupported encoding"
        ):
            dem.flow_direction_index(encoding=bad_encoding)


class TestFlowDirectionIndexValidation:
    """Tests for flow direction code validation."""

    @pytest.mark.parametrize(
        "invalid_code, encoding",
        [
            (3, "esri"),
            (5, "esri"),
            (7, "esri"),
            (9, "esri"),
            (100, "esri"),
            (256, "esri"),
            (9, "saga"),
            (10, "saga"),
            (0, "grass"),
            (9, "grass"),
            (10, "grass"),
        ],
        ids=[
            "esri-3", "esri-5", "esri-7", "esri-9",
            "esri-100", "esri-256",
            "saga-9", "saga-10",
            "grass-0", "grass-9", "grass-10",
        ],
    )
    def test_invalid_code_raises_valueerror(
        self, invalid_code, encoding
    ):
        """Test that rasters with invalid codes raise ValueError.

        Args:
            invalid_code: A flow direction value not in the encoding.
            encoding: The D8 encoding being tested.

        Test scenario:
            A raster containing a code outside the valid set for the
            given encoding should raise ValueError.
        """
        ds = _single_code_raster(float(invalid_code))
        dem = DEM(ds)
        with pytest.raises(
            ValueError, match="Flow direction raster"
        ):
            dem.flow_direction_index(encoding=encoding)

    def test_mixed_valid_and_invalid_codes_raises(self):
        """Test that mixing valid and invalid codes raises ValueError.

        Test scenario:
            A raster with ESRI codes 1 (valid) and 3 (invalid)
            should raise ValueError.
        """
        data = np.full((3, 3), NO_DATA)
        data[0, 0] = 1.0
        data[1, 1] = 3.0
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        with pytest.raises(
            ValueError, match="Flow direction raster"
        ):
            dem.flow_direction_index(encoding="esri")


class TestFlowDirectionIndexOutputShape:
    """Tests for output array shape and dtype."""

    @pytest.mark.parametrize(
        "rows, cols", [(1, 1), (3, 3), (2, 5), (5, 2), (13, 14)]
    )
    def test_output_shape(self, rows, cols):
        """Test output shape matches (rows, cols, 2).

        Args:
            rows: Number of raster rows.
            cols: Number of raster columns.

        Test scenario:
            For various raster dimensions, the output should always
            be (rows, cols, 2).
        """
        data = np.full((rows, cols), 1.0)
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="esri")
        assert result.shape == (rows, cols, 2), (
            f"Expected shape ({rows}, {cols}, 2), "
            f"got {result.shape}"
        )

    def test_output_dtype_is_float(self):
        """Test output array dtype supports NaN.

        Test scenario:
            The output must be a floating-point array so that
            no-data cells can be represented as NaN.
        """
        ds = _single_code_raster(1.0)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert np.issubdtype(result.dtype, np.floating), (
            f"Expected floating dtype, got {result.dtype}"
        )


class TestFlowDirectionIndexNoData:
    """Tests for no-data cell handling."""

    def test_nodata_cells_produce_nan(self):
        """Test that no-data cells result in NaN output.

        Test scenario:
            In a 3x3 grid with only the center cell valid, all
            surrounding no-data cells should be NaN in both layers.
        """
        ds = _single_code_raster(1.0)
        dem = DEM(ds)
        result = dem.flow_direction_index()

        for r in range(3):
            for c in range(3):
                if (r, c) != (1, 1):
                    assert np.isnan(result[r, c, 0]), (
                        f"Cell ({r},{c}) is no-data but row index "
                        f"is {result[r, c, 0]}"
                    )
                    assert np.isnan(result[r, c, 1]), (
                        f"Cell ({r},{c}) is no-data but col index "
                        f"is {result[r, c, 1]}"
                    )

    def test_all_nodata_produces_all_nan(self):
        """Test an all-no-data raster produces entirely NaN output.

        Test scenario:
            A 3x3 raster where every cell equals the no-data value
            should yield an output filled with NaN.
        """
        data = np.full((3, 3), NO_DATA)
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert np.all(np.isnan(result)), (
            "All cells are no-data; output should be entirely NaN"
        )

    def test_valid_cell_unaffected_by_surrounding_nodata(self):
        """Test valid cell indices are correct despite no-data neighbors.

        Test scenario:
            ESRI code 4 (south) at center (1,1) of a 3x3 grid.
            Expected downstream cell: row=2, col=1.
        """
        ds = _single_code_raster(4.0)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert result[1, 1, 0] == 2.0, (
            f"Row index should be 2, got {result[1, 1, 0]}"
        )
        assert result[1, 1, 1] == 1.0, (
            f"Col index should be 1, got {result[1, 1, 1]}"
        )

    def test_1x1_nodata_raster(self):
        """Test a 1x1 raster with only no-data.

        Test scenario:
            A single-cell raster at no-data value should produce
            NaN in both output layers.
        """
        data = np.array([[NO_DATA]])
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert np.all(np.isnan(result)), (
            "Single no-data cell should produce all NaN"
        )


class TestFlowDirectionIndexEsriDirections:
    """Tests for ESRI encoding direction offsets (powers of 2)."""

    @pytest.mark.parametrize(
        "code, expected_row, expected_col",
        [
            (1, 1, 2),
            (2, 2, 2),
            (4, 2, 1),
            (8, 2, 0),
            (16, 1, 0),
            (32, 0, 0),
            (64, 0, 1),
            (128, 0, 2),
        ],
        ids=[
            "east", "south-east", "south", "south-west",
            "west", "north-west", "north", "north-east",
        ],
    )
    def test_direction_offset(
        self, code, expected_row, expected_col
    ):
        """Test each ESRI D8 code maps to the correct neighbor.

        Args:
            code: ESRI flow direction code.
            expected_row: Expected row index of downstream cell.
            expected_col: Expected column index of downstream cell.

        Test scenario:
            Place a single code at (1,1) in a 3x3 grid and verify
            the downstream cell indices match the D8 offset.
        """
        ds = _single_code_raster(float(code))
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="esri")
        assert result[1, 1, 0] == expected_row, (
            f"ESRI code {code}: row should be {expected_row}, "
            f"got {result[1, 1, 0]}"
        )
        assert result[1, 1, 1] == expected_col, (
            f"ESRI code {code}: col should be {expected_col}, "
            f"got {result[1, 1, 1]}"
        )

    def test_all_eight_directions_in_one_raster(self):
        """Test a raster containing all eight ESRI codes.

        Test scenario:
            A 3x4 raster with codes 1,2,4,8,16,32,64,128 placed
            in distinct cells. Each cell's downstream index is
            verified against the offset table.
        """
        data = np.full((3, 4), NO_DATA)
        codes_and_positions = [
            (1, 0, 0), (2, 0, 1), (4, 0, 2), (8, 0, 3),
            (16, 1, 0), (32, 1, 1), (64, 1, 2), (128, 1, 3),
        ]
        for code, r, c in codes_and_positions:
            data[r, c] = float(code)

        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="esri")

        for code, r, c in codes_and_positions:
            dr, dc = D8_OFFSETS_ESRI[code]
            assert result[r, c, 0] == r + dr, (
                f"Code {code} at ({r},{c}): expected row "
                f"{r + dr}, got {result[r, c, 0]}"
            )
            assert result[r, c, 1] == c + dc, (
                f"Code {code} at ({r},{c}): expected col "
                f"{c + dc}, got {result[r, c, 1]}"
            )


class TestFlowDirectionIndexSagaDirections:
    """Tests for SAGA encoding direction offsets (0-7)."""

    @pytest.mark.parametrize(
        "code, expected_row, expected_col",
        [
            (0, 1, 2),
            (1, 0, 2),
            (2, 0, 1),
            (3, 0, 0),
            (4, 1, 0),
            (5, 2, 0),
            (6, 2, 1),
            (7, 2, 2),
        ],
        ids=[
            "east", "north-east", "north", "north-west",
            "west", "south-west", "south", "south-east",
        ],
    )
    def test_direction_offset(
        self, code, expected_row, expected_col
    ):
        """Test each SAGA D8 code maps to the correct neighbor.

        Args:
            code: SAGA flow direction code (0-7).
            expected_row: Expected row index of downstream cell.
            expected_col: Expected column index of downstream cell.

        Test scenario:
            Place a single code at (1,1) in a 3x3 grid and verify
            the downstream cell matches the SAGA offset table.
        """
        ds = _single_code_raster(float(code))
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="saga")
        assert result[1, 1, 0] == expected_row, (
            f"SAGA code {code}: row should be {expected_row}, "
            f"got {result[1, 1, 0]}"
        )
        assert result[1, 1, 1] == expected_col, (
            f"SAGA code {code}: col should be {expected_col}, "
            f"got {result[1, 1, 1]}"
        )

    def test_all_eight_saga_directions(self):
        """Test a raster containing all eight SAGA codes.

        Test scenario:
            A 3x4 raster with codes 0-7 placed in distinct cells.
            Each cell's downstream index is verified against the
            SAGA offset table.
        """
        data = np.full((3, 4), NO_DATA)
        codes_and_positions = [
            (0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3),
            (4, 1, 0), (5, 1, 1), (6, 1, 2), (7, 1, 3),
        ]
        for code, r, c in codes_and_positions:
            data[r, c] = float(code)

        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="saga")

        for code, r, c in codes_and_positions:
            dr, dc = D8_OFFSETS_SAGA[code]
            assert result[r, c, 0] == r + dr, (
                f"SAGA code {code} at ({r},{c}): expected row "
                f"{r + dr}, got {result[r, c, 0]}"
            )
            assert result[r, c, 1] == c + dc, (
                f"SAGA code {code} at ({r},{c}): expected col "
                f"{c + dc}, got {result[r, c, 1]}"
            )


class TestFlowDirectionIndexGrassDirections:
    """Tests for GRASS encoding direction offsets (1-8)."""

    @pytest.mark.parametrize(
        "code, expected_row, expected_col",
        [
            (1, 0, 1),
            (2, 0, 2),
            (3, 1, 2),
            (4, 2, 2),
            (5, 2, 1),
            (6, 2, 0),
            (7, 1, 0),
            (8, 0, 0),
        ],
        ids=[
            "north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west",
        ],
    )
    def test_direction_offset(
        self, code, expected_row, expected_col
    ):
        """Test each GRASS D8 code maps to the correct neighbor.

        Args:
            code: GRASS flow direction code (1-8).
            expected_row: Expected row index of downstream cell.
            expected_col: Expected column index of downstream cell.

        Test scenario:
            Place a single code at (1,1) in a 3x3 grid and verify
            the downstream cell matches the GRASS offset table.
        """
        ds = _single_code_raster(float(code))
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="grass")
        assert result[1, 1, 0] == expected_row, (
            f"GRASS code {code}: row should be {expected_row}, "
            f"got {result[1, 1, 0]}"
        )
        assert result[1, 1, 1] == expected_col, (
            f"GRASS code {code}: col should be {expected_col}, "
            f"got {result[1, 1, 1]}"
        )

    def test_all_eight_grass_directions(self):
        """Test a raster containing all eight GRASS codes.

        Test scenario:
            A 3x4 raster with codes 1-8 placed in distinct cells.
            Each cell's downstream index is verified against the
            GRASS offset table.
        """
        data = np.full((3, 4), NO_DATA)
        codes_and_positions = [
            (1, 0, 0), (2, 0, 1), (3, 0, 2), (4, 0, 3),
            (5, 1, 0), (6, 1, 1), (7, 1, 2), (8, 1, 3),
        ]
        for code, r, c in codes_and_positions:
            data[r, c] = float(code)

        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="grass")

        for code, r, c in codes_and_positions:
            dr, dc = D8_OFFSETS_GRASS[code]
            assert result[r, c, 0] == r + dr, (
                f"GRASS code {code} at ({r},{c}): expected row "
                f"{r + dr}, got {result[r, c, 0]}"
            )
            assert result[r, c, 1] == c + dc, (
                f"GRASS code {code} at ({r},{c}): expected col "
                f"{c + dc}, got {result[r, c, 1]}"
            )


class TestFlowDirectionIndexCrossEncoding:
    """Tests verifying consistent physical directions across encodings."""

    @pytest.mark.parametrize(
        "direction, esri_code, saga_code, grass_code",
        [
            ("east", 1, 0, 3),
            ("south-east", 2, 7, 4),
            ("south", 4, 6, 5),
            ("south-west", 8, 5, 6),
            ("west", 16, 4, 7),
            ("north-west", 32, 3, 8),
            ("north", 64, 2, 1),
            ("north-east", 128, 1, 2),
        ],
        ids=[
            "east", "south-east", "south", "south-west",
            "west", "north-west", "north", "north-east",
        ],
    )
    def test_same_direction_same_result(
        self, direction, esri_code, saga_code, grass_code
    ):
        """Test that all encodings produce identical downstream cells.

        Args:
            direction: Cardinal/ordinal direction name.
            esri_code: ESRI code for this direction.
            saga_code: SAGA code for this direction.
            grass_code: GRASS code for this direction.

        Test scenario:
            Place each encoding's code at (1,1) in a 3x3 grid and
            verify all three produce the same downstream cell.
        """
        results = {}
        for enc, code in [
            ("esri", esri_code),
            ("saga", saga_code),
            ("grass", grass_code),
        ]:
            ds = _single_code_raster(float(code))
            dem = DEM(ds)
            r = dem.flow_direction_index(encoding=enc)
            results[enc] = (r[1, 1, 0], r[1, 1, 1])

        assert results["esri"] == results["saga"], (
            f"{direction}: ESRI {results['esri']} != "
            f"SAGA {results['saga']}"
        )
        assert results["esri"] == results["grass"], (
            f"{direction}: ESRI {results['esri']} != "
            f"GRASS {results['grass']}"
        )


class TestFlowDirectionIndexEdgeCases:
    """Tests for boundary and edge-case scenarios."""

    def test_1x1_raster_valid_code(self):
        """Test a 1x1 raster with a valid direction code.

        Test scenario:
            A single-cell raster with ESRI code 1 (east). The
            downstream cell (0, 1) is out of bounds but should
            still be computed without error.
        """
        data = np.array([[1.0]])
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert result[0, 0, 0] == 0.0, (
            f"Row should be 0, got {result[0, 0, 0]}"
        )
        assert result[0, 0, 1] == 1.0, (
            f"Col should be 1 (out of bounds), got {result[0, 0, 1]}"
        )

    def test_corner_cell_points_outside_grid(self):
        """Test corner cells pointing outward yield negative indices.

        Test scenario:
            ESRI code 32 (north-west) at (0,0) should produce
            downstream cell (-1, -1) — outside the raster.
        """
        data = np.full((3, 3), NO_DATA)
        data[0, 0] = 32.0
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert result[0, 0, 0] == -1.0, (
            f"Row should be -1, got {result[0, 0, 0]}"
        )
        assert result[0, 0, 1] == -1.0, (
            f"Col should be -1, got {result[0, 0, 1]}"
        )

    def test_edge_cell_points_outside_grid(self):
        """Test edge cells pointing outward yield out-of-bounds row.

        Test scenario:
            ESRI code 64 (north) at (0,1) should produce downstream
            cell (-1, 1) — row is outside the raster.
        """
        data = np.full((3, 3), NO_DATA)
        data[0, 1] = 64.0
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert result[0, 1, 0] == -1.0, (
            f"Row should be -1, got {result[0, 1, 0]}"
        )
        assert result[0, 1, 1] == 1.0, (
            f"Col should be 1, got {result[0, 1, 1]}"
        )

    def test_uniform_raster_all_cells_flow_east(self):
        """Test a uniform raster where every cell flows east.

        Test scenario:
            A 3x3 raster filled with ESRI code 1 (east). Every cell
            at (r, c) should point to (r, c+1).
        """
        data = np.full((3, 3), 1.0)
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        for r in range(3):
            for c in range(3):
                assert result[r, c, 0] == r, (
                    f"Cell ({r},{c}): row should be {r}, "
                    f"got {result[r, c, 0]}"
                )
                assert result[r, c, 1] == c + 1, (
                    f"Cell ({r},{c}): col should be {c + 1}, "
                    f"got {result[r, c, 1]}"
                )

    def test_rectangular_raster_wide(self):
        """Test a non-square wide raster (2 rows x 5 cols).

        Test scenario:
            All cells have ESRI code 4 (south), so each should
            point one row down, same column.
        """
        data = np.full((2, 5), 4.0)
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert result.shape == (2, 5, 2), (
            f"Shape should be (2, 5, 2), got {result.shape}"
        )
        for c in range(5):
            assert result[0, c, 0] == 1.0, (
                f"Row 0, col {c}: should flow to row 1"
            )
            assert result[0, c, 1] == float(c), (
                f"Row 0, col {c}: should stay in col {c}"
            )

    def test_rectangular_raster_tall(self):
        """Test a non-square tall raster (5 rows x 2 cols).

        Test scenario:
            All cells have ESRI code 1 (east), so each should
            point one column right, same row.
        """
        data = np.full((5, 2), 1.0)
        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()
        assert result.shape == (5, 2, 2), (
            f"Shape should be (5, 2, 2), got {result.shape}"
        )
        for r in range(5):
            assert result[r, 0, 0] == float(r), (
                f"Row {r}, col 0: should stay in row {r}"
            )
            assert result[r, 0, 1] == 1.0, (
                f"Row {r}, col 0: should flow to col 1"
            )

    def test_checkerboard_nodata_pattern(self):
        """Test alternating valid/no-data cells in a checkerboard.

        Test scenario:
            A 4x4 grid where only cells at even (r+c) positions
            hold ESRI code 4 (south). Odd-position cells are
            no-data. Valid cells should compute correctly; no-data
            cells should be NaN.
        """
        data = np.full((4, 4), NO_DATA)
        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 0:
                    data[r, c] = 4.0

        ds = _create_fd_raster(data)
        dem = DEM(ds)
        result = dem.flow_direction_index()

        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 0:
                    assert result[r, c, 0] == r + 1, (
                        f"Valid cell ({r},{c}): row should be "
                        f"{r + 1}, got {result[r, c, 0]}"
                    )
                    assert result[r, c, 1] == float(c), (
                        f"Valid cell ({r},{c}): col should be "
                        f"{c}, got {result[r, c, 1]}"
                    )
                else:
                    assert np.isnan(result[r, c, 0]), (
                        f"No-data cell ({r},{c}) should be NaN"
                    )

    def test_saga_code_zero_not_confused_with_nodata(self):
        """Test SAGA code 0 (east) is not confused with no-data.

        Test scenario:
            SAGA encoding uses 0 as a valid code (east). With
            no-data set to -1, code 0 should be treated as a valid
            direction, not as no-data.
        """
        ds = _single_code_raster(0.0)
        dem = DEM(ds)
        result = dem.flow_direction_index(encoding="saga")
        assert not np.isnan(result[1, 1, 0]), (
            "SAGA code 0 should be valid, not treated as no-data"
        )
        assert result[1, 1, 0] == 1.0, (
            f"SAGA code 0 (east): row should be 1, "
            f"got {result[1, 1, 0]}"
        )
        assert result[1, 1, 1] == 2.0, (
            f"SAGA code 0 (east): col should be 2, "
            f"got {result[1, 1, 1]}"
        )
