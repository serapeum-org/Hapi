"""Tests for ``FlowNetwork``, which owns the routing network and the grid it defines.

``test_read_raster_inputs`` already covers reading the two rasters -- masking, the
near-sentinel case, the D8 code check, the direction table. This module covers the class
itself: the construction-time check, the values derived from the accumulation array, and the
integer-code helper the loaders share.
"""

from __future__ import annotations

import numpy as np
import pytest

from hapi.inputs import D8_CODES, FlowNetwork, _to_int_codes

NO_DATA = -9999.0


def _network(
    acc: np.ndarray, direction: np.ndarray | None = None, px_area: float = 16.0
) -> FlowNetwork:
    """Build a FlowNetwork straight from arrays, bypassing the rasters.

    Args:
        acc: Flow-accumulation array; `NaN` marks cells outside the domain.
        direction: Optional flow-direction array on the same grid.
        px_area: Pixel area in km2.

    Returns:
        FlowNetwork: The constructed network.
    """
    return FlowNetwork(
        flow_acc_arr=acc,
        no_data_value=NO_DATA,
        cell_size=4000.0,
        px_area=px_area,
        flow_dir_arr=direction,
        FDT={"0,0": []} if direction is not None else None,
    )


@pytest.fixture
def square_network() -> FlowNetwork:
    """A 2x2 network with one cell outside the domain.

    Returns:
        FlowNetwork: Accumulation `[[0, 1], [2, NaN]]` with a matching direction array.
    """
    acc = np.array([[0.0, 1.0], [2.0, np.nan]])
    direction = np.array([[1.0, 2.0], [4.0, np.nan]])
    return _network(acc, direction)


class TestConstruction:
    """Tests for ``FlowNetwork.__post_init__``."""

    def test_matching_grids_are_accepted(self):
        """Test that two rasters on the same grid construct cleanly.

        Test scenario:
            The normal case: accumulation and direction describe the same catchment, so a
            cell index means the same place in both.
        """
        acc = np.zeros((3, 4))

        network = _network(acc, np.ones((3, 4)))

        assert network.shape == (3, 4), f"expected a 3x4 grid, got {network.shape}"

    def test_accumulation_alone_is_accepted(self):
        """Test that the direction raster is optional.

        Test scenario:
            The triangular (MAXBAS) path sends every cell straight to the outlet and never
            reads a direction raster, so a network must be constructible without one.
        """
        network = _network(np.zeros((2, 2)))

        assert network.flow_dir_arr is None, "no direction array was given"
        assert network.FDT is None, "no direction table should be derived"

    def test_mismatched_grids_are_rejected(self):
        """Test that rasters on different grids are refused, naming both shapes.

        Test scenario:
            A cell index would mean a different place in each array. Left unchecked, the run
            loop either reads the wrong cell or walks off the end of the shorter one, far from
            the cause.
        """
        with pytest.raises(ValueError, match=r"\(2, 3\)") as exc:
            _network(np.zeros((2, 3)), np.zeros((4, 2)))

        assert "(4, 2)" in str(exc.value), (
            f"the error should name the direction grid too, got: {exc.value}"
        )


class TestDerivedValues:
    """Tests for the values ``FlowNetwork`` derives from the accumulation array."""

    def test_shape_rows_and_cols_follow_the_array(self, square_network: FlowNetwork):
        """Test that the grid is read off the accumulation array.

        Args:
            square_network: A 2x2 network.

        Test scenario:
            Deriving rather than storing is what stops the grid and the array disagreeing;
            they cannot drift apart because there is only one of them.
        """
        assert square_network.shape == (2, 2), f"got {square_network.shape}"
        assert square_network.rows == 2, f"got {square_network.rows}"
        assert square_network.cols == 2, f"got {square_network.cols}"

    def test_no_elem_counts_only_domain_cells(self, square_network: FlowNetwork):
        """Test that masked cells are excluded from the domain count.

        Args:
            square_network: A 2x2 network with one NaN cell.

        Test scenario:
            `no_elem` sizes the parameter vectors a calibration produces, so counting a
            masked cell would silently widen every saved vector.
        """
        assert square_network.no_elem == 3, (
            f"3 of 4 cells are inside the domain, got {square_network.no_elem}"
        )

    def test_acc_val_is_sorted_distinct_integers(self):
        """Test that the accumulation values are de-duplicated, truncated and sorted.

        Test scenario:
            Values are truncated to integers *before* de-duplication, so 1.2 and 1.8 collapse
            to a single 1 rather than surviving as two entries that both render as "1".
        """
        acc = np.array([[1.2, 1.8], [3.0, np.nan]])

        # Built once: constructing it again inside the failure message would re-run the code
        # under test to report on it.
        values = _network(acc).acc_val

        assert values == [1, 3], f"expected [1, 3], got {values}"

    def test_outlet_is_the_most_accumulated_cell(self, square_network: FlowNetwork):
        """Test that the outlet is where accumulation peaks.

        Args:
            square_network: A 2x2 network whose maximum is at [1, 0].

        Test scenario:
            The MAXBAS path routes every cell straight here, so the index must point at the
            largest accumulation value and ignore the masked cells.
        """
        outlet = square_network.outlet

        assert (int(outlet[0][0]), int(outlet[1][0])) == (1, 0), (
            f"expected the outlet at [1, 0], got {outlet}"
        )

    def test_px_tot_area_scales_with_the_domain(self):
        """Test that the total area counts only domain cells.

        Test scenario:
            `px_tot_area` is `no_elem * px_area`, so masked cells must not contribute area.
        """
        acc = np.array([[0.0, 1.0], [2.0, np.nan]])

        assert _network(acc, px_area=16.0).px_tot_area == pytest.approx(48.0), (
            "3 domain cells at 16 km2 each is 48 km2"
        )

    @pytest.mark.parametrize(
        "direction, expected",
        [(np.ones((2, 2)), True), (None, False)],
        ids=["with-direction", "accumulation-only"],
    )
    def test_has_flow_direction_reports_what_was_loaded(self, direction, expected):
        """Test the flag the run layer uses to tell the two routing paths apart.

        Args:
            direction: The direction array, or None.
            expected: Whether the network should report a direction raster.

        Test scenario:
            Muskingum routes cell to cell and needs a direction raster; MAXBAS does not. The
            flag lets a caller check before reaching into `flow_dir_arr`.
        """
        network = _network(np.zeros((2, 2)), direction)

        assert network.has_flow_direction is expected, (
            f"expected has_flow_direction={expected}, got {network.has_flow_direction}"
        )

    @pytest.mark.parametrize(
        "rows, cols, expected",
        [(2, 2, True), (2, 3, False), (3, 2, False), (1, 1, False)],
        ids=["match", "wrong-cols", "wrong-rows", "neither"],
    )
    def test_matches_compares_the_whole_grid(
        self, square_network: FlowNetwork, rows: int, cols: int, expected: bool
    ):
        """Test that ``matches`` compares both axes.

        Args:
            square_network: A 2x2 network.
            rows: Row count to compare.
            cols: Column count to compare.
            expected: Whether the grids should be reported as equal.

        Test scenario:
            Used to check another input covers the catchment; a comparison that looked at
            only one axis would pass a transposed grid.
        """
        assert square_network.matches(rows, cols) is expected, (
            f"matches({rows}, {cols}) should be {expected}"
        )


class TestToIntCodes:
    """Tests for ``_to_int_codes``, shared by the accumulation and direction reads."""

    def test_truncates_before_deduplicating(self):
        """Test that near-integer values collapse to one code.

        Test scenario:
            De-duplicating first would leave 1.2 and 1.8 as two values that both truncate to
            1, yielding a repeated code. Truncating first gives a single 1.
        """
        codes = _to_int_codes(np.array([[1.2, 1.8], [2.0, np.nan]]))

        assert sorted(np.unique(codes).tolist()) == [1, 2], (
            f"expected codes [1, 2], got {sorted(np.unique(codes).tolist())}"
        )

    def test_masked_cells_are_dropped(self):
        """Test that NaN cells never reach the integer conversion.

        Test scenario:
            `int(nan)` is undefined, so masked cells must be filtered out first.
        """
        codes = _to_int_codes(np.array([[np.nan, 5.0], [np.nan, np.nan]]))

        assert codes.tolist() == [5], (
            f"only the one real cell should survive, got {codes}"
        )

    def test_infinite_values_are_rejected(self):
        """Test that an infinite cell raises rather than saturating.

        Test scenario:
            `astype(np.int64)` would turn `inf` into `INT64_MIN` with only a RuntimeWarning,
            which would then read as a real accumulation code.
        """
        with pytest.raises(ValueError, match="infinite values"):
            _to_int_codes(np.array([[1.0, np.inf], [2.0, 3.0]]))

    def test_values_beyond_int64_are_rejected(self):
        """Test that a value too large for int64 raises rather than saturating.

        Test scenario:
            Same silent-saturation trap as the infinite case, reached from a raster whose
            values are merely enormous.
        """
        with pytest.raises(ValueError, match="int64 range"):
            _to_int_codes(np.array([[1.0, 1e30], [2.0, 3.0]]))

    def test_empty_domain_returns_an_empty_array(self):
        """Test that an all-masked array yields no codes rather than raising.

        Test scenario:
            The bounds check guards on `finite.size`, so an empty selection must pass through
            instead of tripping `min()` on an empty array.
        """
        codes = _to_int_codes(np.full((2, 2), np.nan))

        assert codes.size == 0, f"an all-masked array has no codes, got {codes}"


class TestD8Codes:
    """Tests for the ``D8_CODES`` constant."""

    def test_holds_the_eight_esri_directions(self):
        """Test that the constant is the eight powers of two ESRI uses.

        Test scenario:
            `from_rasters` rejects any direction raster holding a code outside this set, so
            the set itself is part of the contract.
        """
        assert set(D8_CODES) == {1, 2, 4, 8, 16, 32, 64, 128}, (
            f"expected the eight ESRI D8 codes, got {sorted(D8_CODES)}"
        )


class TestAccValCaching:
    """Tests for the caching of `FlowNetwork.acc_val` and its invalidation."""

    @pytest.fixture
    def network(self) -> FlowNetwork:
        """Build a small network with four distinct accumulation levels.

        Returns:
            FlowNetwork: Grid whose in-domain cells accumulate 0, 1, 2 and 3.
        """
        acc = np.array([[0.0, 1.0], [2.0, 3.0]])
        return FlowNetwork(acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0)

    def test_repeated_reads_return_the_same_object(self, network: FlowNetwork):
        """Test that `acc_val` is computed once rather than on every read.

        Test scenario:
            `route_muskingum` reads this once per (accumulation level, row, column), so a
            property that reruns `np.unique` over the whole grid turns the routing loop into
            `(n_acc - 1) x rows x cols` full-grid scans. On the 13x14 test catchment that is
            invisible; on a 100x100 catchment it dominates the run. Identity across reads is
            what distinguishes a cached value from a recomputed one.
        """
        first = network.acc_val

        assert network.acc_val is first, (
            "acc_val must be cached; recomputing it makes the routing loop quadratic"
        )

    def test_replacing_the_accumulation_array_clears_the_cache(
        self, network: FlowNetwork
    ):
        """Test that the cache does not outlive the array it was derived from.

        Test scenario:
            Caching is only safe if it invalidates. Masking the two highest cells must be
            reflected on the next read, not served from the value computed beforehand.
        """
        stale = network.acc_val
        assert stale == [0, 1, 2, 3], f"unexpected starting levels: {stale}"

        network.flow_acc_arr = np.array([[0.0, 1.0], [np.nan, np.nan]])

        assert network.acc_val == [0, 1], (
            f"the cache must be rebuilt from the new array, got {network.acc_val}"
        )

    def test_the_cached_value_still_matches_the_uncached_computation(
        self, network: FlowNetwork
    ):
        """Test that caching did not change what `acc_val` reports.

        Test scenario:
            The values are truncated before de-duplication, and the domain mask excludes
            NaN. Caching must preserve both, so compare against the computation done inline.
        """
        expected = np.unique(_to_int_codes(network.flow_acc_arr)).tolist()

        assert network.acc_val == expected, (
            f"Expected {expected}, got {network.acc_val}"
        )


class TestCellsByAccVal:
    """The bucketed index that makes the routing pass linear in the domain."""

    def test_it_groups_every_in_domain_cell_exactly_once(self):
        """Test that the buckets partition the domain -- no cell missed, none duplicated.

        Test scenario:
            The routing visits cells through these buckets, so a missing cell is a cell that
            never gets routed and a duplicated one is routed twice. The count is the invariant
            that makes the replacement safe.
        """
        acc = np.array([[0.0, 1.0, 1.0], [2.0, np.nan, 2.0], [3.0, 3.0, np.nan]])
        network = FlowNetwork(
            acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
        )

        grouped = network.cells_by_acc_val
        total = sum(len(cells) for cells in grouped.values())

        assert total == network.no_elem, (
            f"the buckets must hold every domain cell once; {total} against "
            f"{network.no_elem} in the domain"
        )
        flat = [cell for cells in grouped.values() for cell in cells]
        assert len(set(flat)) == len(flat), "no cell may appear in two buckets"

    def test_order_within_a_level_is_row_major(self):
        """Test that cells come back in the order the replaced scan visited them.

        Test scenario:
            The loop this replaces was `for x: for y:`, so within one accumulation level it
            went row by row. Muskingum routing accumulates, so a different order inside a
            level could change the result -- this is what makes the change bit-identical.
        """
        acc = np.array([[5.0, 5.0], [5.0, 5.0]])
        network = FlowNetwork(
            acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
        )

        assert network.cells_by_acc_val[5.0] == [(0, 0), (0, 1), (1, 0), (1, 1)], (
            "cells within a level must be row-major, matching the x-outer/y-inner scan"
        )

    def test_it_visits_far_fewer_cells_than_a_per_level_grid_scan(self):
        """Test that the index is what removes the quadratic term.

        Test scenario:
            Answering "which cells are at level j" by scanning the grid costs
            `n_acc x rows x cols`; the number of levels grows with the domain, so that is
            effectively quadratic. This pins the linear count, so a change back to a scan
            shows up here rather than as an unexplained slowdown on a real catchment.
        """
        side = 12
        acc = np.arange(side * side, dtype=float).reshape(side, side)
        network = FlowNetwork(
            acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
        )

        bucketed = sum(len(cells) for cells in network.cells_by_acc_val.values())
        scanned = len(network.acc_val) * network.rows * network.cols

        assert bucketed == network.no_elem, "one visit per domain cell"
        assert scanned // bucketed >= side * side // 2, (
            f"the scan this replaces cost {scanned} tests against {bucketed} visits; if that "
            "ratio has collapsed the index is no longer doing its job"
        )

    def test_a_fractional_raster_still_reaches_every_cell(self):
        """Test that fractional accumulation values are routed, not silently dropped.

        Test scenario:
            `acc_val` truncates, so a raster holding 1.2 yields the code 1. The grid scan this
            index replaced compared the *raw* value against that code, so `1.2 == 1` was False
            and the cell was never routed -- and since the routing sums `quz_routed` from
            upstream neighbours, its entire contribution vanished from every cell downstream,
            with nothing raised. Keying on the same truncated code is what `acc_val` has always
            documented; only the comparison had drifted away from it.
        """
        acc = np.array([[0.0, 1.2], [2.5, 3.0]])
        network = FlowNetwork(
            acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
        )

        selected = sum(
            len(network.cells_by_acc_val.get(level, ())) for level in network.acc_val
        )

        assert selected == network.no_elem, (
            f"every domain cell must be reachable through an acc_val code; {selected} of "
            f"{network.no_elem} were, so the rest would never be routed"
        )

    def test_values_sharing_a_code_are_grouped(self):
        """Test that 1.2 and 1.8 land in one bucket, as `acc_val` documents.

        Test scenario:
            `acc_val` collapses them to the single code 1, so the routing treats them as one
            level. The bucket has to agree, or the level would select only some of its cells.
        """
        acc = np.array([[1.2, 1.8], [3.0, np.nan]])
        network = FlowNetwork(
            acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
        )

        assert network.acc_val == [1, 3], "the two fractional values are one code"
        assert network.cells_by_acc_val[1] == [(0, 0), (0, 1)], (
            "both cells at that code must be in its bucket"
        )

    def test_replacing_the_accumulation_array_drops_the_index(self):
        """Test that the cache is invalidated with `acc_val`, not left describing the old grid.

        Test scenario:
            `acc_val` is already dropped on replacement. An index that survived would send the
            routing to cell coordinates from a different grid, which is worse than recomputing.
        """
        acc = np.array([[0.0, 1.0], [2.0, np.nan]])
        network = FlowNetwork(
            acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
        )
        assert len(network.cells_by_acc_val) == 3, "primed from the first array"

        network.flow_acc_arr = np.array([[7.0, 7.0], [7.0, 7.0]])

        assert network.cells_by_acc_val == {7.0: [(0, 0), (0, 1), (1, 0), (1, 1)]}, (
            "the index must be rebuilt from the replacement array"
        )
