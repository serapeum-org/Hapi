import numpy as np
import pytest
from pyramids.dataset import Dataset

from hapi.rrm.parameters import Parameters as DP

NO_DATA = -9999
"""No-data sentinel stamped on the synthetic rasters in :class:`TestParametersMasking`."""

NEAR_SENTINEL = -9990
"""Legitimate value within 0.1% of :data:`NO_DATA`.

``math.isclose(-9990, -9999, rel_tol=0.001)`` is ``True``, so the loop that
``Parameters.__init__`` used before delegating to ``read_array(masked=True)`` masked this
value. Exact integer comparison keeps it.
"""


def test_create_distparameters_instance(
    coello_acc_path: str,
    coello_acc_raster: Dataset,
    coello_no_parameters: int,
    coello_rows: int,
    coello_cols: int,
):
    klb = 0.5
    kub = 1
    no_lumped_par = 1
    lumped_par_pos = [7]

    SpatialVarFun = DP(
        coello_acc_raster,
        coello_no_parameters,
        no_lumped_par=no_lumped_par,
        lumped_par_pos=lumped_par_pos,
        function=2,
        k_lower_bound=klb,
        k_upper_bound=kub,
    )
    assert SpatialVarFun.no_lumped_par == no_lumped_par
    assert SpatialVarFun.lumped_par_pos == lumped_par_pos
    assert isinstance(SpatialVarFun.raster, Dataset)
    assert SpatialVarFun.rows == coello_rows
    assert SpatialVarFun.cols == coello_cols
    assert isinstance(SpatialVarFun.raster_array, np.ndarray)
    assert SpatialVarFun.no_parameters == 11
    assert SpatialVarFun.Par3d.shape == (coello_rows, coello_cols, coello_no_parameters)
    assert SpatialVarFun.totnumberpar == 980
    assert SpatialVarFun.Par2d.shape == (11, 89)


def test_par3d(
    coello_acc_path: str,
    coello_acc_raster: Dataset,
    coello_no_parameters: int,
    coello_parameters: np.ndarray,
    coello_parameters_dist: np.ndarray,
):
    klb = 0.5
    kub = 1
    no_lumped_par = 1
    lumped_par_pos = [7]

    SpatialVarFun = DP(
        coello_acc_raster,
        coello_no_parameters,
        no_lumped_par=no_lumped_par,
        lumped_par_pos=lumped_par_pos,
        function=2,
        k_lower_bound=klb,
        k_upper_bound=kub,
    )
    SpatialVarFun.Function(coello_parameters)
    arr = SpatialVarFun.Par3d
    assert np.array_equal(arr, coello_parameters_dist, equal_nan=True)


class TestParametersMasking:
    """Tests for the no-data masking ``Parameters.__init__`` delegates to pyramids."""

    @staticmethod
    def _raster(array: np.ndarray, no_data_value: int = NO_DATA) -> Dataset:
        """Build an in-memory Dataset from ``array``.

        Args:
            array: Pixel values for the single band.
            no_data_value: Sentinel stamped on the band.

        Returns:
            Dataset: An in-memory raster on a 4000 m UTM 18N grid.
        """
        return Dataset.create_from_array(
            array,
            top_left_corner=(0.0, array.shape[0] * 4000.0),
            cell_size=4000.0,
            epsg=32618,
            no_data_value=no_data_value,
        )

    def test_masks_no_data_cells_to_nan(self):
        """Test that sentinel cells become NaN in ``raster_array``.

        Test scenario:
            A 2x2 raster whose bottom-right cell holds the sentinel. Only that cell is NaN
            and the remaining three values survive unchanged.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32")), 12
        )

        assert np.isnan(parameters.raster_array[1, 1]), (
            f"the no-data cell should be NaN, got {parameters.raster_array[1, 1]}"
        )
        assert parameters.raster_array[0, 0] == 1.0, (
            f"the value 1 should survive unchanged, got {parameters.raster_array[0, 0]}"
        )

    def test_preserves_value_near_sentinel(self):
        """Test that a real value within 0.1% of the sentinel is no longer destroyed.

        Test scenario:
            The regression guarding the behaviour change on this branch. ``-9990`` beside a
            ``-9999`` marker survives, so every cell counts toward the domain and toward the
            parameter vector length.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NEAR_SENTINEL]], dtype="int32")), 12
        )

        assert not np.isnan(parameters.raster_array).any(), (
            f"no cell holds the sentinel, so nothing should be masked, got {parameters.raster_array}"
        )
        assert parameters.raster_array[1, 1] == float(NEAR_SENTINEL), (
            f"expected {NEAR_SENTINEL} to survive masking, got {parameters.raster_array[1, 1]}"
        )
        assert parameters.no_elem == 4, (
            f"all 4 cells are real data, got no_elem={parameters.no_elem}"
        )

    def test_no_elem_counts_domain_cells(self):
        """Test that ``no_elem`` counts only cells inside the domain.

        Test scenario:
            A 2x3 raster with two sentinel cells leaves four real cells, which sizes the
            per-cell parameter array.
        """
        parameters = DP(
            self._raster(np.array([[1, 2, 3], [4, NO_DATA, NO_DATA]], dtype="int32")),
            12,
        )

        assert parameters.no_elem == 4, (
            f"expected 4 domain cells out of 6, got {parameters.no_elem}"
        )
        assert parameters.Par2d.shape == (12, 4), (
            f"the per-cell parameter array should be 12x4, got {parameters.Par2d.shape}"
        )

    def test_cell_indices_exclude_no_data(self):
        """Test that ``celli``/``cellj`` list only domain cells, in row-major order.

        Test scenario:
            A 2x2 raster with one sentinel cell yields the three remaining coordinates.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32")), 12
        )

        assert list(zip(parameters.celli, parameters.cellj)) == [
            (0, 0),
            (0, 1),
            (1, 0),
        ], (
            "expected the three domain cells in row-major order, got "
            f"{list(zip(parameters.celli, parameters.cellj))}"
        )

    def test_raster_array_is_float(self):
        """Test that an integer source raster is promoted to float.

        Test scenario:
            An ``int32`` source must come back floating, since an integer array cannot hold
            the NaN used to mark cells outside the domain.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32")), 12
        )

        assert parameters.raster_array.dtype.kind == "f", (
            f"expected a floating dtype so NaN can be stored, got {parameters.raster_array.dtype}"
        )

    def test_rejects_non_dataset_raster(self):
        """Test that a raw numpy array is rejected with ``TypeError``.

        Test scenario:
            ``Parameters`` requires a pyramids ``Dataset``; passing the array directly must
            fail with a message naming the expected type.
        """
        with pytest.raises(TypeError, match="pyramids Dataset"):
            DP(np.zeros((2, 2)), 12)

    def test_hru_mode_counts_distinct_classes(self):
        """Test that HRU mode sizes the domain by distinct class values, not by cell count.

        Test scenario:
            With ``hru=True`` the raster holds land-use classes rather than one value per
            cell, so ``no_elem`` is the number of distinct non-masked classes. Two cells
            share class ``1``, one holds class ``2``, and the sentinel cell is excluded,
            giving two classes over three domain cells.
        """
        parameters = DP(
            self._raster(np.array([[1, 1], [2, NO_DATA]], dtype="int32")), 12, hru=True
        )

        assert sorted(parameters.values) == [1, 2], (
            f"expected the distinct classes [1, 2], got {sorted(parameters.values)}"
        )
        assert parameters.no_elem == 2, (
            f"expected 2 distinct classes rather than 3 domain cells, got {parameters.no_elem}"
        )

    def test_hru_mode_excludes_masked_cells_from_classes(self):
        """Test that a masked cell never becomes an HRU class.

        Test scenario:
            The sentinel must not appear among the class values; only the two real classes
            are collected.
        """
        parameters = DP(
            self._raster(np.array([[5, 5], [7, NO_DATA]], dtype="int32")), 12, hru=True
        )

        assert NO_DATA not in parameters.values, (
            f"the no-data sentinel leaked into the HRU classes: {parameters.values}"
        )
        assert sorted(parameters.values) == [5, 7], (
            f"expected the distinct classes [5, 7], got {sorted(parameters.values)}"
        )

    def test_rejects_non_list_lumped_par_pos(self):
        """Test that a non-list ``lumped_par_pos`` is rejected with ``ValueError``.

        Test scenario:
            Declaring one lumped parameter but passing its position as a bare integer must
            fail with a message pointing at the expected list form.
        """
        raster = self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32"))

        with pytest.raises(ValueError, match="has to be entered as a list"):
            DP(raster, 12, no_lumped_par=1, lumped_par_pos=7)
