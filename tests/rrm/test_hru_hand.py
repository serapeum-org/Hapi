"""Unit test for Parameters.hru_hand on a synthetic catchment."""

import numpy as np
import pytest
from pyramids.dataset import Dataset

from hapi.rrm.parameters import Parameters

NO_DATA = -9999.0


def _dataset(arr: np.ndarray) -> Dataset:
    """Build an in-memory Dataset from a 2-D array."""
    rows, _ = arr.shape
    return Dataset.create_from_array(
        arr.astype(np.float64),
        top_left_corner=(0.0, float(rows)),
        cell_size=1.0,
        epsg=4326,
        no_data_value=NO_DATA,
    )


def test_hru_hand_synthetic_catchment():
    """Every cell flows east into a river in the last column.

    With elevations [30, 20, 10] and flow path lengths [200, 100, 0] per
    row, the height above nearest drainage is the elevation difference to
    the river column and the distance to the nearest drainage is the flow
    path length difference.
    """
    dem = _dataset(np.array([[30.0, 20.0, 10.0]] * 3))
    # ESRI D8 code 1 = east everywhere
    flow_direction = _dataset(np.full((3, 3), 1.0))
    flow_path_length = _dataset(np.array([[200.0, 100.0, 0.0]] * 3))
    river = _dataset(np.array([[0.0, 0.0, 1.0]] * 3))

    hand, dtnd = Parameters.hru_hand(dem, flow_direction, flow_path_length, river)

    expected_hand = np.array([[20.0, 10.0, 0.0]] * 3)
    expected_dtnd = np.array([[200.0, 100.0, 0.0]] * 3)
    assert np.allclose(hand, expected_hand), f"HAND mismatch: {hand}"
    assert np.allclose(dtnd, expected_dtnd), f"DTND mismatch: {dtnd}"


def test_hru_hand_river_cells_are_their_own_drainage():
    """River cells get HAND = 0 even on an asymmetric grid.

    Test scenario:
        The first cell in scan order is itself a river cell, and the two
        rows drain to different river cells with different elevations, so
        any stale carry-over of the previous cell's trace would produce a
        nonzero HAND for a river cell.
    """
    dem = _dataset(np.array([[5.0, 20.0, 30.0], [7.0, 40.0, 60.0]]))
    # ESRI D8 code 16 = west everywhere
    flow_direction = _dataset(np.full((2, 3), 16.0))
    flow_path_length = _dataset(np.array([[0.0, 10.0, 20.0], [0.0, 11.0, 22.0]]))
    river = _dataset(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

    hand, dtnd = Parameters.hru_hand(dem, flow_direction, flow_path_length, river)

    expected_hand = np.array([[0.0, 15.0, 25.0], [0.0, 33.0, 53.0]])
    expected_dtnd = np.array([[0.0, 10.0, 20.0], [0.0, 11.0, 22.0]])
    assert np.allclose(hand, expected_hand), f"HAND mismatch: {hand}"
    assert np.allclose(dtnd, expected_dtnd), f"DTND mismatch: {dtnd}"


def test_hru_hand_no_river_raises():
    """A flow path that never reaches a river raises a boundary error.

    Test scenario:
        With no river cell anywhere, tracing the flow direction runs off
        the grid edge; hru_hand converts that anomaly into a ValueError
        about the catchment boundaries.
    """
    dem = _dataset(np.array([[30.0, 20.0, 10.0]] * 3))
    flow_direction = _dataset(np.full((3, 3), 1.0))
    flow_path_length = _dataset(np.array([[200.0, 100.0, 0.0]] * 3))
    river = _dataset(np.zeros((3, 3)))

    with pytest.raises(ValueError, match="boundaries"):
        Parameters.hru_hand(dem, flow_direction, flow_path_length, river)
