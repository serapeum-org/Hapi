"""Unit test for Parameters.hru_hand on a synthetic catchment."""

import numpy as np
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
    assert np.allclose(hand, expected_hand)
    assert np.allclose(dtnd, expected_dtnd)
