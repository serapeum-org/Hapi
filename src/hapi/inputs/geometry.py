"""The five rasters the flood model reads the channel geometry from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyramids.dataset import Dataset

#: The five rasters the flood model reads, in the order `read_river_geometry` takes them.
RIVER_GEOMETRY_RASTERS = (
    "dem",
    "bankfull_depth",
    "river_width",
    "river_roughness",
    "flood_plain_roughness",
)


@dataclass(frozen=True)
class RiverGeometry:
    """The hydraulic rasters the flood model reads, held together and checked as a set.

    Five arrays that must describe one grid, previously five loose attributes on
    :class:`~hapi.model.catchment.Catchment` assigned by a single loop that checked nothing -- not
    that they shared a shape, and not that they covered the catchment. Two unrelated places
    then dereferenced them: the flood entry point, and the Muskingum routing loop, which read
    `bankfull_depth[x, y]` to decide whether a cell belongs to a hydraulic model.

    Held together, the invariant is checked once, where the file names are still in hand and
    the error can say which raster is the odd one out.

    Attributes:
        dem: `(rows, cols)` elevation.
        bankfull_depth: `(rows, cols)` bankfull depth. A positive value marks a river cell.
        river_width: `(rows, cols)` channel width.
        river_roughness: `(rows, cols)` channel roughness.
        flood_plain_roughness: `(rows, cols)` floodplain roughness.

    Examples:
        - The five must share a shape:
            ```python
            >>> import numpy as np
            >>> from hapi.inputs import RiverGeometry
            >>> flat = np.ones((3, 4))
            >>> RiverGeometry(flat, flat, flat, flat, flat).shape
            (3, 4)

            ```
        - A raster of the wrong size is named, not silently carried:
            ```python
            >>> import numpy as np
            >>> from hapi.inputs import RiverGeometry
            >>> flat = np.ones((3, 4))
            >>> RiverGeometry(flat, flat, np.ones((2, 4)), flat, flat)
            Traceback (most recent call last):
                ...
            ValueError: the river geometry rasters must share one grid, but river_width is (2, 4) and dem is (3, 4)

            ```
    """

    dem: np.ndarray
    bankfull_depth: np.ndarray
    river_width: np.ndarray
    river_roughness: np.ndarray
    flood_plain_roughness: np.ndarray

    def __post_init__(self):
        """Check the five rasters describe one grid.

        Raises:
            ValueError: A raster is a different shape from the DEM, so a cell index would mean
                a different place in each.
        """
        expected = np.shape(self.dem)
        for name in RIVER_GEOMETRY_RASTERS[1:]:
            shape = np.shape(getattr(self, name))
            if shape != expected:
                raise ValueError(
                    f"the river geometry rasters must share one grid, but {name} is {shape} "
                    f"and dem is {expected}"
                )

    @property
    def shape(self) -> tuple[int, int]:
        """tuple[int, int]: The `(rows, cols)` grid all five share."""
        return np.shape(self.dem)

    def covers(self, rows: int, cols: int) -> bool:
        """Report whether the geometry sits on a given grid.

        Args:
            rows: Number of rows to compare against.
            cols: Number of columns.

        Returns:
            bool: True when the geometry's grid is exactly `(rows, cols)`.
        """
        return self.shape == (rows, cols)

    @classmethod
    def from_rasters(
        cls,
        dem_file: str | Path,
        bankfull_depth_file: str | Path,
        river_width_file: str | Path,
        river_roughness_file: str | Path,
        floodplain_roughness_file: str | Path,
    ) -> RiverGeometry:
        """Read the five rasters and check they agree before returning them.

        Args:
            dem_file: Path to the DEM raster.
            bankfull_depth_file: Path to the bankfull-depth raster.
            river_width_file: Path to the river-width raster.
            river_roughness_file: Path to the channel-roughness raster.
            floodplain_roughness_file: Path to the floodplain-roughness raster.

        Returns:
            RiverGeometry: The five arrays, sharing one grid.

        Raises:
            FileNotFoundError: A path does not exist.
            ValueError: The rasters do not share a grid.
        """
        paths = (
            dem_file,
            bankfull_depth_file,
            river_width_file,
            river_roughness_file,
            floodplain_roughness_file,
        )
        arrays = []
        for raster in paths:
            # Five handles, closed as each raster is read: see FlowNetwork.from_rasters.
            with Dataset.read_file(raster) as dataset:
                arrays.append(dataset.read_array(band=0))
        return cls(*arrays)
