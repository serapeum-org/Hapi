"""Digital Elevation Model module for the Hapi package.

This module provides the `DEM` class, which extends
``pyramids.dataset.Dataset`` with flow-direction utilities used by
the distributed rainfall-runoff modelling pipeline.
"""
from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset


D8_OFFSETS_ESRI: dict[int, tuple[int, int]] = {
    1: (0, 1),      # east
    2: (1, 1),      # south-east
    4: (1, 0),      # south
    8: (1, -1),     # south-west
    16: (0, -1),    # west
    32: (-1, -1),   # north-west
    64: (-1, 0),    # north
    128: (-1, 1),   # north-east
}

D8_OFFSETS_SAGA: dict[int, tuple[int, int]] = {
    0: (0, 1),      # east
    1: (-1, 1),     # north-east
    2: (-1, 0),     # north
    3: (-1, -1),    # north-west
    4: (0, -1),     # west
    5: (1, -1),     # south-west
    6: (1, 0),      # south
    7: (1, 1),      # south-east
}

D8_OFFSETS_GRASS: dict[int, tuple[int, int]] = {
    1: (-1, 0),     # north
    2: (-1, 1),     # north-east
    3: (0, 1),      # east
    4: (1, 1),      # south-east
    5: (1, 0),      # south
    6: (1, -1),     # south-west
    7: (0, -1),     # west
    8: (-1, -1),    # north-west
}

D8_ENCODINGS: dict[str, dict[int, tuple[int, int]]] = {
    "esri": D8_OFFSETS_ESRI,
    "saga": D8_OFFSETS_SAGA,
    "grass": D8_OFFSETS_GRASS,
}


class DEM(Dataset):
    """Digital Elevation Model dataset with flow-direction helpers.

    ``DEM`` wraps a GDAL-backed raster dataset (via
    ``pyramids.dataset.Dataset``) and adds methods that convert
    D8 flow-direction codes into downstream-cell indices and
    upstream-cell lookup tables.  Three encodings are supported:

    - **ESRI** (default): powers-of-2 codes
      (1, 2, 4, 8, 16, 32, 64, 128).
    - **SAGA**: codes 0--7, starting East counter-clockwise.
    - **GRASS**: codes 1--8, starting North clockwise.

    Args:
        src: A GDAL dataset or a file path to a DEM raster that can
            be opened by ``pyramids.dataset.Dataset``.
    """

    def __init__(self, src):
        """Initialize the DEM instance."""
        super().__init__(src)

    def flow_direction_index(
        self, encoding: str = "esri"
    ) -> np.ndarray:
        """Convert flow-direction codes into downstream-cell indices.

        Reads the flow-direction band from the underlying raster and
        maps each D8 direction code to the row/column index of the
        downstream neighbour cell.

        Args:
            encoding: The D8 flow-direction encoding used by the
                raster.  Supported values:

                - ``"esri"`` (default) -- ArcGIS / ESRI powers-of-2
                  codes (1, 2, 4, 8, 16, 32, 64, 128).
                - ``"saga"`` -- SAGA GIS codes (0--7, starting East
                  counter-clockwise).  Produced by QGIS Processing
                  SAGA tools such as *Fill Sinks* and *Channel
                  Network*.
                - ``"grass"`` -- GRASS GIS codes (1--8, starting
                  North clockwise).  Produced by QGIS Processing
                  GRASS tools such as *r.watershed*.

        Returns:
            numpy.ndarray: A 3-D array of shape ``(rows, cols, 2)``.
                The first layer (``[:, :, 0]``) holds the row index
                and the second layer (``[:, :, 1]``) holds the column
                index of the downstream cell.  Cells with no valid
                flow direction are set to ``NaN``.

        Raises:
            ValueError: If *encoding* is not one of the supported
                names, or if the raster contains direction values
                outside the expected set for the chosen encoding.
        """
        encoding = encoding.lower()
        if encoding not in D8_ENCODINGS:
            raise ValueError(
                f"Unsupported encoding {encoding!r}. "
                f"Choose from {list(D8_ENCODINGS)}"
            )
        offsets = D8_ENCODINGS[encoding]

        no_val = self.no_data_value[0]
        cols = self.columns
        rows = self.rows

        fd = self.read_array(band=0)
        fd_val = np.unique(fd[~np.isclose(fd, no_val, rtol=0.00001)])
        valid_codes = set(offsets)
        if not all(int(v) in valid_codes for v in fd_val):
            raise ValueError(
                f"Flow direction raster should contain only "
                f"{sorted(valid_codes)} for encoding {encoding!r}"
            )

        fd_cell = np.full((rows, cols, 2), np.nan)

        row_idx, col_idx = np.meshgrid(
            np.arange(rows), np.arange(cols), indexing="ij"
        )
        d_row = np.full((rows, cols), np.nan)
        d_col = np.full((rows, cols), np.nan)

        for code, (dr, dc) in offsets.items():
            mask = fd == code
            d_row[mask] = dr
            d_col[mask] = dc

        valid = ~np.isnan(d_row)
        fd_cell[valid, 0] = row_idx[valid] + d_row[valid]
        fd_cell[valid, 1] = col_idx[valid] + d_col[valid]

        return fd_cell

    def flow_direction_table(self, encoding: str = "esri") -> dict:
        """Build an upstream-cell lookup table from flow directions.

        Uses ``flow_direction_index`` to determine downstream
        neighbours, then inverts the relationship so that each cell
        maps to the list of cells that flow directly into it.

        Args:
            encoding: The D8 flow-direction encoding.  See
                ``flow_direction_index`` for supported values.

        Returns:
            dict[str, list[tuple[int, int]]]: A dictionary keyed by
                ``"row,col"`` strings.  Each value is a list of
                ``(row, col)`` tuples identifying the cells whose
                flow direction points directly into the key cell.
        """
        flow_direction_index = self.flow_direction_index(encoding=encoding)

        rows = self.rows
        cols = self.columns

        cell_i = []
        cell_j = []
        celli_content = []
        cellj_content = []
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(flow_direction_index[i, j, 0]):
                    # store the indexes of not empty cells and the indexes stored inside these cells
                    cell_i.append(i)
                    cell_j.append(j)
                    # store the index of the receiving cells
                    celli_content.append(flow_direction_index[i, j, 0])
                    cellj_content.append(flow_direction_index[i, j, 1])

        flow_acc_table: dict[str, list[tuple[int, int]]] = {}
        # for each cell store the directly giving cells
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(flow_direction_index[i, j, 0]):
                    # get the indexes of the cell and use it as a key in a dictionary
                    name = str(i) + "," + str(j)
                    flow_acc_table[name] = []
                    for k in range(len(celli_content)):
                        # search if any cell are giving this cell
                        if i == celli_content[k] and j == cellj_content[k]:
                            flow_acc_table[name].append((cell_i[k], cell_j[k]))

        return flow_acc_table
