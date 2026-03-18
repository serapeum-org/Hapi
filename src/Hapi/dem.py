"""Digital Elevation Model module for the Hapi package.

This module provides the `DEM` class, which extends
``pyramids.dataset.Dataset`` with flow-direction utilities used by
the distributed rainfall-runoff modelling pipeline.
"""
from __future__ import annotations

import numpy as np
from pyramids.dataset import Dataset


class DEM(Dataset):
    """Digital Elevation Model dataset with flow-direction helpers.

    ``DEM`` wraps a GDAL-backed raster dataset (via
    ``pyramids.dataset.Dataset``) and adds methods that convert
    standard 8-direction flow codes (1, 2, 4, 8, 16, 32, 64, 128)
    into downstream-cell indices and upstream-cell lookup tables.

    Args:
        src: A GDAL dataset or a file path to a DEM raster that can
            be opened by ``pyramids.dataset.Dataset``.
    """

    def __init__(self, src):
        super().__init__(src)

    def flow_direction_index(self) -> np.ndarray:
        """Convert flow-direction codes into downstream-cell indices.

        Reads the flow-direction band from the underlying raster and
        maps each of the eight D8 direction codes
        (1, 2, 4, 8, 16, 32, 64, 128) to the row/column index of the
        downstream neighbour cell.

        Returns:
            numpy.ndarray: A 3-D array of shape ``(rows, cols, 2)``.
                The first layer (``[:, :, 0]``) holds the row index
                and the second layer (``[:, :, 1]``) holds the column
                index of the downstream cell.  Cells with no valid
                flow direction are set to ``NaN``.

        Raises:
            ValueError: If the flow-direction raster contains values
                other than 1, 2, 4, 8, 16, 32, 64, or 128 (excluding
                the no-data value).
        """
        # check flow direction input raster
        no_val = self.no_data_value[0]
        cols = self.columns
        rows = self.rows

        fd = self.read_array(band=0)
        fd_val = np.unique(fd[~np.isclose(fd, no_val, rtol=0.00001)])
        fd_should = [1, 2, 4, 8, 16, 32, 64, 128]
        if not all(fd_val[i] in fd_should for i in range(len(fd_val))):
            raise ValueError(
                "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "
            )

        fd_cell = np.ones((rows, cols, 2)) * np.nan

        for i in range(rows):
            for j in range(cols):
                if fd[i, j] == 1:
                    # index of the rows
                    fd_cell[i, j, 0] = i
                    # index of the column
                    fd_cell[i, j, 1] = j + 1
                elif fd[i, j] == 128:
                    fd_cell[i, j, 0] = i - 1
                    fd_cell[i, j, 1] = j + 1
                elif fd[i, j] == 64:
                    fd_cell[i, j, 0] = i - 1
                    fd_cell[i, j, 1] = j
                elif fd[i, j] == 32:
                    fd_cell[i, j, 0] = i - 1
                    fd_cell[i, j, 1] = j - 1
                elif fd[i, j] == 16:
                    fd_cell[i, j, 0] = i
                    fd_cell[i, j, 1] = j - 1
                elif fd[i, j] == 8:
                    fd_cell[i, j, 0] = i + 1
                    fd_cell[i, j, 1] = j - 1
                elif fd[i, j] == 4:
                    fd_cell[i, j, 0] = i + 1
                    fd_cell[i, j, 1] = j
                elif fd[i, j] == 2:
                    fd_cell[i, j, 0] = i + 1
                    fd_cell[i, j, 1] = j + 1

        return fd_cell

    def flow_direction_table(self) -> dict:
        """Build an upstream-cell lookup table from flow directions.

        Uses ``flow_direction_index`` to determine downstream
        neighbours, then inverts the relationship so that each cell
        maps to the list of cells that flow directly into it.

        Returns:
            dict[str, list[tuple[int, int]]]: A dictionary keyed by
                ``"row,col"`` strings.  Each value is a list of
                ``(row, col)`` tuples identifying the cells whose
                flow direction points directly into the key cell.
        """
        flow_direction_index = self.flow_direction_index()

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

        flow_acc_table = {}
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
