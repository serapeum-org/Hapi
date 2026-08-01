import numpy as np

from hapi.dem import DEM


def test_flow_direction_index(coello_df_4000: DEM):
    fd_cell = coello_df_4000.flow_direction_index()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (coello_df_4000.rows, coello_df_4000.columns, 2)


def test_flow_direction_table_type(coello_df_4000: DEM):
    fd_table = coello_df_4000.flow_direction_table()
    assert isinstance(fd_table, dict)


def test_flow_direction_table_values(coello_df_4000: DEM, coello_fdt):
    fd_table = coello_df_4000.flow_direction_table()
    assert fd_table == coello_fdt
