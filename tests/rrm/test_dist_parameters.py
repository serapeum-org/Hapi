import numpy as np
from pyramids.dataset import Dataset

from hapi.rrm.parameters import Parameters as DP


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
    assert isinstance(SpatialVarFun.raster_A, np.ndarray)
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
