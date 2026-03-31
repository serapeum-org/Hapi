from pathlib import Path

import numpy as np
from geopandas import GeoDataFrame
from pyramids.multidataset import MultiDataset as Datacube

from hapi.inputs import Inputs


def test_prepare_inputs(
    coello_prec_path: str, coello_acc_path: str, rrm_test_results: str
):
    """Test prepare_inputs function in Inputs class"""
    rpath = Path(f"{rrm_test_results}/prepare_inputs")
    # if rpath.exists():
    #     rpath.unlink()

    inputs = Inputs(coello_acc_path)
    inputs.prepare_inputs(coello_prec_path, rpath)
    assert rpath.exists()
    files = list(rpath.iterdir())
    assert len(files) == 10
    cube = Datacube.read_multiple_files(str(rpath), with_order=False)
    cube.open_multi_dataset()
    # if rpath.exists():
    #     rpath.unlink()


class TestExtractParameters:
    def test_as_raster(
        self,
        download_03_parameter,
        coello_prec_path: str,
        coello_acc_path: str,
        rrm_test_results: str,
    ):
        """Test extract_parameters function in Inputs class"""
        rpath = Path(f"{rrm_test_results}/extract_parameter")
        # if rpath.exists():
        #     rpath.unlink()

        inputs = Inputs(coello_acc_path)
        inputs.extract_parameters(None, "3", as_raster=True, save_to=str(rpath))
        assert rpath.exists()
        files = list(rpath.iterdir())
        assert len(files) == 19
        cube = Datacube.read_multiple_files(str(rpath), with_order=False)
        cube.open_multi_dataset()
        # if rpath.exists():
        #     rpath.unlink()

    # def test_as_raster_false(
    #     self,
    #     download_03_parameter,
    #     coello_acc_path: str,
    #     coello_basin: GeoDataFrame,
    # ):
    #     """Test extract_parameters function in Inputs class"""
    #     inputs = Inputs(coello_acc_path)
    #     par = inputs.extract_parameters(coello_basin, "03")
    #     par_vals = [
    #         0.8952,
    #         1.0,
    #         1.230,
    #         3.099,
    #         0.07358,
    #         0.05464,
    #         548.72,
    #         3.085,
    #         1.0,
    #         0.911,
    #         0.8657,
    #         0.5961,
    #         0.09381,
    #         38.313,
    #         3.919,
    #         1.873,
    #         1.0,
    #         0.20,
    #     ]
    #     assert np.isclose(
    #         par.loc[:, "max"].to_list(), par_vals, atol=0.001, rtol=0.001
    #     ).all()


def test_extract_parameters_boundaries(
    download_max_min_parameter, coello_basin: GeoDataFrame
):
    """Test extract_parameters function in Inputs class"""
    par = Inputs.extract_parameters_boundaries(coello_basin)
    upper_bound_valid = [
        2.262565,
        1,
        1.49494,
        4.502295,
        0.138411,
        0.079819,
        608.27124,
        3.669066,
        1,
        1,
        0.865717,
        0.8,
        0.107622,
        72.304459,
        5.275979,
        2.34628,
        1,
        0.2,
    ]
    lower_bound_valid = [
        -0.966476,
        1,
        1.044623,
        0.55258,
        0.035901,
        0.011214,
        50,
        1.148444,
        1,
        0.460137,
        0.227757,
        0.123802,
        0.005037,
        16.123743,
        1.657871,
        1.194185,
        1,
        0.2,
    ]
    assert np.isclose(
        par.loc[:, "ub"], upper_bound_valid, rtol=0.00001, atol=0.00001
    ).all()
    assert np.isclose(
        par.loc[:, "lb"], lower_bound_valid, rtol=0.00001, atol=0.00001
    ).all()


def test_create_lumped_parameter():
    path = "tests/rrm/data/coello/prec"
    lumped_data = Inputs.create_lumped_inputs(
        path,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    validation_values = [
        0.0,
        0.3153739,
        43.65052,
        0.9195719,
        8.736267,
        6.411966,
        21.155485,
        1.027534,
        4.1313076,
        1.5406969,
    ]
    assert np.isclose(lumped_data, validation_values, atol=0.001, rtol=0.001).all()
