import os
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pytest
from geopandas import GeoDataFrame
from osgeo import gdal

from Hapi.parameters.parameters import Parameter
from tests.rrm.calibration.conftest import *
from tests.rrm.catchment.conftest import *


@pytest.fixture(scope="session")
def hapi_data_dir() -> str:
    data_dir = os.getenv("HAPI_DATA_DIR")

    if data_dir is None or not os.path.exists(data_dir):
        raise ValueError("please set the `HAPI_DATA_DIR` emvironment variable")
    return data_dir


@pytest.fixture(scope="session")
def download_03_parameter(hapi_data_dir: str):
    """Download Parameter Set 03"""
    if not os.path.exists(f"{hapi_data_dir}/3"):
        par = Parameter()
        par.get_parameter_set(3)


@pytest.fixture(scope="session")
def download_max_min_parameter(hapi_data_dir: str):
    """Download Parameter Set 03"""
    par = Parameter()
    if not os.path.exists(f"{hapi_data_dir}/max"):
        par.get_parameter_set("max")
    if not os.path.exists(f"{hapi_data_dir}/min"):
        par.get_parameter_set("min")


@pytest.fixture(scope="module")
def rrm_test_results() -> str:
    return "tests/rrm/data/test_results"


@pytest.fixture(scope="module")
def coello_basin() -> GeoDataFrame:
    return gpd.read_file("tests/rrm/data/coello/coello-basin-extended.geojson")


@pytest.fixture(scope="module")
def coello_start_date() -> str:
    return "2009-01-01"


@pytest.fixture(scope="module")
def coello_end_date() -> str:
    return "2009-01-10"


@pytest.fixture(scope="module")
def coello_evap_path() -> str:
    return "tests/rrm/data/coello/evap"


@pytest.fixture(scope="module")
def coello_prec_path() -> str:
    return "tests/rrm/data/coello/prec"


@pytest.fixture(scope="module")
def coello_temp_path() -> str:
    return "tests/rrm/data/coello/temp"


@pytest.fixture(scope="module")
def coello_acc_path() -> str:
    return "tests/rrm/data/coello/gis/acc4000.tif"


@pytest.fixture(scope="module")
def coello_acc_raster() -> gdal.Dataset:
    return gdal.Open("tests/rrm/data/coello/gis/acc4000.tif")


@pytest.fixture(scope="module")
def coello_acc_values() -> List:
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        13,
        15,
        16,
        17,
        23,
        43,
        44,
        48,
        55,
        59,
        63,
        86,
        88,
    ]


@pytest.fixture(scope="module")
def coello_fd_path() -> str:
    return "tests/rrm/data/coello/gis/fd4000.tif"


@pytest.fixture(scope="module")
def coello_fdt() -> Dict:
    return {
        "1,5": [],
        "1,6": [],
        "1,7": [],
        "2,5": [(1, 5)],
        "2,6": [],
        "2,7": [(1, 6), (1, 7)],
        "3,3": [],
        "3,4": [],
        "3,5": [(2, 5)],
        "3,6": [],
        "3,7": [(2, 6), (2, 7)],
        "3,8": [],
        "3,9": [],
        "4,3": [],
        "4,4": [(3, 3), (3, 4), (4, 3), (5, 3)],
        "4,5": [(3, 5), (3, 6)],
        "4,6": [],
        "4,7": [(3, 7)],
        "4,8": [(3, 8), (3, 9)],
        "4,9": [],
        "5,3": [],
        "5,4": [],
        "5,5": [(4, 4), (4, 5), (4, 6)],
        "5,6": [],
        "5,7": [],
        "5,8": [(4, 7), (4, 8), (5, 7)],
        "5,9": [(4, 9)],
        "6,2": [],
        "6,3": [],
        "6,4": [],
        "6,5": [(5, 4), (5, 5), (5, 6), (6, 4)],
        "6,6": [],
        "6,7": [],
        "6,8": [],
        "6,9": [(5, 8), (5, 9)],
        "7,1": [],
        "7,2": [(7, 1)],
        "7,3": [(6, 2)],
        "7,4": [(6, 3), (7, 3), (8, 3)],
        "7,5": [(7, 4), (8, 4)],
        "7,6": [(6, 5), (6, 6), (7, 5), (8, 5), (8, 6)],
        "7,7": [(6, 7)],
        "7,8": [],
        "7,9": [(6, 8), (6, 9)],
        "8,1": [],
        "8,2": [(8, 1)],
        "8,3": [(7, 2), (8, 2), (9, 2)],
        "8,4": [(9, 3), (9, 4)],
        "8,5": [],
        "8,6": [],
        "8,7": [(7, 6)],
        "8,8": [(7, 7), (7, 8), (8, 7)],
        "8,9": [],
        "8,10": [(7, 9)],
        "8,11": [(8, 10)],
        "8,12": [],
        "9,1": [],
        "9,2": [(9, 1), (10, 1)],
        "9,3": [(10, 2), (10, 3)],
        "9,4": [],
        "9,5": [(10, 4)],
        "9,6": [(9, 5), (10, 5)],
        "9,7": [(9, 6)],
        "9,8": [(9, 7)],
        "9,9": [(8, 8), (9, 8)],
        "9,10": [(8, 9)],
        "9,11": [(9, 10)],
        "9,12": [(8, 11), (8, 12), (9, 11), (10, 11)],
        "10,0": [],
        "10,1": [(10, 0), (11, 0), (11, 1)],
        "10,2": [(11, 2)],
        "10,3": [],
        "10,4": [],
        "10,5": [],
        "10,7": [],
        "10,8": [(10, 7)],
        "10,9": [(10, 8)],
        "10,10": [(9, 9), (10, 9)],
        "10,11": [(10, 10), (11, 10), (11, 11)],
        "10,12": [],
        "10,13": [(9, 12), (10, 12)],
        "11,0": [],
        "11,1": [(12, 1)],
        "11,2": [(12, 2)],
        "11,9": [],
        "11,10": [(11, 9)],
        "11,11": [],
        "12,1": [],
        "12,2": [],
    }


@pytest.fixture(scope="module")
def coello_cat_area() -> int:
    return 1530


@pytest.fixture(scope="module")
def coello_initial_cond() -> List:
    return [0, 5, 5, 5, 0]


@pytest.fixture(scope="module")
def coello_parameter_bounds() -> Tuple[list, list]:
    ub = np.loadtxt(
        "tests/rrm/data/coello/calibration/upper-bound-tot.txt", usecols=0
    ).tolist()
    lb = np.loadtxt(
        "tests/rrm/data/coello/calibration/lower-bound-tot.txt", usecols=0
    ).tolist()
    return lb, ub


@pytest.fixture(scope="module")
def coello_gauges_table() -> str:
    return "tests/rrm/data/coello/calibration/gauges.csv"


@pytest.fixture(scope="module")
def coello_gauge_names() -> List:
    return [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def coello_gauges_path() -> str:
    return "tests/rrm/data/coello/calibration"


@pytest.fixture(scope="module")
def coello_no_parameters() -> int:
    return 12


@pytest.fixture(scope="module")
def coello_rows() -> int:
    return 13


@pytest.fixture(scope="module")
def coello_cols() -> int:
    return 14


@pytest.fixture(scope="module")
def coello_shape(coello_rows, coello_cols) -> Tuple:
    return coello_rows, coello_cols


@pytest.fixture(scope="module")
def coello_parameters() -> np.ndarray:
    return np.loadtxt("tests/rrm/data/coello/calibration/parameters.csv")


@pytest.fixture(scope="module")
def coello_parameters_dist() -> np.ndarray:
    return np.load("tests/rrm/data/coello/calibration/parameters.npy")


@pytest.fixture(scope="module")
def coello_dist_parameters_maxbas() -> str:
    return "tests/rrm/data/coello/parameters/maxbas"
