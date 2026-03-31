import pytest

coello_dates = ["2009-01-01", "2009-01-10"]
area = 1530.0
IC = [0, 10, 10, 10, 0]
snow = False
gauges_datefmt = "%Y-%m-%d"


@pytest.fixture(scope="module")
def lumped_parameters_path() -> str:
    return "tests/rrm/data/coello/coello-lumpedparameter-muskingum.txt"


@pytest.fixture(scope="module")
def lumped_meteo_data_path() -> str:
    return "tests/rrm/data/coello/meteo-lumped-data-MSWEP.csv"


@pytest.fixture(scope="module")
def lumped_gauges_path() -> str:
    return "examples/hydrological-model/data/lumped_model/Qout_c.csv"


@pytest.fixture(scope="module")
def coello_rrm_date() -> list:
    return coello_dates


@pytest.fixture(scope="module")
def coello_AreaCoeff() -> float:
    return area


@pytest.fixture(scope="module")
def coello_InitialCond() -> list:
    return IC


@pytest.fixture(scope="module")
def coello_Snow() -> int:
    return snow


@pytest.fixture(scope="module")
def coello_gauges_date_fmt() -> str:
    return gauges_datefmt


# @pytest.fixture(scope="module")
# def coello_lumpedmodel_RoutingFn() -> (inflow: Any, Qinitial: Any, k: Any, x: Any, dt: Any):
#     return Routing.muskingum_v
