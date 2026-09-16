import datetime as dt
from typing import List

# @pytest.fixture(scope="module")
# def gauges_file_extension() -> str:
# return ".csv"
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def lower_bound() -> list:
    path = r"examples/hydrological-model/data/lumped_model/LB-3.txt"
    return pd.read_csv(path, index_col=0, header=None)


@pytest.fixture(scope="module")
def upper_bound() -> list:
    path = r"examples/hydrological-model/data/lumped_model/UB-3.txt"
    return pd.read_csv(path, index_col=0, header=None)


@pytest.fixture(scope="module")
def history_files() -> str:
    return (
        "examples\Hydrological model\Lumped_History"
        + str(dt.datetime.now())[0:10]
        + ".txt"
    )
