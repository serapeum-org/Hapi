import datetime as dt

import numpy as np

from Hapi.catchment import Lake
from Hapi.rrm.hbv_lake import HBVLake


def test_lake():
    start_date = "2012.06.14 19:00:00"
    end_date = "2012.6.15 14:00:00"
    fmt = "%Y.%m.%d %H:%M:%S"
    root_dir = "tests/rrm/data/jiboa"
    lake = Lake(
        start=start_date,
        end=end_date,
        fmt="%Y.%m.%d %H:%M:%S",
        temporal_resolution="Hourly",
        split=True,
    )

    assert len(lake.Index) == 20
    assert lake.start == dt.datetime.strptime(start_date, fmt)
    assert lake.end == dt.datetime.strptime(end_date, fmt)
    lake_meteo_path = f"{root_dir}/lake-data.csv"
    lake_parameters_path = f"{root_dir}/lake-parameters.txt"
    lake.read_meteo_data(lake_meteo_path, fmt="%d.%m.%Y %H:%M")
    lake.read_parameters(lake_parameters_path)
    lake_cat_area = 133.98
    lake_area = 70.64
    snow = 0
    outflow_cell = [2, 1]  # 4km
    initial_cond_lake = [0, 5, 5, 5, 0, 1.021144022048255e+10]
    curve = np.array(
        [
            [1.00000000e-02, 1.01196261e10],
            [8.40000000e-02, 1.01543596e10],
            [9.99000000e-01, 1.01958839e10],
            [1.11000000e00, 1.01980622e10],
            [1.22600000e00, 1.02002405e10],
            [2.90200000e00, 1.02244921e10],
            [3.15200000e00, 1.02273965e10],
            [4.18500000e00, 1.02382879e10],
            [1.05920000e01, 1.02847580e10],
            [1.16520000e01, 1.02905668e10],
            [1.53900000e01, 1.03087192e10],
            [2.50430000e01, 1.03450240e10],
            [3.85590000e01, 1.03824905e10],
            [8.88000000e01, 1.04539383e10],
        ]
    )
    lake.read_lumped_model(
        HBVLake, lake_cat_area, lake_area, initial_cond_lake, outflow_cell, curve, snow
    )
