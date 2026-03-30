import numpy as np
import pandas as pd
import pytest

from hapi.rrm.hbv import HBV
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
from hapi.rrm.hbv_lake import HBVLake


@pytest.fixture()
def coello_lumped_meteo_data() -> np.ndarray:
    path = "tests/rrm/data/coello/meteo-lumped-data-MSWEP.csv"
    data = pd.read_csv(path, header=0, delimiter=",", index_col=0).to_numpy()
    return data


@pytest.fixture()
def coello_lumped_parameters() -> list:
    path = "tests/rrm/data/coello/coello-lumpedparameter-muskingum.txt"
    parameters = pd.read_csv(path, index_col=0, header=None)[1].tolist()
    return parameters


def test_hbv_bergestrom(
    coello_lumped_meteo_data: np.ndarray, coello_lumped_parameters: list
):
    p = coello_lumped_meteo_data[:, 0]
    et = coello_lumped_meteo_data[:, 1]
    t = coello_lumped_meteo_data[:, 2]
    tm = coello_lumped_meteo_data[:, 3]
    initial_conditions = [0, 10, 10, 10, 0]
    snow = False
    q_init = None
    conceptual_model = HBVBergestrom92()

    quz, qlz, state_variables = conceptual_model.simulate(
        p,
        t,
        et,
        tm,
        coello_lumped_parameters,
        init_st=initial_conditions,
        q_init=q_init,
        snow=snow,
    )
    assert len(quz) == len(qlz) == len(state_variables) == len(p) + 1
    assert state_variables.shape == (len(p) + 1, 5)

    expected_quz = np.array([3.5651002, 1.6847236, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert all(np.isclose(quz, expected_quz))

    expected_qlz = np.array(
        [
            0.32230002,
            0.49229392,
            0.57979316,
            0.56222373,
            0.54827434,
            0.58935225,
            0.6560198,
            0.7031159,
            0.71430916,
            0.70471734,
            0,
        ]
    )
    assert all(np.isclose(qlz, expected_qlz))

    expected_state_variables = np.array(
        [
            [0.0, 10.0, 10.0, 10.0, 0.0],
            [0.0, 9.767188, 3.0408764, 14.782106, 0.0],
            [0.0, 10.355642, 0.0, 17.409445, 0.0],
            [0.0, 10.257657, 0.0, 16.881887, 0.0],
            [0.0, 10.617212, 0.0, 16.46303, 0.0],
            [0.0, 19.594128, 0.0, 17.696476, 0.0],
            [0.0, 24.455137, 0.0, 19.698301, 0.0],
            [0.0, 26.748932, 0.0, 21.112457, 0.0],
            [0.0, 27.195393, 0.0, 21.448555, 0.0],
            [0.0, 26.771988, 0.0, 21.160543, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(state_variables, expected_state_variables)


def test_hbv(coello_lumped_meteo_data: np.ndarray, coello_lumped_parameters: list):
    p = coello_lumped_meteo_data[:, 0]
    et = coello_lumped_meteo_data[:, 1]
    t = coello_lumped_meteo_data[:, 2]
    tm = coello_lumped_meteo_data[:, 3]
    initial_conditions = [0, 10, 10, 10, 0]
    snow = False
    q_init = None
    conceptual_model = HBV()

    quz, qlz, state_variables = conceptual_model.simulate(
        p,
        t,
        et,
        coello_lumped_parameters,
        init_st=initial_conditions,
        ll_temp=tm,
        q_init=q_init,
        snow=snow,
    )
    assert len(quz) == len(qlz) == len(state_variables) == len(p) + 1
    assert state_variables.shape == (len(p) + 1, 5)

    expected_quz = np.array([np.inf, 4.115642, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert all(np.isclose(quz, expected_quz))

    expected_qlz = np.array(
        [
            0.3223,
            0.49229392,
            0.4764273,
            0.46107203,
            0.44621167,
            0.4318303,
            0.46364802,
            0.5265117,
            0.5704197,
            0.57703793,
            0.5617763,
        ]
    )
    assert all(np.isclose(qlz, expected_qlz))

    expected_state_variables = np.array(
        [
            [0.0, 10.0, 10.0, 10.0, 0.0],
            [0.0, 10.94771, 0.0, 14.782106, 0.0],
            [0.0, 10.692835, 0.0, 14.305678, 0.0],
            [0.0, 11.413589, 0.0, 13.844606, 0.0],
            [0.0, 11.321631, 0.0, 13.398396, 0.0],
            [0.0, 11.776372, 0.0, 12.966565, 0.0],
            [0.0, 21.125572, 0.0, 13.921956, 0.0],
            [0.0, 26.192625, 0.0, 15.809565, 0.0],
            [0.0, 28.677124, 0.0, 17.12799, 0.0],
            [0.0, 29.34668, 0.0, 17.326714, 0.0],
            [0.0, 29.167826, 0.0, 16.868452, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(state_variables, expected_state_variables)


def test_hbv_lake(coello_lumped_meteo_data: np.ndarray, coello_lumped_parameters: list):
    conceptual_model = HBVLake()

    p_lake = np.array(
        [
            3.76561010e-02,
            6.96146940e-02,
            2.78458776e-01,
            7.88452500e-02,
            8.89887938e-01,
            1.54292840e01,
            1.70514966e01,
            1.31323313e00,
            1.53350692e-01,
            0.00000000e00,
            4.34211460e-02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.35691100e-03,
            3.91494616e-01,
            0.00000000e00,
            4.34211460e-02,
            0.00000000e00,
        ]
    )
    t = np.array(
        [
            17.6984801,
            17.7632534,
            18.57975622,
            20.0210873,
            21.86323426,
            21.56323426,
            19.7210873,
            18.27975622,
            17.4632534,
            17.3984801,
            18.09550343,
            19.44599168,
            21.2400513,
            23.19884866,
            25.01794653,
            26.41461981,
            27.17179687,
            27.17179687,
            26.41461981,
            25.01794653,
        ]
    )
    et = np.array(
        [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]
    )

    parameters = [
        332.976352,
        0.351721971,
        0.0061904952,
        0.300592039,
        0.308911517,
        0.00714373545,
        0.01138262,
        0.289143383,
        0.266252731,
        0.901418103,
        1.02616694,
        0.621835363,
        0.172067046,
    ]
    p2 = [3.6, 133.98, 70.64]
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
    tm = np.array(
        [
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
            21.83147458,
        ]
    )
    initial_conditions = [0.0, 5.0, 5.0, 5.0, 0.0, 10211440220.48255]

    q_sim, state_variables = conceptual_model.simulate(
        p_lake,
        t,
        et,
        parameters,
        p2,
        curve,
        init_st=initial_conditions,
        q_0=0,
        ll_temp=tm,
        lake_sim=True,
    )
    assert len(q_sim) == len(state_variables) == len(p_lake) + 1
    state_variables = np.array((state_variables))
    assert state_variables.shape == (len(p_lake) + 1, 6)

    expected_q_sim = np.array(
        [
            0,
            2.0001480481915865,
            1.9985492645588254,
            1.9974008310010531,
            1.9957792189225483,
            1.9959804018363263,
            2.0292485403628544,
            2.06662373945165,
            2.0684290114566273,
            2.0675683881660802,
            2.0663129614237716,
            2.065111089542004,
            2.063767046981697,
            2.0623810289635,
            2.06095506831234,
            2.0594911697597524,
            2.057994426917328,
            2.0573572818751087,
            2.0558028078454287,
            2.0543170966103084,
            2.0527045917810325,
        ]
    )
    assert all(np.isclose(q_sim, expected_q_sim))

    expected_state_variables = np.array(
        [
            [0, 5.0, 5.0 + 00, 5.0, 0, 1.02114402e10],
            [0, 5.32435407e00, 4.38951493e00, 5.20630898e00, 0, 1.02114164e10],
            [0, 5.67271196, 3.79554683, 5.41026962, 0, 1.02113933e10],
            [0, 6.18277830, 3.26065181, 5.61190865, 0, 1.02113767e10],
            [0, 6.53473580, 2.68529241, 5.81125251, 0, 1.02113533e10],
            [0, 7.50855931, 2.32389212, 6.00832730, 0, 1.02113562e10],
            [0, 1.94567032e01, 5.85669917, 6.20315887, 0, 1.02118371e10],
            [0, 3.07624194e01, 1.15730113e01, 6.39577275, 0, 1.02123775e10],
            [0, 3.17471727e01, 1.14409752e01, 6.58619418, 0, 1.02124036e10],
            [0, 3.20534240e01, 1.08075818e01, 6.77444811, 0, 1.02123911e10],
            [0, 3.22703072e01, 1.01184213e01, 6.96055922, 0, 1.02123730e10],
            [0, 3.25112398e01, 9.46103893, 7.14455190, 0, 1.02123556e10],
            [0, 3.27259850e01, 8.79613244, 7.32645025, 0, 1.02123362e10],
            [0, 3.29393821e01, 8.14282242, 7.50627813, 0, 1.02123161e10],
            [0, 3.31513582e01, 7.50065309, 7.68405910, 0, 1.02122955e10],
            [0, 3.33619645e01, 6.86917454, 7.85981645, 0, 1.02122744e10],
            [0, 3.35721423e01, 6.24855127, 8.03357322, 0, 1.02122527e10],
            [0, 3.40030883e01, 5.81362425, 8.20535219, 0, 1.02122435e10],
            [0, 3.42102630e01, 5.20913001, 8.37517586, 0, 1.02122210e10],
            [0, 3.44416819e01, 4.63339920, 8.54306649, 0, 1.02121996e10],
            [0, 3.46484619e01, 4.04627882, 8.70904609, 0, 1.02121763e10],
        ]
    )
    assert np.allclose(state_variables, expected_state_variables)
