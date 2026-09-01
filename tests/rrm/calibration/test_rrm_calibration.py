import datetime as dt

import numpy as np
import statista.descriptors as metrics

from hapi.calibration import Calibration
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped


def test_read_parameters_bounds(
    coello_rrm_date: list,
    lower_bound: list,
    upper_bound: list,
):
    Coello = Calibration("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Maxbas = True
    Snow = False
    Coello.read_parameters_bound(lower_bound, upper_bound, Snow, maxbas=Maxbas)
    assert isinstance(Coello.UB, np.ndarray)
    assert isinstance(Coello.LB, np.ndarray)
    assert isinstance(Coello.snow, bool)
    assert isinstance(Coello.maxbas, bool)


def test_lumped_calibration(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
    lumped_parameters_path: str,
    coello_Snow: bool,
    lower_bound: list,
    upper_bound: list,
    lumped_gauges_path: str,
    coello_gauges_date_fmt: str,
    history_files: str,
):
    Coello = Calibration("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.read_lumped_inputs(lumped_meteo_data_path)
    Coello.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Maxbas = True
    Coello.read_parameters_bound(lower_bound, upper_bound, coello_Snow, maxbas=Maxbas)

    parameters = []
    # Routing
    Route = 1
    routing_fn = Routing.triangular_routing_1

    basic_inputs = dict(Route=Route, RoutingFn=routing_fn, InitialValues=parameters)

    # discharge gauges
    Coello.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)

    OF_args = []
    objective_function = metrics.rmse

    Coello.read_objective_function(objective_function, OF_args)

    ApiObjArgs = dict(
        hms=100,
        hmcr=0.95,
        par=0.65,
        dbw=2000,
        fileout=1,
        xinit=0,
        filename=history_files,
    )

    for i in range(len(ApiObjArgs)):
        print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

    # pll_type = 'POA'
    pll_type = None

    ApiSolveArgs = dict(
        store_sol=True, display_opts=True, store_hst=False, hot_start=False
    )

    optimization_args = [ApiObjArgs, pll_type, ApiSolveArgs]

    # cal_parameters = Coello.calibrate_lumped(basic_inputs, optimization_args, print_error=None)

    # assert len(Coello.Qsim) == 1095 and Coello.Qsim.columns.to_list() == ['q']


class TestDistributed:
    def test_create_calibration_instance(
        self, coello_start_date: str, coello_end_date: str
    ):
        coello = Calibration(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        assert coello.spatial_resolution == "distributed"
        assert coello.routing_method == "Muskingum"
        assert isinstance(coello.start, dt.datetime)

    def test_read_objective_fn(self, coello_start_date: str, coello_end_date: str):
        coello = Calibration(
            "coello",
            coello_start_date,
            coello_end_date,
        )
        coello.read_objective_function(metrics.rmse, [])
        assert coello.objective_function == metrics.rmse
        assert coello.OFArgs == []
