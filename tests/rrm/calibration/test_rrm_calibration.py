import datetime as dt

import numpy as np
import pytest
import statista.descriptors as metrics

from hapi.calibration import Calibration
from hapi.catchment import Catchment
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped


def test_read_parameters_bounds(
    coello_rrm_date: list,
    lower_bound: list,
    upper_bound: list,
):
    Coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))
    Maxbas = True
    Snow = False
    Coello.read_parameters_bound(lower_bound, upper_bound, Snow, maxbas=Maxbas)
    assert isinstance(Coello.bounds.upper, np.ndarray)
    assert isinstance(Coello.bounds.lower, np.ndarray)
    assert isinstance(Coello.bounds.snow, bool)
    assert isinstance(Coello.bounds.maxbas, bool)


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
    Coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))
    Coello.model.read_lumped_inputs(lumped_meteo_data_path)
    Coello.model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Maxbas = True
    Coello.read_parameters_bound(lower_bound, upper_bound, coello_Snow, maxbas=Maxbas)

    parameters = []
    # Routing
    Route = 1
    routing_fn = Routing.triangular_routing_1

    basic_inputs = dict(Route=Route, RoutingFn=routing_fn, InitialValues=parameters)

    # discharge gauges
    Coello.model.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)

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
            Catchment(
                "coello",
                coello_start_date,
                coello_end_date,
                spatial_resolution="Distributed",
                temporal_resolution="Daily",
                fmt="%Y-%m-%d",
            )
        )
        assert coello.model.spatial_resolution == "distributed"
        assert coello.model.routing_method == "Muskingum"
        assert isinstance(coello.model.period.start, dt.datetime)

    def test_read_objective_fn(self, coello_start_date: str, coello_end_date: str):
        coello = Calibration(
            Catchment(
                "coello",
                coello_start_date,
                coello_end_date,
            )
        )
        coello.read_objective_function(metrics.rmse, [])
        assert coello.objective_function == metrics.rmse
        assert coello.OFArgs == []


class TestCalibrationHoldsACatchment:
    """`Calibration` composes a catchment rather than being one."""

    def test_it_is_not_a_catchment_subclass(self):
        """Test that the inheritance is gone.

        Test scenario:
            It inherited a forty-attribute builder to use a dozen fields of, and inherited
            `plot_hydrograph` -- which reads `Qsim.loc[...]` and so could never work against
            the bare array this class's own `extract_discharge` produces. Composition makes
            that impossible rather than merely fixed.
        """
        assert not issubclass(Calibration, Catchment), (
            "Calibration must hold a Catchment, not be one"
        )
        assert not hasattr(Calibration, "plot_hydrograph"), (
            "it must not inherit a plotting method its own extract_discharge would break"
        )

    def test_it_refuses_anything_but_a_catchment(self):
        """Test that the constructor takes the model it calibrates.

        Test scenario:
            The old signature mirrored `Catchment.__init__` and built one internally, so a
            caller could not calibrate a model they had already assembled -- notably one from
            `Catchment.from_yaml`.
        """
        with pytest.raises(TypeError, match="takes the Catchment it calibrates"):
            Calibration("coello")

    def test_the_bounds_live_on_the_calibration(self, coello_rrm_date: list):
        """Test that the search space belongs to the optimiser, not the model.

        Test scenario:
            `ParameterBounds` is read by nothing but a calibration, and carries the
            `(snow, maxbas)` pair every trial vector is checked against -- so keeping it on
            `Catchment` put calibration configuration on the model being calibrated.
        """
        model = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        calibration = Calibration(model)

        assert not hasattr(model, "read_parameters_bound"), (
            "read_parameters_bound must move to Calibration with the bounds it builds"
        )
        calibration.read_parameters_bound([1.0] * 12, [0.0] * 12)

        assert len(calibration.bounds) == 12, "the bounds land on the calibration"
        assert not hasattr(model, "bounds"), "and not on the model"
