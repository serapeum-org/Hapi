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


@pytest.mark.parametrize("width", [12, 243, 980])
def test_read_parameters_bounds_accepts_any_search_width(
    coello_rrm_date: list, width: int
):
    """Test that the bounds are not held to the conceptual model's parameter count.

    Args:
        coello_rrm_date: Start and end dates for the model.
        width: Number of bound values supplied.

    Test scenario:
        `ParameterBounds` delimits the *optimiser's* flat search vector, whose length is the
        spatial distribution's `ParametersNO` -- 980 for a totally distributed run on the
        Coello grid and 243 for the HRU one, against 12 for a lumped one. Holding it to
        `PARAMETER_COUNTS` made every distributed calibration raise at
        `read_parameters_bound`, and no width satisfied both that rule and `par3d`'s.
    """
    coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))

    coello.read_parameters_bound([0.0] * width, [1.0] * width, False)

    assert len(coello.bounds) == width, (
        f"the search space is {width} wide; got {len(coello.bounds)}"
    )


def test_read_parameters_bounds_still_refuses_mismatched_lengths(
    coello_rrm_date: list,
):
    """Test that a lower and upper bound of different lengths are still refused.

    Args:
        coello_rrm_date: Start and end dates for the model.

    Test scenario:
        The two are read from separate files and the optimiser samples between them per
        position, so this rule holds whatever the search width is -- it is the one thing
        `ParameterBounds` can check without knowing how the vector is mapped onto the grid.
    """
    coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))

    with pytest.raises(ValueError, match="same as LB"):
        coello.read_parameters_bound([0.0] * 12, [1.0] * 11, False)


@pytest.mark.parametrize("width", [10, 16])
def test_a_lumped_calibration_checks_the_search_width_before_it_starts(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
    width: int,
):
    """Test that the lumped path holds the search vector to the model's parameter count.

    Args:
        coello_rrm_date: Start and end dates for the model.
        lumped_meteo_data_path: Driver record.
        coello_AreaCoeff: Catchment area.
        coello_InitialCond: Initial state.
        width: A search width the conceptual model cannot read.

    Test scenario:
        A lumped calibration is the one case where the optimiser's vector *is* the parameter
        set, so a mismatch there is a real error -- and it used to surface once per trial
        from inside the objective, after the optimiser had started. Checked before the
        problem is declared, where the caller can act on it.
    """
    coello = Calibration(Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1]))
    coello.model.read_lumped_inputs(lumped_meteo_data_path)
    coello.model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    coello.read_parameters_bound([0.0] * width, [1.0] * width, False)
    coello.read_objective_function(metrics.rmse, [])

    with pytest.raises(ValueError, match="takes 12 parameters"):
        coello.calibrate_lumped(
            dict(Route=0, RoutingFn=None), [{}, None, {}], print_error=None
        )


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
