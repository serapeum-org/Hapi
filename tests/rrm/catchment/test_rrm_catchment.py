import datetime as dt
import os
from typing import Dict, List, Tuple

import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


def test_create_catchment_instance(coello_rrm_date: list):
    coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    assert coello.dt == 1
    assert isinstance(coello.date_index, DatetimeIndex)
    assert isinstance(coello.routing_method, str)


class TestLumped:
    def test_read_lumped_meteo_inputs(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
    ):
        coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_lumped_inputs(lumped_meteo_data_path)
        assert isinstance(coello.data, np.ndarray)

    def test_read_lumped_model(
        self,
        coello_rrm_date: list,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
    ):
        coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        assert isinstance(coello.lumped_model, HBVLumped)
        assert isinstance(coello.area, float)
        assert isinstance(coello.initial_cond, list)

    def test_read_lumped_read_parameters(
        self,
        coello_rrm_date: list,
        lumped_parameters_path: str,
        coello_Snow: int,
    ):
        coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_parameters(lumped_parameters_path, coello_Snow)
        assert isinstance(coello.parameters, list)
        assert coello.snow == coello_Snow

    def test_read_discharge_gauges(
        self,
        coello_rrm_date: list,
        lumped_gauges_path: str,
        coello_gauges_date_fmt: str,
    ):
        coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
        assert isinstance(coello.QGauges, DataFrame)

    def test_run_lumped(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
        lumped_parameters_path: str,
        coello_Snow: int,
        lumped_gauges_path: str,
        coello_gauges_date_fmt: str,
    ):
        coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_lumped_inputs(lumped_meteo_data_path)
        coello.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        coello.read_parameters(lumped_parameters_path, coello_Snow)
        # discharge gauges
        coello.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
        routing_fn = Routing.muskingum_v
        route = 1
        Run.runLumped(coello, route, routing_fn)

        assert len(coello.Qsim) == 10
        assert coello.Qsim.columns.to_list() == ["q"]

    def test_save_lumped_results(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
        lumped_parameters_path: str,
        coello_Snow: int,
        lumped_gauges_path: str,
        coello_gauges_date_fmt: str,
    ):
        path = "tests/rrm/data/test-Lumped-Model_results.txt"
        if os.path.exists(path):
            os.remove(path)
        coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        coello.read_lumped_inputs(lumped_meteo_data_path)
        coello.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        coello.read_parameters(lumped_parameters_path, coello_Snow)
        # discharge gauges
        coello.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
        Route = 1
        Run.runLumped(coello, Route, Routing.muskingum_v)
        coello.save_results(result=5, path=path)

    # # TODO: still not finished as it does not run the plotHydrograph method
    # def test_PlotHydrograph(
    #         self,
    #         coello_rrm_date: list,
    #         lumped_meteo_data_path: str,
    #         coello_AreaCoeff: float,
    #         coello_InitialCond: list,
    #         lumped_parameters_path: str,
    #         coello_Snow: int,
    #         lumped_gauges_path: str,
    #         coello_gauges_date_fmt: str,
    # ):
    #     coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    #     coello.readLumpedInputs(lumped_meteo_data_path)
    #     coello.readLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    #     coello.read_parameters(lumped_parameters_path, coello_Snow)
    #     # discharge gauges
    #     coello.readDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    #     RoutingFn = Routing.muskingum_v
    #     Route = 1
    #     Run.runLumped(coello, Route, RoutingFn)
    #     assert len(coello.Qsim) == 10 and coello.Qsim.columns.to_list() == ["q"]


class TestDistributed:
    def test_create_catchment_instance(
        self, coello_start_date: str, coello_end_date: str
    ):
        coello = Catchment(
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

    def test_read_meteo_inputs(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_evap_path: str,
        coello_prec_path: str,
        coello_temp_path: str,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.meteo = MeteoInputs.from_rasters(
            coello_prec_path,
            coello_temp_path,
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        assert isinstance(coello.meteo.precipitation, np.ndarray)
        assert isinstance(coello.meteo.temperature, np.ndarray)
        assert isinstance(coello.meteo.evapotranspiration, np.ndarray)
        assert coello.meteo.shape == (13, 14, 10)

    def test_read_gis_inputs(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_fd_path: str,
        coello_acc_path: str,
        coello_fdt: Dict,
        coello_acc_values: List,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.flow_network = FlowNetwork.from_rasters(coello_acc_path, coello_fd_path)
        assert coello.flow_network.outlet[0][0] == 10
        assert coello.flow_network.outlet[1][0] == 13
        assert coello.flow_network.acc_val == coello_acc_values
        assert isinstance(coello.flow_network.flow_dir_arr, np.ndarray)
        assert coello.flow_network.flow_dir_arr.shape == (13, 14)
        assert coello.flow_network.FDT == coello_fdt

    def test_read_lumped_model(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_cat_area: int,
        coello_initial_cond: List,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        assert isinstance(coello.lumped_model, HBVLumped)
        assert coello.area == coello_cat_area
        assert coello.initial_cond == coello_initial_cond

    def test_read_parameters_bound(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_parameter_bounds: Tuple[List, List],
    ):
        LB = coello_parameter_bounds[0]
        UB = coello_parameter_bounds[1]
        Snow = False
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.read_parameters_bound(UB, LB, Snow)
        assert all(coello.LB == LB)
        assert all(coello.UB == UB)
        assert coello.snow == Snow
        assert coello.maxbas == False

    def test_read_gauge_table(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_acc_path: str,
        coello_gauges_table: str,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.read_gauge_table(coello_gauges_table, coello_acc_path)
        assert isinstance(coello.GaugesTable, DataFrame)
        assert all(
            elem in coello.GaugesTable.columns for elem in ["cell_row", "cell_col"]
        )

    def test_read_gauge(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_acc_path: str,
        coello_gauges_table: str,
        coello_gauges_path: str,
        coello_gauge_names: List,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.read_gauge_table(coello_gauges_table, coello_acc_path)
        coello.read_discharge_gauges(coello_gauges_path, column="id", fmt="%Y-%m-%d")
        assert isinstance(coello.QGauges, DataFrame)
        assert all(elem in coello.QGauges.columns for elem in coello_gauge_names)

    def test_read_parameters_maxbas(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_dist_parameters_maxbas: str,
        coello_rows: int,
        coello_cols: int,
        coello_no_parameters: int,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        Snow = False
        coello.read_parameters(coello_dist_parameters_maxbas, Snow, maxbas=True)
        assert coello.parameters.shape == (
            coello_rows,
            coello_cols,
            coello_no_parameters - 1,
        )
        assert coello.snow == Snow
        assert coello.maxbas is True


class TestFW1:
    def test_run_dist(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_evap_path: str,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_fd_path: str,
        coello_acc_path: str,
        coello_cat_area: int,
        coello_initial_cond: List,
        coello_dist_parameters_maxbas: str,
        coello_shape: Tuple,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.meteo = MeteoInputs.from_rasters(
            coello_prec_path,
            coello_temp_path,
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
        # coello.readFlowDir(coello_fd_path)
        coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
        coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        Run.runFW1(coello)
        assert isinstance(coello.qout, np.ndarray)
        assert len(coello.qout) == 10
        assert coello.state_variables.shape == (coello_shape[0], coello_shape[1], 11, 5)
        assert coello.quz.shape == (coello_shape[0], coello_shape[1], 11)
        assert coello.qlz.shape == (coello_shape[0], coello_shape[1], 11)

    def test_extract_results(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_evap_path: str,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_fd_path: str,
        coello_acc_path: str,
        coello_cat_area: int,
        coello_initial_cond: List,
        coello_dist_parameters_maxbas: str,
        coello_shape: Tuple,
        coello_gauges_table: str,
        coello_gauges_path: str,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.meteo = MeteoInputs.from_rasters(
            coello_prec_path,
            coello_temp_path,
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
        # coello.readFlowDir(coello_fd_path)

        coello.read_gauge_table(coello_gauges_table, coello_acc_path)
        coello.read_discharge_gauges(coello_gauges_path, column="id", fmt="%Y-%m-%d")

        snow = False
        coello.read_parameters(coello_dist_parameters_maxbas, snow, maxbas=True)
        coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        Run.runFW1(coello)

        coello.extract_discharge(calculate_metrics=True, frame_work_1=True)
        assert isinstance(coello.metrics, DataFrame)
        assert len(coello.metrics) == 7
        assert len(coello.Qsim) == 10


class TestMuskingum:
    def test_run_dist(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_evap_path: str,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_fd_path: str,
        coello_acc_path: str,
        coello_cat_area: int,
        coello_initial_cond: List,
        coello_dist_parameters_maxbas: str,
        coello_shape: Tuple,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.meteo = MeteoInputs.from_rasters(
            coello_prec_path,
            coello_temp_path,
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
        # coello.readFlowDir(coello_fd_path)
        Snow = False
        coello.read_parameters(coello_dist_parameters_maxbas, Snow, maxbas=True)
        coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        Run.runFW1(coello)
        assert isinstance(coello.qout, np.ndarray)
        assert len(coello.qout) == 10
        assert coello.state_variables.shape == (coello_shape[0], coello_shape[1], 11, 5)
        assert coello.quz.shape == (coello_shape[0], coello_shape[1], 11)
        assert coello.qlz.shape == (coello_shape[0], coello_shape[1], 11)

    def test_extract_results(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_evap_path: str,
        coello_prec_path: str,
        coello_temp_path: str,
        coello_fd_path: str,
        coello_acc_path: str,
        coello_cat_area: int,
        coello_initial_cond: List,
        coello_dist_parameters_maxbas: str,
        coello_shape: Tuple,
        coello_gauges_table: str,
        coello_gauges_path: str,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            spatial_resolution="Distributed",
            temporal_resolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.meteo = MeteoInputs.from_rasters(
            coello_prec_path,
            coello_temp_path,
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
        # coello.readFlowDir(coello_fd_path)

        coello.read_gauge_table(coello_gauges_table, coello_acc_path)
        coello.read_discharge_gauges(coello_gauges_path, column="id", fmt="%Y-%m-%d")

        Snow = False
        coello.read_parameters(coello_dist_parameters_maxbas, Snow, maxbas=True)
        coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        Run.runFW1(coello)

        coello.extract_discharge(calculate_metrics=True, frame_work_1=True)
        assert isinstance(coello.metrics, DataFrame)
        assert len(coello.metrics) == 7
        assert len(coello.Qsim) == 10
