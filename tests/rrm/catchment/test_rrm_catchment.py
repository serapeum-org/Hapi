import datetime as dt
import os
from typing import Dict, List, Tuple

import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from hapi.catchment import Catchment
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


def test_create_catchment_instance(coello_rrm_date: list):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    assert Coello.dt == 1
    assert isinstance(Coello.Index, DatetimeIndex)
    assert isinstance(Coello.routing_method, str)


class TestLumped:
    def test_read_lumped_meteo_inputs(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
    ):
        Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        Coello.read_lumped_inputs(lumped_meteo_data_path)
        assert isinstance(Coello.data, np.ndarray)

    def test_read_lumped_model(
        self,
        coello_rrm_date: list,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
    ):
        Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        Coello.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        assert isinstance(Coello.LumpedModel, HBVLumped)
        assert isinstance(Coello.CatArea, float)
        assert isinstance(Coello.InitialCond, list)

    def test_read_lumped_read_parameters(
        self,
        coello_rrm_date: list,
        lumped_parameters_path: str,
        coello_Snow: int,
    ):
        Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        Coello.read_parameters(lumped_parameters_path, coello_Snow)
        assert isinstance(Coello.Parameters, list)
        assert Coello.Snow == coello_Snow

    def test_read_discharge_gauges(
        self,
        coello_rrm_date: list,
        lumped_gauges_path: str,
        coello_gauges_date_fmt: str,
    ):
        Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        Coello.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
        assert isinstance(Coello.QGauges, DataFrame)

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

        assert len(coello.Qsim) == 10 and coello.Qsim.columns.to_list() == ["q"]

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
        Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        Coello.read_lumped_inputs(lumped_meteo_data_path)
        Coello.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        Coello.read_parameters(lumped_parameters_path, coello_Snow)
        # discharge gauges
        Coello.read_discharge_gauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
        Route = 1
        Run.runLumped(Coello, Route, Routing.muskingum_v)
        Coello.save_results(result=5, path=path)

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
    #     Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    #     Coello.readLumpedInputs(lumped_meteo_data_path)
    #     Coello.readLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    #     Coello.readParameters(lumped_parameters_path, coello_Snow)
    #     # discharge gauges
    #     Coello.readDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    #     RoutingFn = Routing.muskingum_v
    #     Route = 1
    #     Run.runLumped(Coello, Route, RoutingFn)
    #     assert len(Coello.Qsim) == 10 and Coello.Qsim.columns.to_list() == ["q"]


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
        coello.read_rainfall(
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_temperature(
            coello_prec_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_et(
            coello_temp_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        assert isinstance(coello.Prec, np.ndarray)
        assert isinstance(coello.Temp, np.ndarray)
        assert isinstance(coello.ET, np.ndarray)
        assert coello.Prec.shape == (13, 14, 10)
        assert coello.ET.shape == (13, 14, 10)

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
        coello.read_flow_acc(coello_acc_path)
        coello.read_flow_dir(coello_fd_path)
        assert coello.Outlet[0][0] == 10
        assert coello.Outlet[1][0] == 13
        assert coello.acc_val == coello_acc_values
        assert isinstance(coello.FlowDirArr, np.ndarray)
        assert coello.FlowDirArr.shape == (13, 14)
        assert coello.FDT == coello_fdt

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
        assert isinstance(coello.LumpedModel, HBVLumped)
        assert coello.CatArea == coello_cat_area
        assert coello.InitialCond == coello_initial_cond

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
        assert coello.Snow == Snow
        assert coello.Maxbas == False

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
        assert coello.Parameters.shape == (
            coello_rows,
            coello_cols,
            coello_no_parameters - 1,
        )
        assert coello.Snow == Snow
        assert coello.Maxbas is True


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
        coello.read_rainfall(
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_temperature(
            coello_prec_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_et(
            coello_temp_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_flow_acc(coello_acc_path)
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
        coello.read_rainfall(
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_temperature(
            coello_prec_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_et(
            coello_temp_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_flow_acc(coello_acc_path)
        # coello.readFlowDir(coello_fd_path)

        coello.read_gauge_table(coello_gauges_table, coello_acc_path)
        coello.read_discharge_gauges(coello_gauges_path, column="id", fmt="%Y-%m-%d")

        snow = False
        coello.read_parameters(coello_dist_parameters_maxbas, snow, maxbas=True)
        coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        Run.runFW1(coello)

        coello.extract_discharge(calculate_metrics=True, frame_work_1=True)
        assert isinstance(coello.Metrics, DataFrame)
        assert len(coello.Metrics) == 7
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
        coello.read_rainfall(
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_temperature(
            coello_prec_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_et(
            coello_temp_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_flow_acc(coello_acc_path)
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
        coello.read_rainfall(
            coello_evap_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_temperature(
            coello_prec_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_et(
            coello_temp_path,
            start=coello_start_date,
            end=coello_end_date,
            regex_string=r"\d{4}.\d{2}.\d{2}",
            date=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        coello.read_flow_acc(coello_acc_path)
        # coello.readFlowDir(coello_fd_path)

        coello.read_gauge_table(coello_gauges_table, coello_acc_path)
        coello.read_discharge_gauges(coello_gauges_path, column="id", fmt="%Y-%m-%d")

        Snow = False
        coello.read_parameters(coello_dist_parameters_maxbas, Snow, maxbas=True)
        coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        Run.runFW1(coello)

        coello.extract_discharge(calculate_metrics=True, frame_work_1=True)
        assert isinstance(coello.Metrics, DataFrame)
        assert len(coello.Metrics) == 7
        assert len(coello.Qsim) == 10
