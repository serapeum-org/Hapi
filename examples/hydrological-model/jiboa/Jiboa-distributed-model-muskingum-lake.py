"""This code is used to Run the distributed model for jiboa river in El Salvador where the catchment has an a ustream lake and a volcanic area.

-   you have to make the root directory to the examples folder to enable the code
    from reading input files
"""

import datetime as dt
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import statista.descriptors as metrics

from Hapi.catchment import Catchment, Lake
from Hapi.rrm.hbv import HBV
from Hapi.rrm.hbv_lake import HBVLake
from Hapi.run import Run

# %%
root_dir = Path(r"examples/hydrological-model/jiboa/data")
meteo_inputs_path = root_dir / "meteo-data"
gis_inputs_path = root_dir / "gis-data"
# %% Paths
res = 4000
# paths to meteorological data
prec_path = meteo_inputs_path / f"prec"
evap_path = meteo_inputs_path / f"evap"
temp_path = meteo_inputs_path / f"temp"
# gis data
flow_acc_path = gis_inputs_path / f"acc{res}.tif"
flow_direction_path = gis_inputs_path / f"fd{res}.tif"
par_path = root_dir / f"parameters/"

# Lake
lake_meteo_path = root_dir / "lakedata.csv"
lake_parameters_path = root_dir / f"Lakeparameters.txt"
gauges_path = root_dir / "gauges"
save_to = root_dir / "results/"
# %% Distributed Model Object

catchment_area = 227.31
initial_conditions = np.loadtxt(root_dir / "initial-jiboa.txt", usecols=0).tolist()
Snow = 0

start_date = "2012-06-14 19:00:00"
# Edate = '2014-11-17 00:00:00'
end_date = "2013-12-23 00:00:00"
name = "Jiboa"
Jiboa = Catchment(
    name,
    start_date,
    end_date,
    spatial_resolution="Distributed",
    temporal_resolution="Hourly",
    fmt="%Y-%m-%d %H:%M:%S",
)
regex_exp = r"\d{4}_\d{1,2}_\d{1,2}_\d{1,2}"
date_format = "%Y_%m_%d_%H"
Jiboa.read_rainfall(
    str(prec_path), regex_string=regex_exp, file_name_data_fmt=date_format
)
Jiboa.read_temperature(
    str(temp_path), regex_string=regex_exp, file_name_data_fmt=date_format
)
Jiboa.read_et(str(evap_path), regex_string=regex_exp, file_name_data_fmt=date_format)

Jiboa.read_flow_acc(str(flow_acc_path))
Jiboa.read_flow_dir(str(flow_direction_path))
Jiboa.read_parameters(str(par_path), Snow)

Jiboa.read_lumped_model(HBV, catchment_area, initial_conditions)
# %% Lake Object
"""
lake meteorological data
"""
# where the lake discharges its flow (give the indices of the cell)
if res == 4000:
    OutflowCell = [2, 1]  # 4km
elif res == 2000:
    OutflowCell = [4, 2]  # 2km
elif res == 1000:
    OutflowCell = [10, 4]  # 1km
elif res == 500:
    OutflowCell = [19, 10]  # 500m

start_date = "2012.06.14 19:00:00"
# Edate = '2014.11.17 00:00:00'
# end_date = "2013.12.23 00:00:00"
end_date = "2012.6.15 14:00:00"
JiboaLake = Lake(
    start=start_date,
    end=end_date,
    fmt="%Y.%m.%d %H:%M:%S",
    temporal_resolution="Hourly",
    split=True,
)

JiboaLake.read_meteo_data(lake_meteo_path, fmt="%d.%m.%Y %H:%M")
JiboaLake.read_parameters(lake_parameters_path)

StageDischargeCurve = np.loadtxt(root_dir / "curve.txt")

LakeInitCond = np.loadtxt(root_dir / "Initial-lake.txt", usecols=0).tolist()

LakeCatArea = 133.98
LakeArea = 70.64
Snow = 0
JiboaLake.read_lumped_model(
    HBVLake, LakeCatArea, LakeArea, LakeInitCond, OutflowCell, StageDischargeCurve, Snow
)
# %% Gauges
Date1 = "14.06.2012 19:00"
Date2 = "23.12.2013 00:00"
Jiboa.read_gauge_table(str(gauges_path / "GaugesTable.csv"), flow_acc_path)
Jiboa.read_discharge_gauges(
    gauges_path,
    column="id",
    fmt="%d.%m.%Y %H:%M",
    split=True,
    start_date=Date1,
    end_date=Date2,
)
# %% run the model
Run.runHAPIwithLake(Jiboa, JiboaLake)
# %% calculate some metrics
Jiboa.extract_discharge(only_outlet=True)

for i in range(len(Jiboa.GaugesTable)):
    gaugeid = Jiboa.GaugesTable.loc[i, "id"]
    print("----------------------------------")
    print("Gauge - " + str(gaugeid))
    print("RMSE= " + str(round(Jiboa.Metrics.loc["RMSE", gaugeid], 2)))
    print("NSE= " + str(round(Jiboa.Metrics.loc["NSE", gaugeid], 2)))
    print("NSEhf= " + str(round(Jiboa.Metrics.loc["NSEhf", gaugeid], 2)))
    print("KGE= " + str(round(Jiboa.Metrics.loc["KGE", gaugeid], 2)))
    print("WB= " + str(round(Jiboa.Metrics.loc["WB", gaugeid], 2)))
    print("Pearson CC= " + str(round(Jiboa.Metrics.loc["Pearson-CC", gaugeid], 2)))
    print("R2 = " + str(round(Jiboa.Metrics.loc["R2", gaugeid], 2)))
# %%
Qobs = Jiboa.QGauges[Jiboa.GaugesTable.loc[0, "id"]]

gaugeid = Jiboa.GaugesTable.loc[0, "id"]

WS = {}
WS["type"] = 1
WS["N"] = 3
ModelMetrics = dict()
ModelMetrics["Calib_RMSEHF"] = round(
    metrics.rmse_hf(Qobs, Jiboa.Qsim[gaugeid], WS["type"], WS["N"], 0.75), 3
)
ModelMetrics["Calib_RMSELF"] = round(
    metrics.rmse_lf(Qobs, Jiboa.Qsim[gaugeid], WS["type"], WS["N"], 0.75), 3
)
ModelMetrics["Calib_NSEHf"] = round(metrics.nse(Qobs, Jiboa.Qsim[gaugeid]), 3)
ModelMetrics["Calib_NSELf"] = round(
    metrics.nse(np.log(Qobs), np.log(Jiboa.Qsim[gaugeid])), 3
)
ModelMetrics["Calib_RMSE"] = round(metrics.rmse(Qobs, Jiboa.Qsim[gaugeid]), 3)
ModelMetrics["Calib_KGE"] = round(metrics.kge(Qobs, Jiboa.Qsim[gaugeid]), 3)
ModelMetrics["Calib_WB"] = round(metrics.wb(Qobs, Jiboa.Qsim[gaugeid]), 3)

print(ModelMetrics)
# %% plot
gaugei = 0
plotstart = "2012-06-16"
plotend = "2013-12-23"

Jiboa.plot_hydrograph(plotstart, plotend, gaugei)
# %%
"""
Animate the distributed results.

plot_distributed_results animates the time series of the meteorological
inputs and the results calculated by the model, like the total discharge,
upper zone and lower zone discharge, and the state variables. The keyword
arguments are forwarded to ``cleopatra.array_glyph.ArrayGlyph.animate``;
see its docstring for the full list of supported options.
"""

plotstart = "2012-07-20"
plotend = "2012-08-20"

Anim = Jiboa.plot_distributed_results(
    plotstart,
    plotend,
    figsize=(8, 8),
    option=3,
    background_color_threshold=160,
    display_cell_value=False,
    ticks_spacing=10,
    interval=10,
    gauges=False,
    cmap="inferno",
    text_loc=[0.6, 0.8],
    point_color="red",
    color_scale="power",
    pid_color="blue",
    pid_size=25,
    gamma=0.08,
)
# %%
Path = save_to + "anim.mov"
Jiboa.save_animation(Path, fps=2)
# %% Save Results
start_date = "2012-07-20"
end_date = "2012-08-20"

Path = save_to + "Lumped_Parameters_" + str(dt.datetime.now())[0:10] + "_"
Jiboa.save_results(
    result=1, start=start_date, end=end_date, path=Path, flow_acc_path=flow_acc_path
)
