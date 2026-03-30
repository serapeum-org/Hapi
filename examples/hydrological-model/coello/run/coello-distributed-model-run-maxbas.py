"""Distributed model with a maxbas routing scheme."""

import datetime as dt

import pandas as pd
from osgeo import gdal

from Hapi.catchment import Catchment
from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBV
from Hapi.run import Run

# %% Paths
Path = "examples/hydrological-model/data/distributed_model"
PrecPath = f"{Path}/prec"
Evap_Path = f"{Path}/evap"
TempPath = f"{Path}/temp"
FlowAccPath = f"{Path}/GIS/acc4000.tif"
FlowDPath = f"{Path}/GIS/fd4000.tif"
ParPath = f"{Path}/parameters_initial_maxbas"
# %% Meteorological data
AreaCoeff = 1530
InitialCond = [0, 5, 5, 5, 0]
Snow = 0
"""
Create the model object and read the input data
"""
start = "2009-01-01"
end = "2009-04-10"
name = "Coello"
Coello = Catchment(name, start, end, spatial_resolution="Distributed")
Coello.read_rainfall(PrecPath, file_name_data_fmt = "%Y.%m.%d")
Coello.read_temperature(TempPath, file_name_data_fmt = "%Y.%m.%d")
Coello.read_et(Evap_Path, file_name_data_fmt = "%Y.%m.%d")

Coello.read_flow_acc(FlowAccPath)
Coello.read_parameters(ParPath, Snow, maxbas=True)
Coello.read_lumped_model(HBV, AreaCoeff, InitialCond)
# %% Gauges
Coello.read_gauge_table(f"{Path}/stations/gauges.csv", FlowAccPath)
Coello.read_discharge_gauges(f"{Path}/stations/", column="id", fmt="%Y-%m-%d")
# %% Run the model
"""
Outputs:
    ----------
    1-state_variables: [numpy attribute]
        4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
    2-qlz: [numpy attribute]
        3D array of the lower zone discharge
    3-quz: [numpy attribute]
        3D array of the upper zone discharge
    4-qout: [numpy attribute]
        1D timeseries of discharge at the outlet of the catchment
        of unit m3/sec
    5-quz_routed: [numpy attribute]
        3D array of the upper zone discharge  accumulated and
        routed at each time step
    6-qlz_translated: [numpy attribute]
        3D array of the lower zone discharge translated at each time step
"""
Run.runFW1(Coello)
# %% calculate performance criteria
Coello.extract_discharge(calculate_metrics=True, frame_work_1=True)

gaugeid = Coello.GaugesTable.loc[Coello.GaugesTable.index[-1], "id"]
print("----------------------------------")
print("Gauge - " + str(gaugeid))
print("RMSE= " + str(round(Coello.Metrics.loc["RMSE", gaugeid], 2)))
print("NSE= " + str(round(Coello.Metrics.loc["NSE", gaugeid], 2)))
print("NSEhf= " + str(round(Coello.Metrics.loc["NSEhf", gaugeid], 2)))
print("KGE= " + str(round(Coello.Metrics.loc["KGE", gaugeid], 2)))
print("WB= " + str(round(Coello.Metrics.loc["WB", gaugeid], 2)))
# %% plot
i = 5
gaugei = 5
plotstart = "2009-01-01"
plotend = "2011-12-31"

Coello.plot_hydrograph(plotstart, plotend, gaugei)
# %% store the result into rasters
# create list of names
src = gdal.Open(FlowAccPath)
s = dt.datetime(2012, 6, 14, 19, 00, 00)
e = dt.datetime(2013, 12, 23, 00, 00, 00)
index = pd.date_range(s, e, freq="1H")
resultspath = "results/"
names = [resultspath + str(i)[:-6] for i in index]
names = [i.replace("-", "_") for i in names]
names = [i.replace(" ", "_") for i in names]
names = [i + ".tif" for i in names]

# Raster.RastersLike(src,q_uz_routed[:,:,:-1],names)
