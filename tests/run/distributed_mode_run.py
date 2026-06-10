"""Created on Sun Jun 24 21:02:34 2018.

@author: Mostafa
"""

Comp = "F:/01Algorithms/Hydrology/HAPI/examples"

import hapi.rrm.hbv_bergestrom92 as HBV
from hapi.catchment import Catchment
from hapi.run import Run

# %% Paths
Path = Comp + "/data/distributed/coello"
PrecPath = Path + "/prec"
Evap_Path = Path + "/evap"
TempPath = Path + "/temp"
FlowAccPath = Path + "/GIS/acc4000.tif"
FlowDPath = Path + "/GIS/fd4000.tif"

ParPathRun = Path + "/Parameter set-Avg/"
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
Coello.read_rainfall(PrecPath)
Coello.read_temperature(TempPath)
Coello.read_et(Evap_Path)

Coello.read_flow_acc(FlowAccPath)
Coello.read_flow_dir(FlowDPath)
Coello.read_parameters(ParPathRun, Snow)
Coello.read_lumped_model(HBV, AreaCoeff, InitialCond)
# %% Gauges
Coello.read_gauge_table(Path + "/stations/gauges.csv", FlowAccPath)
GaugesPath = Path + "/stations/"
Coello.read_discharge_gauges(GaugesPath, column="id", fmt="%Y-%m-%d")
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
Run.RunHapi(Coello)
# %% calculate performance criteria
Coello.extract_discharge(factor=Coello.GaugesTable["area ratio"].tolist())

for i in range(len(Coello.GaugesTable)):
    gaugeid = Coello.GaugesTable.loc[i, "id"]
    print("----------------------------------")
    print("Gauge - " + str(gaugeid))
    print("RMSE= " + str(round(Coello.Metrics.loc["RMSE", gaugeid], 2)))
    print("NSE= " + str(round(Coello.Metrics.loc["NSE", gaugeid], 2)))
    print("NSEhf= " + str(round(Coello.Metrics.loc["NSEhf", gaugeid], 2)))
    print("KGE= " + str(round(Coello.Metrics.loc["KGE", gaugeid], 2)))
    print("WB= " + str(round(Coello.Metrics.loc["WB", gaugeid], 2)))
    print("Pearson CC= " + str(round(Coello.Metrics.loc["Pearson-CC", gaugeid], 2)))
    print("R2 = " + str(round(Coello.Metrics.loc["R2", gaugeid], 2)))
# %% plot
gaugei = 5
plotstart = "2009-01-01"
plotend = "2011-12-31"

Coello.plot_hydrograph(plotstart, plotend, gaugei)
# %%
"""
Animate the distributed results.

plot_distributed_results forwards the keyword arguments to
``cleopatra.array_glyph.ArrayGlyph.animate``; see its docstring for the
full list of supported options.
"""

plotstart = "2009-01-01"
plotend = "2009-02-01"

Anim = Coello.plot_distributed_results(
    plotstart,
    plotend,
    figsize=(9, 9),
    option=1,
    background_color_threshold=160,
    display_cell_value=True,
    ticks_spacing=5,
    interval=200,
    gauges=True,
    cmap="inferno",
    text_loc=[0.1, 0.2],
    point_color="red",
    color_scale="linear",
    pid_color="blue",
    pid_size=25,
)

# %%
SaveTo = Path + "/results/anim.gif"
Coello.save_animation(SaveTo, fps=2)
# %% Save the result into rasters

StartDate = "2009-01-01"
EndDate = "2009-04-10"
Prefix = "Qtot_"
SaveTo = Path + "/results/"
Coello.save_results(
    FlowAccPath,
    result=1,
    StartDate=StartDate,
    EndDate=EndDate,
    path=SaveTo,
    prefix=Prefix,
)
