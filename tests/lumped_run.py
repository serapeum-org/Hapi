import datetime as dt

import statista.descriptors as metrics

from Hapi.catchment import Catchment
from Hapi.routing import Routing
from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as hbv_lumped
from Hapi.run import Run

# %% Paths
Comp = "examples"
Parameterpath = Comp + "/data/lumped/Coello_Lumped2021-03-08_muskingum.txt"
MeteoDataPath = Comp + "/data/lumped/meteo_data-MSWEP.csv"
Path = Comp + "/data/lumped/"
# %%
### meteorological data
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(name, start, end)
Coello.read_lumped_inputs(MeteoDataPath)
# %%
### Basic_inputs
# catchment area
AreaCoeff = 1530
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]

Coello.read_lumped_model(hbv_lumped, AreaCoeff, InitialCond)

### parameters
Snow = 0  # no snow subroutine
Coello.read_parameters(Parameterpath, Snow)
# %% observed flow
Coello.read_discharge_gauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")
# %% Routing
# RoutingFn = Routing.TriangularRouting2
RoutingFn = Routing.Muskingum_V
Route = 1
### run the model
Run.runLumped(Coello, Route, RoutingFn)
# %% calculate performance criteria
Metrics = dict()

Qobs = Coello.QGauges["q"]

Metrics["RMSE"] = metrics.rmse(Qobs, Coello.Qsim["q"])
Metrics["NSE"] = metrics.nse(Qobs, Coello.Qsim["q"])
Metrics["NSEhf"] = metrics.nse_hf(Qobs, Coello.Qsim["q"])
Metrics["KGE"] = metrics.kge(Qobs, Coello.Qsim["q"])
Metrics["WB"] = metrics.wb(Qobs, Coello.Qsim["q"])

print("RMSE= " + str(round(Metrics["RMSE"], 2)))
print("NSE= " + str(round(Metrics["NSE"], 2)))
print("NSEhf= " + str(round(Metrics["NSEhf"], 2)))
print("KGE= " + str(round(Metrics["KGE"], 2)))
print("WB= " + str(round(Metrics["WB"], 2)))
# %% Plot
gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
Coello.plot_hydrograph(plotstart, plotend, gaugei, title="Lumped Model")
# %% Save Results
StartDate = "2009-01-01"
EndDate = "2010-04-20"

Path = Path + "Results-Lumped-Model" + str(dt.datetime.now())[0:10] + ".txt"
Coello.save_results(result=1, StartDate=StartDate, EndDate=EndDate, path=Path)
