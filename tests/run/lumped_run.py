import datetime as dt

import statista.descriptors as metrics

from hapi.catchment import Catchment
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as hbv_lumped
from hapi.run import Run

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
### basic_inputs
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
# routing_fn = Routing.triangular_routing_2
routing_fn = Routing.muskingum_v
Route = 1
### run the model
Run.run_lumped(Coello, Route, routing_fn)
# %% calculate performance criteria
scores = dict()

Qobs = Coello.QGauges["q"]

scores["RMSE"] = metrics.rmse(Qobs, Coello.Qsim["q"])
scores["NSE"] = metrics.nse(Qobs, Coello.Qsim["q"])
scores["NSEhf"] = metrics.nse_hf(Qobs, Coello.Qsim["q"])
scores["KGE"] = metrics.kge(Qobs, Coello.Qsim["q"])
scores["WB"] = metrics.wb(Qobs, Coello.Qsim["q"])

print("RMSE= " + str(round(scores["RMSE"], 2)))
print("NSE= " + str(round(scores["NSE"], 2)))
print("NSEhf= " + str(round(scores["NSEhf"], 2)))
print("KGE= " + str(round(scores["KGE"], 2)))
print("WB= " + str(round(scores["WB"], 2)))
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
