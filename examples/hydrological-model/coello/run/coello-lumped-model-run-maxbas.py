import datetime as dt

import matplotlib

matplotlib.use("TkAgg")
import statista.descriptors as metrics

from hapi.catchment import Catchment
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run

# %% data
Parameterpath = "examples/hydrological-model/data/lumped_model/coello-lumped-parameters2022-03-13-maxbas.txt"
MeteoDataPath = "examples/hydrological-model/data/lumped_model/meteo_data-MSWEP.csv"
Path = "examples/hydrological-model/data/lumped_model/"
SaveTo = "examples/hydrological-model/data/lumped_model/"
### Meteorological data
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(name, start, end)
Coello.read_lumped_inputs(MeteoDataPath)
# %% Lumped model
# catchment area
AreaCoeff = 1530
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]

Coello.read_lumped_model(HBVLumped, AreaCoeff, InitialCond)
# %% ### Model Parameters
# no snow subroutine
Snow = False
Maxbas = True
Coello.read_parameters(Parameterpath, Snow, maxbas=Maxbas)
# Coello.parameters
# %% ### Observed flow
Coello.read_discharge_gauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")
# %% Routing
# RoutingFn = Routing.triangular_routing_2
RoutingFn = Routing.triangular_routing_1
Route = 1
# %% ### Run The Model
# Coello.parameters = [1.0171762638840873,
#                      358.6427125027168,
#                      1.459834925116025,
#                      0.2031178594731058,
#                      1.0171762638840873,
#                      0.7767401680547908,
#                      0.24471700755374745,
#                      0.03648724503470574,
#                      46.41655903500876,
#                      3.126313569552141,
#                      1.9894177368962747]

Run.runLumped(Coello, Route, RoutingFn)
# %% ### Calculate performance criteria
scores = dict()

# gaugeid = Coello.QGauges.columns[-1]
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
# %% ### Plot Hydrograph
gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
fig, ax = Coello.plot_hydrograph(plotstart, plotend, gaugei, title="Lumped Model")
# %% ### Save Results

StartDate = "2009-01-01"
EndDate = "2010-04-20"

Path = f"{SaveTo}{Coello.name}Results-Lumped-Model_{str(dt.datetime.now())[0:10]}.txt"
Coello.save_results(result=5, start=StartDate, end=EndDate, path=Path)

# %%
