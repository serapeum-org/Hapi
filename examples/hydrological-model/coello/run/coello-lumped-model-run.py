import datetime as dt

import matplotlib

matplotlib.use("TkAgg")
import statista.descriptors as PC

from Hapi.catchment import Catchment
from Hapi.routing import Routing
from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from Hapi.run import Run

# %% data
parameter_path = "examples/hydrological-model/data/lumped_model/Coello_Lumped2021-03-08_muskingum.txt"
meteo_data_path = "examples/hydrological-model/data/lumped_model/meteo_data-MSWEP.csv"
path = "examples/hydrological-model/data/lumped_model/"
save_to = "examples/hydrological-model/data/lumped_model/"
### Meteorological data
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(name, start, end)
Coello.read_lumped_inputs(meteo_data_path)
# %% Lumped model
# catchment area
AreaCoeff = 1530
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]

Coello.read_lumped_model(HBVLumped, AreaCoeff, InitialCond)
# %% ### Model Parameters

Snow = False  # no snow subroutine
Coello.read_parameters(parameter_path, Snow)
# Coello.Parameters
# %% ### Observed flow
Coello.read_discharge_gauges(path + "Qout_c.csv", fmt="%Y-%m-%d")
# %%  ### Routing

# RoutingFn = Routing.TriangularRouting2
RoutingFn = Routing.Muskingum_V
Route = 1
# %% ### Run The Model
# Coello.Parameters = [1.0171762638840873,
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
# Coello.extractDischarge(OnlyOutlet=True)
Metrics = dict()

# gaugeid = Coello.QGauges.columns[-1]
Qobs = Coello.QGauges["q"]

Metrics["RMSE"] = PC.rmse(Qobs, Coello.Qsim["q"])
Metrics["NSE"] = PC.nse(Qobs, Coello.Qsim["q"])
Metrics["NSEhf"] = PC.nse_hf(Qobs, Coello.Qsim["q"])
Metrics["KGE"] = PC.kge(Qobs, Coello.Qsim["q"])
Metrics["WB"] = PC.wb(Qobs, Coello.Qsim["q"])

print("RMSE= " + str(round(Metrics["RMSE"], 2)))
print("NSE= " + str(round(Metrics["NSE"], 2)))
print("NSEhf= " + str(round(Metrics["NSEhf"], 2)))
print("KGE= " + str(round(Metrics["KGE"], 2)))
print("WB= " + str(round(Metrics["WB"], 2)))
# %% ### Plot Hydrograph
gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
fig, ax = Coello.plot_hydrograph(plotstart, plotend, gaugei, title="Lumped Model")
# %% ### Save Results

StartDate = "2009-01-01"
EndDate = "2010-04-20"

path = save_to + "Results-Lumped-Model_" + str(dt.datetime.now())[0:10] + ".txt"
Coello.save_results(result=5, start=StartDate, end=EndDate, path=path)
