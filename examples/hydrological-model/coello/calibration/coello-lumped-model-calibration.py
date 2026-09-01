# Lumped Model Calibration
# - Please change the Path in the following cell to the directory where you stored the case study data
### Modules
import datetime as dt

import numpy as np
import pandas as pd
import statista.descriptors as metrics

from hapi.calibration import Calibration
from hapi.routing import Routing
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run

# %% Paths
Parameterpath = "examples/hydrological-model/data/lumped_model/Coello_Lumped2021-03-08_muskingum.txt"
MeteoDataPath = "examples/hydrological-model/data/lumped_model/meteo_data-MSWEP.csv"
Path = "examples/hydrological-model/data/lumped_model/"

start = "2009-01-01"
end = "2011-12-31"
name = "Coello"

Coello = Calibration(name, start, end)
Coello.read_lumped_inputs(MeteoDataPath)
# %% Basic_inputs

# catchment area
AreaCoeff = 1530
# temporal resolution
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]
# no snow subroutine
Snow = False
Coello.read_lumped_model(HBVLumped, AreaCoeff, InitialCond)

# Calibration parameters

# Calibration boundaries
UB = pd.read_csv(Path + "/UB-3.txt", index_col=0, header=None)
parnames = UB.index
UB = UB[1].tolist()
LB = pd.read_csv(Path + "/LB-3.txt", index_col=0, header=None)
LB = LB[1].tolist()

Maxbas = True
Coello.read_parameters_bound(UB, LB, Snow, maxbas=Maxbas)

### Additional arguments

parameters = []
# Routing
Route = 1
RoutingFn = Routing.triangular_routing_1

Basic_inputs = dict(Route=Route, RoutingFn=RoutingFn, InitialValues=parameters)

### Objective function

# outlet discharge
Coello.read_discharge_gauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")

OF_args = []
objective_function = metrics.rmse

Coello.read_objective_function(metrics.rmse, OF_args)

# Calibration

# API options
# Create the options dictionary all the optimization parameters should be passed
# to the optimization object inside the option dictionary:


# to see all options import Optimizer class and check the documentation of the
# method setOption

ApiObjArgs = dict(
    hms=100,
    hmcr=0.95,
    par=0.65,
    dbw=2000,
    fileout=1,
    xinit=0,
    filename=Path + "/Lumped_History" + str(dt.datetime.now())[0:10] + ".txt",
)

for i in range(len(ApiObjArgs)):
    print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

# pll_type = 'POA'
pll_type = None

ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=False, hot_start=False)

OptimizationArgs = [ApiObjArgs, pll_type, ApiSolveArgs]

# %% Run Calibration

cal_parameters = Coello.calibrate_lumped(
    Basic_inputs, OptimizationArgs, print_error=None
)

print("Objective Function = " + str(round(cal_parameters[0], 2)))
print("Parameters are " + str(cal_parameters[1]))
print("Time = " + str(round(cal_parameters[2]["time"] / 60, 2)) + " min")
# %% Run the Model

Coello.parameters = cal_parameters[1]
Run.run_lumped(Coello, Route, RoutingFn)

### Calculate Performance Criteria

scores = dict()

Qobs = Coello.QGauges[Coello.QGauges.columns[0]]

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

### Plotting Hydrograph

gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
Coello.plot_hydrograph(plotstart, plotend, gaugei, title="Lumped Model")

### Save the Parameters

ParPath = Path + "Parameters" + str(dt.datetime.now())[0:10] + ".txt"
parameters = pd.DataFrame(index=parnames)
parameters["values"] = cal_parameters[1]
parameters.to_csv(ParPath, header=None, float_format="%0.4f")

### Save Results

StartDate = "2009-01-01"
EndDate = "2010-04-20"

Path = Path + "Results-Lumped-Model" + str(dt.datetime.now())[0:10] + ".txt"
Coello.save_results(result=5, start=StartDate, end=EndDate, path=Path)
