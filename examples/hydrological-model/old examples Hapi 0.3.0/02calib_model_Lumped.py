"""This code is developed to calibrate the HBV model in a lumped spatial representation using."""
import numpy as np
import pandas as pd

# The HBVLumped module exists in the `examples/conceptual-model`
# import Hapi.rrm.hbvlumped as HBVLumped
import statista.descriptors as PC

from Hapi.calibration import Calibration
from Hapi.routing import Routing

# %%
### meteorological data

data = pd.read_csv(
    "data/lumped/meteo_data.txt",
    header=0,
    delimiter=",",  # "\t", #skiprows=11,
    engine="python",
    index_col=0,
)
data = data.as_matrix()

### Basic_inputs
ConceptualModel = HBVLumped

p2 = [24, 1530]
init_st = [0, 5, 5, 5, 0]
snow = 0

UB = np.loadtxt("data/Basic_inputs/UB_Lumped.txt", usecols=0)
LB = np.loadtxt("data/Basic_inputs/LB_Lumped.txt", usecols=0)

# Routing
routing = 1
RoutingFn = Routing.TriangularRouting

Basic_inputs = dict(
    p2=p2,
    init_st=init_st,
    UB=UB,
    LB=LB,
    snow=snow,
    Routing=routing,
    RoutingFn=RoutingFn,
)

### Objective function
# outlet discharge
# Sdate='2009-01-01'
# Edate='2011-12-31'
Qobs = np.loadtxt(path + "Qout_c.txt")

# define the objective function and its arguments
OF_args = []
objective_function = PC.RMSE

### Optimization
store_history = 1
history_fname = "par_history.txt"
OptimizationArgs = [store_history, history_fname]
# %%
# run calibration
cal_parameters = Calibration.LumpedCalibration(
    ConceptualModel,
    data,
    Basic_inputs,
    objective_function,
    OF_args,
    Qobs,
    OptimizationArgs,
    printError=None,
)
# %% convert parameters to rasters
ParPath = "par15_7_2018.txt"
par = np.loadtxt(ParPath)
klb = 0.5
kub = 1
Path = "parameters/"

# DP.SaveParameters(SpatialVarFun, raster, par, no_parameters,snow ,kub, klb,Path)
