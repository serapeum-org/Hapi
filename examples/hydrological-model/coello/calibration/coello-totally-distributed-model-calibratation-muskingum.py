import datetime as dt

import numpy as np
from osgeo import gdal
from statista.descriptors import rmse

from Hapi.calibration import Calibration
from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBV
from Hapi.rrm.parameters import Parameters as DP

# %% Paths
Path = "examples/hydrological-model/data/distributed_model/"
PrecPath = Path + "/prec"
Evap_Path = Path + "/evap"
TempPath = Path + "/temp"
FlowAccPath = Path + "/GIS/acc4000.tif"
FlowDPath = Path + "/GIS/fd4000.tif"
CalibPath = Path + "/calibration"
SaveTo = Path + "/results"
# %% Calibration Object
AreaCoeff = 1530
# [sp,sm,uz,lz,wc]
InitialCond = [0, 5, 5, 5, 0]
Snow = False

"""
Create the model object and read the input data
"""
start_date = "2009-01-01"
end_date = "2009-04-10"
name = "Coello"
Coello = Calibration(name, start_date, end_date, spatial_resolution="Distributed")
# %% Meteorological & GIS Data

Coello.read_rainfall(PrecPath)
Coello.read_temperature(TempPath)
Coello.read_et(Evap_Path)

Coello.read_flow_acc(FlowAccPath)
Coello.read_flow_dir(FlowDPath)
Coello.read_lumped_model(HBV, AreaCoeff, InitialCond)
# %%
UB = np.loadtxt(CalibPath + "/UB - tot.txt", usecols=0)
LB = np.loadtxt(CalibPath + "/LB - tot.txt", usecols=0)
Coello.read_parameters_bound(UB, LB, Snow)
# %% ### spatial variability function
"""
define how generated parameters are going to be distributed spatially
totaly distributed or totally distributed with some parameters are lumped
for the whole catchment or HRUs or HRUs with some lumped parameters
for muskingum parameters k & x include the upper and lower bound in both
UB & LB with the order of Klb then kub
function inside the calibration algorithm is written as following

par_dist = SpatialVarFun(par,*SpatialVarArgs,kub=kub,klb=klb)
"""
raster = gdal.Open(FlowAccPath)
# -------------
# for lumped catchment parameters
no_parameters = 12
klb = 0.5
kub = 1
# ------------
no_lumped_par = 1
lumped_par_pos = [7]

SpatialVarFun = DP(
    raster,
    no_parameters,
    no_lumped_par=no_lumped_par,
    lumped_par_pos=lumped_par_pos,
    function=2,
    klb=klb,
    kub=kub,
)
# calculate no of parameters that optimization algorithm is going to generate
print(SpatialVarFun.ParametersNO)
# %% Gauges
Coello.read_gauge_table(Path + "/stations/gauges.csv", FlowAccPath)
GaugesPath = Path + "/stations/"
Coello.read_discharge_gauges(GaugesPath, column="id", fmt="%Y-%m-%d")
print(Coello.GaugesTable)
# %% ### Objective function

coordinates = Coello.GaugesTable[["id", "x", "y", "weight"]][:]

# define the objective function and its arguments
OF_args = [coordinates]


def objective_function(q_obs, coordinates):
    Coello.extract_discharge()
    all_errors = []
    # error for all internal stations
    for i in range(len(coordinates)):
        all_errors.append(
            (rmse(q_obs.loc[:, q_obs.columns[0]], Coello.Qsim[:, i]))
        )  # *coordinates.loc[coordinates.index[i],'weight']
    print(all_errors)
    error = sum(all_errors)
    return error


Coello.read_objective_function(objective_function, OF_args)
# %% Optimization
"""

API options
Create the options dictionary all the optimization parameters should be passed
to the optimization object inside the option dictionary:

to see all options import Optimizer class and check the documentation of the
method setOption

- for the filename please provide the full path
"""

ApiObjArgs = dict(
    hms=100,
    hmcr=0.95,
    par=0.65,
    dbw=2000,
    fileout=1,
    filename=SaveTo + "/Coello_" + str(dt.datetime.now())[0:10] + ".txt",
)

for i in range(len(ApiObjArgs)):
    print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

# pll_type = 'POA'
pll_type = None

ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=True, hot_start=False)

OptimizationArgs = [ApiObjArgs, pll_type, ApiSolveArgs]
# %% ### Run Calibration
cal_parameters = Coello.runCalibration(SpatialVarFun, OptimizationArgs, printError=1)
# %%
SpatialVarFun.Function(Coello.Parameters, kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb)
SpatialVarFun.save_parameters(SaveTo)
