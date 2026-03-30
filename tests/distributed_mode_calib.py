"""Created on Sun Jun 24 21:02:34 2018.

@author: Mostafa
"""

Comp = "F:/01Algorithms/Hydrology/HAPI/examples"

import datetime as dt

import numpy as np
import statista.descriptors as metrics
from osgeo import gdal

from Hapi.calibration import Calibration
from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBV
from Hapi.rrm.parameters import Parameters as DP

# %% Paths
Path = Comp + "/data/distributed/coello"
PrecPath = Path + "/prec"
Evap_Path = Path + "/evap"
TempPath = Path + "/temp"
FlowAccPath = Path + "/GIS/acc4000.tif"
FlowDPath = Path + "/GIS/fd4000.tif"
CalibPath = Path + "/calibration"
SaveTo = Path + "/results"
# %% Basic_inputs
AreaCoeff = 1530
# [sp,sm,uz,lz,wc]
InitialCond = [0, 5, 5, 5, 0]
Snow = 0

"""
Create the model object and read the input data
"""
Sdate = "2009-01-01"
Edate = "2011-12-31"
name = "Coello"
Coello = Calibration(name, Sdate, Edate, spatial_resolution="Distributed")
### Meteorological & GIS Data
Coello.read_rainfall(PrecPath)
Coello.read_temperature(TempPath)
Coello.read_et(Evap_Path)

Coello.read_flow_acc(FlowAccPath)
Coello.read_flow_dir(FlowDPath)
Coello.read_lumped_model(HBV, AreaCoeff, InitialCond)


UB = np.loadtxt(CalibPath + "/UB - tot.txt", usecols=0)
LB = np.loadtxt(CalibPath + "/LB - tot.txt", usecols=0)
Coello.read_parameters_bound(UB, LB, Snow)
# %% spatial variability function
"""
define how generated parameters are going to be distributed spatially
totaly distributed or totally distributed with some parameters are lumped
for the whole catchment or HRUs or HRUs with some lumped parameters
for muskingum parameters k & x include the upper and lower bound in both
UB & LB with the order of Klb then kub
function inside the calibration algorithm is written as following
par_dist=SpatialVarFun(par,*SpatialVarArgs,kub=kub,klb=klb)

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
SpatialVarFun.ParametersNO
# %% Gauges
Coello.read_gauge_table(Path + "/stations/gauges.csv", FlowAccPath)
GaugesPath = Path + "/stations/"
Coello.read_discharge_gauges(GaugesPath, column="id", fmt="%Y-%m-%d")
# %% Objective function
coordinates = Coello.GaugesTable[["id", "x", "y", "weight"]][:]

# define the objective function and its arguments
OF_args = [coordinates]


def objective_function(Qobs, Qout, q_uz_routed, q_lz_trans, coordinates):
    Coello.extract_discharge()
    all_errors = []
    # error for all internal stations
    for i in range(len(coordinates)):
        all_errors.append(
            (metrics.rmse(Qobs.loc[:, Qobs.columns[0]], Coello.Qsim[:, i]))
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

"""
ApiObjArgs = dict(
    hms=50,
    hmcr=0.95,
    par=0.65,
    dbw=2000,
    fileout=1,
    filename=SaveTo + "/Coello_" + str(dt.datetime.now())[0:10] + ".txt",
)

for i in range(len(ApiObjArgs)):
    print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

pll_type = "POA"
pll_type = None

ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=True, hot_start=False)

OptimizationArgs = [ApiObjArgs, pll_type, ApiSolveArgs]
# %% run calibration
cal_parameters = Coello.runCalibration(SpatialVarFun, OptimizationArgs, printError=0)
# %% convert parameters to rasters
# Coello.Parameters = [0.700, 399, 1.704, 0.1021, 0.4622, 0.6237, 0.1251, 0.005, 59.85, 5.241, 94.91, 0.2075]
SpatialVarFun.Function(Coello.Parameters, kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb)
SpatialVarFun.save_parameters(SaveTo)
