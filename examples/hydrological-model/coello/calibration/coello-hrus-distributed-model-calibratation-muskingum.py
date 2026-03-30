import os

Comp = "L:\My Drive\case studies"
os.chdir(Comp + "/Coello/HAPI/Model")
import datetime as dt

import numpy as np
from osgeo import gdal
from statista.descriptors import rmse

from Hapi.calibration import Calibration
from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBV
from Hapi.rrm.parameters import Parameters as DP

path = Comp + "/Coello/HAPI/Data/00inputs/"  # GIS/4000/
SaveTo = Comp + "/Coello/Hapi/Model/results/"
# %% Meteorological & GIS Data
PrecPath = path + "meteodata/4000/calib/prec-MSWEP"
Evap_Path = path + "meteodata/4000/calib/evap"
TempPath = path + "meteodata/4000/calib/temp"
FlowAccPath = path + "GIS/4000/acc4000.tif"
FlowDPath = path + "GIS/4000/fd4000.tif"
# %% Basic_inputs
AreaCoeff = 1530
# [sp,sm,uz,lz,wc]
InitialCond = [0, 5, 5, 5, 0]
Snow = False
# %%
"""
Create the model object and read the input data
"""
start_date = "2009-01-01"
end_date = "2011-12-31"
name = "Coello"
Coello = Calibration(name, start_date, end_date, spatial_resolution="Distributed")
Coello.read_rainfall(PrecPath)
Coello.read_temperature(TempPath)
Coello.read_et(Evap_Path)
Coello.read_flow_acc(FlowAccPath)
Coello.read_flow_dir(FlowDPath)
Coello.read_lumped_model(HBV, AreaCoeff, InitialCond)
# %%
UB = np.loadtxt(path + "/Basic_inputs/UB_HRU.txt", usecols=0)
LB = np.loadtxt(path + "/Basic_inputs/LB_HRU.txt", usecols=0)
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
# check which parameter you want to make it lumped and its position
# [rfcf, fc, beta, etf, lp, c_flux, k, k1, alpha, perc] + [k,x]
"""
"01_rfcf","02_FC", "03_BETA", "04_ETF", "05_LP", "06_K0","07_K1", "08_K2",
"09_UZL","10_PERC", + ["11_Kmuskingum", "12_Xmuskingum"]
"""
raster = gdal.Open(path + "GIS/4000/subcatch_classes.tif")
no_parameters = 12
no_lumped_par = 1
lumped_par_pos = [7]

SpatialVarFun = DP(
    raster,
    no_parameters,
    no_lumped_par=no_lumped_par,
    lumped_par_pos=lumped_par_pos,
    hru=1,
    function=4,
)

# calculate no of parameters that optimization algorithm is going to generate
print("Number of parameters = " + str(SpatialVarFun.ParametersNO))
# based on this number LB & UB text file should have this number of values
# plus klb & kub, so you have to move the lumped parameter to the end of the file
# (before klb & kub) then copy all the values and paste them as many as no of
# cells or no of HRU as you want
# this nomber is just an indication to prepare the UB & LB file don't input it to the model
# SpatialVarArgs=[raster,no_parameters,no_lumped_par,lumped_par_pos]
# %% Gauges
Coello.read_gauge_table(path + "Discharge/stations/gauges.csv", FlowAccPath)
GaugesPath = path + "Discharge/stations/"
Coello.read_discharge_gauges(GaugesPath, column="id", fmt="%Y-%m-%d")
# %% Objective function
coordinates = Coello.GaugesTable[["id", "x", "y", "weight"]][:]

# define the objective function and its arguments
OF_args = [coordinates]


def objective_function(q_obs, coordinates):  # Qout, q_uz_routed, q_lz_trans,
    Coello.extract_discharge()
    all_errors = []
    # error for all internal stations
    for i in range(len(coordinates)):
        all_errors.append((rmse(q_obs.loc[:, q_obs.columns[0]], Coello.Qsim[:, i])))
    # outlet observed discharge is at the end of the array
    # all_errors.append((PC.rmse(q_obs.loc[:,q_obs.columns[-1]],Qout))*coordinates.loc[coordinates.index[-1],'weight'])
    print(str(np.round(all_errors, 3)))
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
    filename=SaveTo + "parameters/Coello_" + str(dt.datetime.now())[0:10] + ".txt",
)

for i in range(len(ApiObjArgs)):
    print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

# pll_type = 'POA'
pll_type = None

ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=True, hot_start=False)
OptimizationArgs = [ApiObjArgs, pll_type, ApiSolveArgs]
# %% run calibration
cal_parameters = Coello.runCalibration(SpatialVarFun, OptimizationArgs, printError=0)
# %% convert parameters to rasters
SpatialVarFun.Function(Coello.Parameters, kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb)
SpatialVarFun.save_parameters(SaveTo)
