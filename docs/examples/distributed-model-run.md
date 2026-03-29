# Distributed Hydrological Model
After preparing all the meteorological, GIS inputs required for the model, and Extracting the parameters for the catchment

```python
import numpy as np
import datetime as dt
from osgeo import gdal
from Hapi.calibration import Calibration
import Hapi.rrm.hbv_bergestrom92 as HBV

import statista.descriptors as metrics

Path = Comp + "/data/distributed/coello"
PrecPath = Path + "/prec"
Evap_Path = Path + "/evap"
TempPath = Path + "/temp"
FlowAccPath = Path + "/GIS/acc4000.tif"
FlowDPath = Path + "/GIS/fd4000.tif"
CalibPath = Path + "/calibration"
SaveTo = Path + "/results"

AreaCoeff = 1530
#[sp,sm,uz,lz,wc]
InitialCond = [0,5,5,5,0]
Snow = 0

# Create the model object and read the input data

Sdate = '2009-01-01'
Edate = '2011-12-31'
name = "Coello"
Coello = Calibration(name, Sdate, Edate, SpatialResolution = "Distributed")

# Meteorological & GIS Data
Coello.readRainfall(PrecPath)
Coello.readTemperature(TempPath)
Coello.readET(Evap_Path)

Coello.readFlowAcc(FlowAccPath)
Coello.readFlowDir(FlowDPath)

# Lumped Model
Coello.readLumpedModel(HBV, AreaCoeff, InitialCond)

# Gauges Data
Coello.readGaugeTable(Path+"/stations/gauges.csv", FlowAccPath)
GaugesPath = Path+"/stations/"
Coello.readDischargeGauges(GaugesPath, column='id', fmt="%Y-%m-%d")



```
## Spatial Variability Object

- The `DistParameters` distribute the parameter vector on the cells following some spatial logic (same set of parameters for all cells, different parameters for each cell, HRU, different parameters for each class in an additional map)

```python
from Hapi.rrm.distparameters import DistParameters as DP

raster = gdal.Open(FlowAccPath)
#-------------
# for lumped catchment parameters
no_parameters = 12
klb = 0.5
kub = 1
#------------
no_lumped_par = 1
lumped_par_pos = [7]

SpatialVarFun = DP(raster, no_parameters, no_lumped_par=no_lumped_par,
                   lumped_par_pos=lumped_par_pos,Function=2, Klb=klb, Kub=kub)
# calculate no of parameters that optimization algorithm is going to generate
SpatialVarFun.ParametersNO


```
## Define the objective function

```python

coordinates = Coello.GaugesTable[['id','x','y','weight']][:]

# define the objective function and its arguments
OF_args = [coordinates]

def objective_function(Qobs, Qout, q_uz_routed, q_lz_trans, coordinates):
    Coello.extractDischarge()
    all_errors=[]
    # error for all internal stations
    for i in range(len(coordinates)):
        all_errors.append((metrics.rmse(Qobs.loc[:,Qobs.columns[0]],Coello.Qsim[:,i]))) #*coordinates.loc[coordinates.index[i],'weight']
        print(all_errors)
        error = sum(all_errors)
        return error

    Coello.read_objective_function(objective_function, OF_args)
```
## Calibration algorithm Arguments

- Create the options dictionary all the optimization parameters should be passed to the optimization object inside the option dictionary:

to see all options import Optimizer class and check the documentation of the
method setOption

```pythonthon

ApiObjArgs = dict(hms=50, hmcr=0.95, par=0.65, dbw=2000, fileout=1,
                  filename=SaveTo + "/Coello_"+str(dt.datetime.now())[0:10]+".txt")

for i in range(len(ApiObjArgs)):
    print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

pll_type = 'POA'
pll_type = None

ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=True,hot_start=False)

OptimizationArgs=[ApiObjArgs, pll_type, ApiSolveArgs]

```
## Run Calibration algorithm

```python
cal_parameters = Coello.runCalibration(SpatialVarFun, OptimizationArgs,printError=0)

```
## Save results

```python
SpatialVarFun.Function(Coello.Parameters, kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb)
SpatialVarFun.saveParameters(SaveTo)
```