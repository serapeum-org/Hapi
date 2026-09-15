# Distributed Hydrological Model

!!! tip "Or drive it from a YAML file"

    Everything this page assembles in Python can live in a run configuration instead --
    one file holding the paths, dates and settings, read by `Catchment.from_yaml`. See
    [Run configuration](run-configuration.md).

After preparing all the meteorological, GIS inputs required for the model, and Extracting the parameters for the catchment

```python
import numpy as np
import datetime as dt
from osgeo import gdal
from hapi.calibration import Calibration
from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.conceptual.hbv_bergestrom92 import HBVBergestrom92 as HBV

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
# [sp,sm,uz,lz,wc]
InitialCond = [0, 5, 5, 5, 0]
Snow = 0

# Create the model object and read the input data

Sdate = '2009-01-01'
Edate = '2011-12-31'
name = "Coello"
# `Calibration` holds a catchment rather than being one, so the model is built first
# and every reader is called on `Coello.model`.
Coello = Calibration(Catchment(name, Sdate, Edate, spatial_resolution="Distributed"))

# Meteorological & GIS Data
Coello.model.meteo = MeteoInputs.from_rasters(PrecPath, TempPath, Evap_Path)

Coello.model.flow_network = FlowNetwork.from_rasters(FlowAccPath, FlowDPath)

# Lumped Model
Coello.model.read_lumped_model(HBV, AreaCoeff, InitialCond)

# Gauges Data
Coello.model.read_gauge_table(Path + "/stations/gauges.csv", FlowAccPath)
GaugesPath = Path + "/stations/"
Coello.model.read_discharge_gauges(GaugesPath, column='id', fmt="%Y-%m-%d")



```
## Spatial Variability Object

- The `DistParameters` distribute the parameter vector on the cells following some spatial logic (same set of parameters for all cells, different parameters for each cell, HRU, different parameters for each class in an additional map)

```python
from hapi.calibration.distribution import Parameters as DP
from pyramids.dataset import Dataset

# A pyramids `Dataset`, not a bare GDAL handle — `Parameters.__init__` refuses anything else.
raster = Dataset.read_file(FlowAccPath)
#-------------
# for lumped catchment parameters
no_parameters = 12
klb = 0.5
kub = 1
#------------
no_lumped_par = 1
lumped_par_pos = [7]

SpatialVarFun = DP(raster, no_parameters, no_lumped_par=no_lumped_par,
                   lumped_par_pos=lumped_par_pos, function=2,
                   k_lower_bound=klb, k_upper_bound=kub)
# calculate no of parameters that optimization algorithm is going to generate
SpatialVarFun.ParametersNO


```
## Define the objective function

```python

coordinates = Coello.model.GaugesTable[['id','x','y','weight']][:]

# `run_calibration` calls the objective as `objective(QGauges, GaugesTable)` — two
# positional arguments, and nothing else. The arity is checked against the signature
# before the search starts, so a mismatch is reported rather than scored as `nan`.
def objective_function(Qobs, gauges_table):
    Coello.model.extract_discharge()
    all_errors = []
    # error for all internal stations
    for i in range(len(gauges_table)):
        all_errors.append(
            metrics.rmse(Qobs.loc[:, Qobs.columns[0]], Coello.model.Qsim.iloc[:, i])
        )
    return sum(all_errors)


# Registered outside the function — indented inside it, this line never ran.
Coello.read_objective_function(objective_function, [])
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
cal_parameters = Coello.run_calibration(SpatialVarFun, OptimizationArgs,print_error=0)

```
## Save results

```python
# `best_parameters` is the flat vector the optimiser produced, which is what `Function`
# maps onto the grid; it takes that one argument and nothing else.
SpatialVarFun.Function(Coello.best_parameters)
SpatialVarFun.save_parameters(SaveTo)
```
