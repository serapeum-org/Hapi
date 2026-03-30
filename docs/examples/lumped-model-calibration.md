# Lumped Model Calibration

To calibrate the HBV lumped model inside Hapi you need to follow the same steps of running the lumped model with few extra steps to define the requirement of the calibration algorithm.


```python
	import pandas as pd
	import datetime as dt
	import Hapi.rrm.hbv_bergestrom92 as HBVLumped
	from Hapi.calibration import Calibration
	from Hapi.routing import Routing
	from Hapi.run import Run
	import statista.metrics as metrics

	Parameterpath = Comp + "/data/lumped/Coello_Lumped2021-03-08_muskingum.txt"
	MeteoDataPath = Comp + "/data/lumped/meteo_data-MSWEP.csv"
	Path = Comp + "/data/lumped/"

	start = "2009-01-01"
	end = "2011-12-31"
	name = "Coello"

	Coello = Calibration(name, start, end)
	Coello.readLumpedInputs(MeteoDataPath)


	# catchment area
	AreaCoeff = 1530
	# temporal resolution
	# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
	InitialCond = [0,10,10,10,0]
	# no snow subroutine
	Snow = 0
	Coello.readLumpedModel(HBVLumped, AreaCoeff, InitialCond)

	# Calibration boundaries
	UB = pd.read_csv(Path + "/lumped/UB-3.txt", index_col = 0, header = None)
	parnames = UB.index
	UB = UB[1].tolist()
	LB = pd.read_csv(Path + "/lumped/LB-3.txt", index_col = 0, header = None)
	LB = LB[1].tolist()

	Maxbas = True
	Coello.readParametersBounds(UB, LB, Snow, Maxbas=Maxbas)

	parameters = []
	# Routing
	Route = 1
	RoutingFn = Routing.triangular_routing_1

	Basic_inputs = dict(Route=Route, RoutingFn=RoutingFn, InitialValues = parameters)

	### Objective function
	# outlet discharge
	Coello.readDischargeGauges(Path+"Qout_c.csv", fmt="%Y-%m-%d")

	OF_args=[]
	OF=metrics.rmse

	Coello.read_objective_function(metrics.rmse, OF_args)
```
- after defining all the components of the lumped model, we have to define the calibration arguments

```python
	ApiObjArgs = dict(hms=100, hmcr=0.95, par=0.65, dbw=2000, fileout=1, xinit =0,
	                      filename=Path + "/Lumped_History"+str(dt.datetime.now())[0:10]+".txt")

	for i in range(len(ApiObjArgs)):
	    print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

	# pll_type = 'POA'
	pll_type = None

	ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=True, hot_start=False)

	OptimizationArgs=[ApiObjArgs, pll_type, ApiSolveArgs]
```

- Run Calibration

```python
	cal_parameters = Coello.lumpedCalibration(Basic_inputs, OptimizationArgs, printError=None)

	print("Objective Function = " + str(round(cal_parameters[0],2)))
	print("Parameters are " + str(cal_parameters[1]))
	print("Time = " + str(round(cal_parameters[2]['time']/60,2)) + " min")
```
