# Distributed Hydrological Model Calibration
The calibration of the Distributed rainfall runoff model follows the same steps of running the model with extra steps to define the calibration algorithm arguments

## 1- Catchment Object
- Import the Catchment object which is the main object in the distributed model, to read and check the input data,  and when the model finish the simulation it stores the results and do the visualization


```python

class Catchment:

    def __init__(self, name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
                 TemporalResolution = "Daily"):
		"""
        =============================================================================
            Catchment(name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
                             TemporalResolution = "Daily")
        =============================================================================
        Parameters
        ----------
        name : [str]
            Name of the Catchment.
        StartDate : [str]
            starting date.
        EndDate : [str]
            end date.
        fmt : [str], optional
            format of the given date. The default is "%Y-%m-%d".
        SpatialResolution : TYPE, optional
            Lumped or 'Distributed' . The default is 'Lumped'.
        TemporalResolution : TYPE, optional
            "Hourly" or "Daily". The default is "Daily".
	"""
```
- To instantiate the object you need to provide the `name`, `statedate`, `enddate`, and the `SpatialResolution`

```python
from Hapi.catchment import Catchment

start = "2009-01-01"
end = "2011-12-31"
name = "Coello"

Coello = Catchment(name, start, end, SpatialResolution = "Distributed")
```

# Read Meteorological Inputs


- First define the directory where the data exist

```python
Path = Comp + "/data/distributed/coello"
PrecPath = Path + "/prec"
Evap_Path = Path + "/evap"
TempPath = Path + "/temp"
FlowAccPath = Path + "/GIS/acc4000.tif"
FlowDPath = Path + "/GIS/fd4000.tif"
ParPathRun = Path + "/Parameter set-Avg/"
```
- Then use the each method in the object to read the coresponding data

```python
Coello.readRainfall(PrecPath)
Coello.readTemperature(TempPath)
Coello.readET(Evap_Path)
Coello.readFlowAcc(FlowAccPath)
Coello.readFlowDir(FlowDPath)
```
- To read the parameters you need to provide whether you need to consider the snow subroutine or not

```python
Snow = 0
Coello.readParameters(ParPathRun, Snow)
```

## 2- Lumped Model

- Get the Lumpde conceptual model you want to couple it with the distributed routing module which in our case HBV
	and define the initial condition, and catchment area.

```python
from Hapi.hbv_bergestrom92 import HBVBergestrom92 as HBV

CatchmentArea = 1530
InitialCond = [0,5,5,5,0]
Coello.readLumpedModel(HBV, CatchmentArea, InitialCond)
```
- If the Inpus are consistent in dimensions you will get a the following message

![check_inputs](../img/check_inputs.PNG)


- to check the performance of the model we need to read the gauge hydrographs

```python
Coello.readGaugeTable("Hapi/Data/00inputs/Discharge/stations/gauges.csv", FlowAccPath)
GaugesPath = "Hapi/Data/00inputs/Discharge/stations/"
Coello.readDischargeGauges(GaugesPath, column='id', fmt="%Y-%m-%d")
```
## 3-Run Object


- The `Run` object connects all the components of the simulation together, the `Catchment` object, the `Lake` object and the `distributedrouting` object
- import the Run object and use the `Catchment` object as a parameter to the `Run` object, then call the RunHapi method to start the simulation

```python
from Hapi.run import Run
Run.RunHapi(Coello)
```
- the result of the simulation will be stored as attributes in the Catchment object as follow

```python
"""
Outputs:
    1-statevariables: [numpy attribute]
        4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
    2-qlz: [numpy attribute]
        3D array of the lower zone discharge
    3-quz: [numpy attribute]
        3D array of the upper zone discharge
    4-qout: [numpy attribute]
        1D timeseries of discharge at the outlet of the catchment
        of unit m3/sec
    5-quz_routed: [numpy attribute]
        3D array of the upper zone discharge  accumulated and
        routed at each time step
    6-qlz_translated: [numpy attribute]
        3D array of the lower zone discharge translated at each time step
"""
```
## 4-Extract Hydrographs

- The final step is to extract the simulated Hydrograph from the cells at the location of the gauges to compare
- The `extractDischarge` method extracts the hydrographs, however you have to provide in the gauge file the coordinates of the gauges with the same coordinate system of the `FlowAcc` raster

```python
Coello.extractDischarge(Factor=Coello.GaugesTable['area ratio'].tolist())

for i in range(len(Coello.GaugesTable)):
	gaugeid = Coello.GaugesTable.loc[i,'id']
	print("----------------------------------")
	print("Gauge - " +str(gaugeid))
	print("RMSE= " + str(round(Coello.Metrics.loc['RMSE',gaugeid],2)))
	print("NSE= " + str(round(Coello.Metrics.loc['NSE',gaugeid],2)))
	print("NSEhf= " + str(round(Coello.Metrics.loc['NSEhf',gaugeid],2)))
	print("KGE= " + str(round(Coello.Metrics.loc['KGE',gaugeid],2)))
	print("WB= " + str(round(Coello.Metrics.loc['WB',gaugeid],2)))
	print("Pearson CC= " + str(round(Coello.Metrics.loc['Pearson-CC',gaugeid],2)))
	print("R2 = " + str(round(Coello.Metrics.loc['R2',gaugeid],2)))
```
- The `extractDischarge` will print the performance metics


## 5-Visualization

- Firts type of visualization we can do with the results is to compare the gauge hydrograph with the simulatied hydrographs
- Call the `plotHydrograph` method and provide the period you want to visualize with the order of the gauge

```python
gaugei = 5
plotstart = "2009-01-01"
plotend = "2011-12-31"

Coello.plotHydrograph(plotstart, plotend, gaugei)
```
![hydrograph](../img/hydrograph.png)


## 6-Animation

- the best way to visualize time series of distributed data is through visualization, for theis reason, The `Catchment` object has `plotDistributedResults` method which can animate all the results of the model

```python
"""
=============================================================================
AnimateArray(Arr, Time, NoElem, TicksSpacing = 2, Figsize=(8,8), PlotNumbers=True,
       NumSize= 8, Title = 'Total Discharge',titlesize = 15, Backgroundcolorthreshold=None,
       cbarlabel = 'Discharge m3/s', cbarlabelsize = 12, textcolors=("white","black"),
       Cbarlength = 0.75, Interval = 200,cmap='coolwarm_r', Textloc=[0.1,0.2],
       Gaugecolor='red',Gaugesize=100, ColorScale = 1,gamma=1./2.,linthresh=0.0001,
       linscale=0.001, midpoint=0, orientation='vertical', rotation=-90,IDcolor = "blue",
          IDsize =10, **kwargs)
=============================================================================
Parameters
----------
Arr : [array]
    the array you want to animate.
Time : [dataframe]
    dataframe contains the date of values.
NoElem : [integer]
    Number of the cells that has values.
TicksSpacing : [integer], optional
    Spacing in the colorbar ticks. The default is 2.
Figsize : [tuple], optional
    figure size. The default is (8,8).
PlotNumbers : [bool], optional
    True to plot the values intop of each cell. The default is True.
NumSize : integer, optional
    size of the numbers plotted intop of each cells. The default is 8.
Title : [str], optional
    title of the plot. The default is 'Total Discharge'.
titlesize : [integer], optional
    title size. The default is 15.
Backgroundcolorthreshold : [float/integer], optional
    threshold value if the value of the cell is greater, the plotted
    numbers will be black and if smaller the plotted number will be white
    if None given the maxvalue/2 will be considered. The default is None.
textcolors : TYPE, optional
    Two colors to be used to plot the values i top of each cell. The default is ("white","black").
cbarlabel : str, optional
    label of the color bar. The default is 'Discharge m3/s'.
cbarlabelsize : integer, optional
    size of the color bar label. The default is 12.
Cbarlength : [float], optional
    ratio to control the height of the colorbar. The default is 0.75.
Interval : [integer], optional
    number to controlthe speed of the animation. The default is 200.
cmap : [str], optional
    color style. The default is 'coolwarm_r'.
Textloc : [list], optional
    location of the date text. The default is [0.1,0.2].
Gaugecolor : [str], optional
    color of the points. The default is 'red'.
Gaugesize : [integer], optional
    size of the points. The default is 100.
IDcolor : [str]
    the ID of the Point.The default is "blue".
IDsize : [integer]
    size of the ID text. The default is 10.
ColorScale : integer, optional
    there are 5 options to change the scale of the colors. The default is 1.
    1- ColorScale 1 is the normal scale
    2- ColorScale 2 is the power scale
    3- ColorScale 3 is the SymLogNorm scale
    4- ColorScale 4 is the PowerNorm scale
    5- ColorScale 5 is the BoundaryNorm scale
    ------------------------------------------------------------------
    gamma : [float], optional
        value needed for option 2 . The default is 1./2..
    linthresh : [float], optional
        value needed for option 3. The default is 0.0001.
    linscale : [float], optional
        value needed for option 3. The default is 0.001.
    midpoint : [float], optional
        value needed for option 5. The default is 0.
    ------------------------------------------------------------------
orientation : [string], optional
    orintation of the colorbar horizontal/vertical. The default is 'vertical'.
rotation : [number], optional
    rotation of the colorbar label. The default is -90.
**kwargs : [dict]
    keys:
        Points : [dataframe].
            dataframe contains two columns 'cell_row', and cell_col to
            plot the point at this location

Returns
-------
animation.FuncAnimation.

"""
```
- choose the period of time you want to animate and the result (total discharge, upper zone discharge, soil moisture,...)

```python
plotstart = "2009-01-01"
plotend = "2009-02-01"

Anim = Coello.plotDistributedResults(plotstart, plotend, Figsize=(9,9), Option = 1,threshold=160, PlotNumbers=True,
                            TicksSpacing = 5,Interval = 200, Gauges=True, cmap='inferno', Textloc=[0.1,0.2],
                            Gaugecolor='red',ColorScale = 1, IDcolor='blue', IDsize=25)
```
.. only:: html

   .. figure:: /img/anim.gif


- to save the animation

	- Please visit https://ffmpeg.org/ and download a version of ffmpeg compitable with your operating system
	- Copy the content of the folder and paste it in the "c:/user/.matplotlib/ffmpeg-static/"
	or

	- define the path where the downloaded folder "ffmpeg-static" exist to matplotlib using the following lines

```python
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = "path where you saved the ffmpeg.exe/ffmpeg.exe"
```
```python
Path = SaveTo + "anim.gif"
Coello.saveAnimation(VideoFormat="gif",Path=Path,SaveFrames=3)
```
## 7-Save the result into rasters

- To save the results as rasters provide the period and the path

```python
StartDate = "2009-01-01"
EndDate = "2010-04-20"
Prefix = 'Qtot_'

Coello.saveResults(FlowAccPath, Result=1, StartDate=StartDate, EndDate=EndDate, Path="F:/02Case studies/Coello/Hapi/Model/results/", Prefix=Prefix)
```