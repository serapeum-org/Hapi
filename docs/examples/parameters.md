# Parameters
the 15 free parameters used by the HBV model for each cell can be obtained through calibration, However
it it an extremely difficult task to keep the model simple and minimize uncertainty with such large number of parameters, Therefore in Hapi 1.0.0 we integrate the global hydrological parameters obtained by Beck et al., (2016), to reduce model complexity and uncertainty of parameters and to help obtain the best set of parameter during the calibration.

Based on Beck et al., (2016) there is 10 sets of global parameters which are derived for catchment with good quality of data and are calibratedvery well calibrated, afterwards golbal parameters are derived by transfering parameters to from the good calibrated catchments to global grid of 0.5 degree cells where each cell receives the parameters of the best 10 catchments that have similar climatic and physiographic characteristics


### Extract Distributed Parameters

The only input we need to extract parameters to our catchment is the DEM or any raster that has the full extent and allignment of the catchment and use the `Inputs` object to select the set of parameters we want

- import the class from the inputs module
```python
from Hapi.inputs import Inputs
```
- define the paths to the DEM and the directory to save the parameters

```python
dem_path = "../../data/GIS/Hapi_GIS_Data/acc4000.tif"
outputpath = "../../data/parameters/03/"
```
- call the `extractParameters` method

```python
Inputs.extractParameters(dem_path, '03', AsRaster=True, SaveTo=outputpath)
```


# Extract Calibration boundaries for the Parameters

You will find the following example in the `ExtractParametersBounds.py` file under the folder `/Examples/Create Inputs`. There is no need for copy paste work.

To Extract the parameters range needed for the Calibration you have to prepare a shapefile of the catchment you are developing a distributed model and read it using `geopandas`,

```python
import geopandas as gpd
import numpy as np
import pandas as pd
import Hapi.inputs as IN

BasinF = "Path to shapefile"
Basin = gpd.read_file(BasinF)
# parameters name with the same order inside the Input module
ind = ["tt","sfcf","cfmax","cwh","cfr","fc","beta","lp","k0","k1","k2","uzl","perc","maxbas"]
Par = pd.DataFrame(index = ind)
```

the `inputs` module in Hapi has a `extractParametersBoundaries` method to overlay the basin shapefile with the global parameters rasters and extract the max and min parameter values within the basin and plots your basin shapefile in top of the world map to make sure of the projection transformation from whatever projection your basin shapefile to the `WGS64` that the parameters rasters have

```python
# extract parameters boundaries
Par['UB'], Par['LB'] = IN.extractParametersBoundaries(Basin)
```
To extract the parameters from one of the ten scenarios developed to derive the Global model `extractParameters` method takes the number of the scenario as a string and return the parameters

```python
# extract parameters in a specific scenarion from the 10 scenarios
Par['1'] = IN.extractParameters(Basin,"01")
```
the extracted parameters needs to be modified incase you are not considering the snow bucket the first 5 parameters are disregarded
