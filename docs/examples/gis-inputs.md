# GIS Inputs

In order to determine the direction of the flow Hapi used the D8 `flow direction` and `flow accumulation` rasters. Since the avialable DEMs are in resolution of few tens of meters (30 or 90 m) and usually hydrological models are build to represent units much bigger than that so we need to get the River network from these fine resolution DEMs first and then burn the river into courser resolution (i.e. 5km ) that we want to use as a resolution for our distributed hydrological model.

![Flow Direction](../img/flowdirection.png){ width="400" }

![Upper Zone](../img/UZ.png){ width="400" }


# Burning the River into the DEM

1- Create suitable Flow accumulation
After deriving the the flow accumulation form the fine resolution DEM (30 or 90 m)
use the raster calulator tool in ArcMap/QGIS to obtain a raster of the cells that has accumulation more than 6 times the acumulation in the corse resolution

##### lets say you want to build the hydrological model for 5km resolution DEM and you have a 100x100 m2 DEM [(5000x5000) / (30x30)] / 30 and use the resulted number in the raster calculator
##### "Flow_acc" > 75000
output will be the raster `acc75000.tif`

2- Create Stream Links

	output will be the raster `StreamLinks75000.tif`

3- Convert the stream Links shapefile into vector

	Stream to feature tool in ArcMap toolbox
	output will be the raster `Streams_V.tif`

the following steps are done in QGIS

4- Convert the Streams vector into Raster using the new DEM (5km resolution)

	in order to have the streams as a raster we have to use the resampled DEM (5km) in the process to allocate a specific value (1) in the cells that has the river and zero in other cells

	output will be the raster `StreamRaster.tif`

	there are tow ways that you can lower the river network in the DEM


`StreamRaster.tif` -100

	output will be the raster `RiverDepth.tif`


5- Reclassify

	now the `StreamRaster.tif` has a value of 1 at the location of the river and the NoData value in other cells, in order to add this raster to the DEM raster we are going to assign 0 at all other locations, using the `reclassify` function

	output will be the raster `reclassifiedGrid.sdat`



6- Standardize the DEM (5km)

	to standerdize the DEM (5km) we need the max and min value in the DEM, for that you can use the `raster layer statistics`, then use the raster calculator (Raster>Raster Calculator) to standardize the DEM

	output will be the raster `DEM_Standardized.tif`

7- River Depth

	in this step we will subtract the reclassified raster from the standardized DEM using raster calculator

	output will be the raster `Standardized_Burn.tif`

8- Final step

	this step we will multiply the DEM (5km) with the `Standardized_Burn.tif` raster

	output will be the raster `DEM_Burn.tif`