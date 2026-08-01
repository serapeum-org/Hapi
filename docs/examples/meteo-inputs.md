# Rainfall Runoff Model Inputs
The required inputs for the distributed model is divided into Meteorological, GIS inputs and Distributed model parameters

![process](../img/process.png)


# Meteorological Inputs
To be able to run the hydrologic simulation with Hapi the following meteorological inputs are required

	- rainfall

	- evapotranspiration

	- Temperature

Distributed meteorological data can be obtain from gauge data with some interpolation method or from remote sensing data

# Remote Sensing Data

The remote sensing module that used to download CHIRPS and ECMWF data was moved out of
Hapi into its own package, [earth2observe](https://pypi.org/project/earth2observe/):

```shell
pip install earth2observe
```

earth2observe covers the same sources Hapi used to handle:

- **CHIRPS** — the Climate Hazards Group InfraRed Precipitation with Station data, a
  quasi-global rainfall data set combining satellite imagery with in-situ station data
  on a 0.05 degree grid, from 1981 to near present.
- **ECMWF** — the ERA-Interim archive, which requires a registered account and an API
  key set up on your machine (see the
  [ECMWF registration](https://apps.ecmwf.int/registration/) and the
  [API key instructions](https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key)).

Once the rasters are downloaded, prepare them for the model with `hapi.inputs.Inputs`,
which aligns every raster to the catchment DEM — see
[GIS inputs](gis-inputs.md) and [Parameters](parameters.md).
