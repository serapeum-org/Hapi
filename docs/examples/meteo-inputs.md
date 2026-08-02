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

Remote sensing inputs are downloaded with [earthlens](https://github.com/serapeum-org/earthlens), which replaces the
deprecated `earth2observe` package:

```shell
pip install earthlens            # CHIRPS and the other keyless backends
pip install earthlens[ecmwf]     # adds cdsapi, required for the ERA5 download below
```

earthlens exposes a single facade, `EarthLens`, that routes the request to the backend named by `data_source`. It
covers the sources Hapi used to handle, among many others:

- **CHIRPS** (`data_source="chc"`) — the Climate Hazards Group InfraRed Precipitation with Station data, a
  quasi-global rainfall data set combining satellite imagery with in-situ station data on a 0.05 degree grid, from
  1981 to near present. Served over anonymous FTP, so it needs no credentials.
- **ERA5** (`data_source="ecmwf"`) — the Copernicus Climate Data Store reanalysis that replaces the retired
  ERA-Interim archive. It requires a free [CDS account](https://cds.climate.copernicus.eu/) and a `~/.cdsapirc`
  holding the Personal Access Token (see the [CDS API instructions](https://cds.climate.copernicus.eu/how-to-api)).

## Rainfall from CHIRPS

Passing `variables` as a plain list uses the global CHIRPS-2.0 series, and `temporal_resolution` selects the
dataset (`"daily"` → `global-daily`, `"monthly"` → `global-monthly`). The dict shape, e.g.
`variables={"africa-pentad": ["precipitation"]}`, reaches the rest of the CHC catalog.

```python
from earthlens.core import EarthLens

EarthLens(
    data_source="chc",
    temporal_resolution="daily",
    start="2009-01-01",
    end="2009-02-01",
    variables=["precipitation"],
    lat_lim=[4.190755, 4.643963],
    lon_lim=[-75.649243, -74.727286],
    path="examples/hydrological-model/data/meteo_data/earthlens/prec",
).download(cores=4)
```

CHIRPS fetches one file per date, so `cores` runs the retrieval in parallel (without a progress bar). Each date is
clipped to the bounding box and written as `<dataset>_<variable>_%Y.%m.%d.tif`, for example
`global-daily_precipitation_2009.01.01.tif`.

## Temperature and evapotranspiration from ERA5

ERA5 variables are addressed by the `(dataset, variable)` pair, so `variables` is a dict keyed by the CDS dataset
name. Use the catalog to look up the codes:

```python
from earthlens.ecmwf import Catalog

catalog = Catalog()
print(list(catalog.datasets))
catalog.get_variable("reanalysis-era5-single-levels", "2m-temperature")
```

Unlike CHIRPS, the ECMWF backend writes **one NetCDF per variable** rather than per-date rasters. Passing
`aggregate=AggregationConfig(...)` slices that NetCDF into per-date GeoTIFFs, which is the shape
`Inputs.prepare_inputs` expects:

```python
from earthlens.core import AggregationConfig, EarthLens

root = "examples/hydrological-model/data/meteo_data/earthlens"

EarthLens(
    data_source="ecmwf",
    temporal_resolution="daily",
    start="2009-01-01",
    end="2009-02-01",
    variables={"reanalysis-era5-single-levels": ["2m-temperature"]},
    lat_lim=[4.190755, 4.643963],
    lon_lim=[-75.649243, -74.727286],
    path=f"{root}/netcdf",
).download(aggregate=AggregationConfig(freq="1D", op="auto", out_dir=f"{root}/temp"))
```

`op="auto"` reads the catalog's `is_flux` flag so each window gets the right reduction: state variables such as
`2m-temperature` are averaged, flux variables such as `evaporation` and `total-precipitation` are summed. Keeping
the NetCDF in its own folder leaves the GeoTIFF directory holding nothing else, since `prepare_inputs` reads a
whole folder at once.

ERA5 evapotranspiration is negative by convention — the meteorological convention is that downward vertical fluxes
are positive — so the values must be flipped before the model reads them. `Dataset.apply(np.abs)` does that; see
`01-coello-prepare_inputdata.py` in the [prepare-Input-data examples][prepare-inputs].

[prepare-inputs]: https://github.com/serapeum-org/Hapi/tree/main/examples/hydrological-model/coello/prepare-Input-data

## Matching the file names when reading the rasters

`read_rainfall`, `read_temperature` and `read_et` locate the date in each file name with `regex_string` and parse it
with `file_name_data_fmt`, so the two must match the names the backend produced:

| Source | Example file name | `regex_string` | `file_name_data_fmt` |
|---|---|---|---|
| CHIRPS | `global-daily_precipitation_2009.01.01.tif` | `r"\d{4}.\d{2}.\d{2}"` (the default) | `"%Y.%m.%d"` |
| ERA5 (aggregated) | `2m_temperature_1D_20090101.tif` | `r"\d{8}"` | `"%Y%m%d"` |

The default `regex_string` does **not** match the ERA5 names, so pass both arguments for those rasters:

```python
model.read_temperature(temp_path, regex_string=r"\d{8}", file_name_data_fmt="%Y%m%d")
```

Once the rasters are downloaded, prepare them for the model with `hapi.inputs.Inputs`, which aligns every raster to
the catchment DEM — see [GIS inputs](gis-inputs.md) and [Parameters](parameters.md).
