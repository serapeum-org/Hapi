"""Download the Coello meteorological inputs with earthlens.

Make sure the working directory is set to the root of the Hapi repo:
current_work_directory = Hapi/

`earthlens` (https://github.com/serapeum-org/earthlens) replaces the deprecated `earth2observe` package as the
source of Hapi's meteorological inputs. It exposes one facade, `EarthLens`, that routes a request to the backend
named by `data_source`:

    pip install earthlens            # CHIRPS and the other keyless backends
    pip install earthlens[ecmwf]     # adds cdsapi, needed for the ERA5 download below

CHIRPS is served over anonymous FTP and needs no credentials. ERA5 goes through the Copernicus Climate Data Store,
which needs a free CDS account and a `~/.cdsapirc` holding the Personal Access Token
(https://cds.climate.copernicus.eu/how-to-api).

The two backends do not write the same thing:

- CHIRPS writes one clipped GeoTIFF per date, named `<dataset>_<variable>_%Y.%m.%d.tif`.
- ECMWF writes a single NetCDF per variable. Passing `aggregate=AggregationConfig(...)` slices it into per-date
  GeoTIFFs named `<cds_variable>_<freq>_%Y%m%d.tif`, which is the shape `Inputs.prepare_inputs` expects.

`op="auto"` reads the catalog's `is_flux` flag so state variables (temperature) are averaged over each window and
flux variables (evaporation, precipitation) are summed.
"""

from __future__ import annotations

from pathlib import Path

from earthlens.core import AggregationConfig, EarthLens

# %% Basin data
start = "2009-01-01"
end = "2009-02-01"
temporal_resolution = "daily"
latlim = [4.190755, 4.643963]
lonlim = [-75.649243, -74.727286]

# Downloads land in their own folder so the rasters shipped with the repo under
# `data/meteo_data/raw_data/` are left untouched. Point `01-coello-prepare_inputdata.py`
# here once the download finishes.
path = Path("examples/hydrological-model/data/meteo_data/earthlens")

# %%
"""
Check the dataset keys and variable names each backend accepts before requesting anything.
"""
print(EarthLens.list_datasets("chc")[:10])
print(EarthLens.describe_dataset("chc", "global-daily"))

print(EarthLens.list_datasets("ecmwf")[:10])
print(EarthLens.describe_dataset("ecmwf", "reanalysis-era5-single-levels"))

# %% Precipitation from CHIRPS
"""
Provide the time period, temporal resolution, extent and variables of interest.

`variables` as a plain list uses the global CHIRPS-2.0 series, and `temporal_resolution` picks the dataset
("daily" -> "global-daily", "monthly" -> "global-monthly"). Pass the dict shape,
e.g. `variables={"africa-pentad": ["precipitation"]}`, to reach the rest of the CHC catalog.
"""
EarthLens(
    data_source="chc",
    start=start,
    end=end,
    variables=["precipitation"],
    lat_lim=latlim,
    lon_lim=lonlim,
    temporal_resolution=temporal_resolution,
    path=path / "prec",
).download()

# %% Temperature and evapotranspiration from ECMWF (ERA5)
"""
ERA5 variables are addressed by the (dataset, variable) pair, so `variables` is a dict keyed by the CDS dataset
name. `2m-temperature` is a state variable and `evaporation` is a flux; `op="auto"` applies the right reduction
to each.

Note that ERA5 evaporation is negative by convention (downward fluxes are positive), so the values have to be
flipped before the model reads them - `01-coello-prepare_inputdata.py` does that with `Dataset.apply(np.abs)`.

The raw NetCDF is kept in its own folder so the per-date GeoTIFFs sit in a directory holding nothing else -
`Inputs.prepare_inputs` reads a whole folder at once.
"""
EarthLens(
    data_source="ecmwf",
    start=start,
    end=end,
    variables={"reanalysis-era5-single-levels": ["2m-temperature"]},
    lat_lim=latlim,
    lon_lim=lonlim,
    temporal_resolution=temporal_resolution,
    path=path / "netcdf",
).download(aggregate=AggregationConfig(freq="1D", op="auto", out_dir=path / "temp"))

EarthLens(
    data_source="ecmwf",
    start=start,
    end=end,
    variables={"reanalysis-era5-single-levels": ["evaporation"]},
    lat_lim=latlim,
    lon_lim=lonlim,
    temporal_resolution=temporal_resolution,
    path=path / "netcdf",
).download(aggregate=AggregationConfig(freq="1D", op="auto", out_dir=path / "evap"))

# %%
"""
The CHIRPS backend fetches one file per date over FTP, so it takes a `cores` argument to run the retrieval in
parallel. Enter the number of cores you want to use.

PS. the multi-core download does not have an indication bar.
"""
EarthLens(
    data_source="chc",
    start=start,
    end=end,
    variables=["precipitation"],
    lat_lim=latlim,
    lon_lim=lonlim,
    temporal_resolution=temporal_resolution,
    path=path / "prec",
).download(cores=4)
