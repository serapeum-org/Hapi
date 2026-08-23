r"""Regenerate the Coello NetCDF fixtures from the raster folders.

Packs `prec/`, `temp/` and `evap/` into one NetCDF each, then merges the three into
`meteo.nc`, whose variables are named after the drivers.

Run from the repository root:

    pixi run -e dev python tests/rrm/data/coello/convert_meteo_inputs_to_netcdf.py

The Coello rasters are named `0_Tair2m_..._2009.01.01.tif` -- a `%Y.%m.%d` date with dots.
Rhine's are `0_Temp_..._1979_1_1.tif`, so they need `r"\d{4}_\d{1,2}_\d{1,2}"` and
`"%Y_%m_%d"` instead.
"""

from __future__ import annotations

# %% Setup
import numpy as np

from hapi.inputs import METEO_VARIABLES, MeteoInputs

root = "tests/rrm/data/coello"

#: driver -> its raster folder. The NetCDF takes the folder's name.
FOLDERS = {"precipitation": "prec", "temperature": "temp", "evapotranspiration": "evap"}

READER = dict(regex_string=r"\d{4}.\d{2}.\d{2}", file_name_data_fmt="%Y.%m.%d")

# On a NAS, add gdal_env={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"} to READER: GDAL
# otherwise lists the directory on every open, 369 ms per raster instead of 18 ms. It also
# stops GDAL finding .aux.xml / world files / .ovr, so only when the folder has none.

# %% Pack each driver, then merge the three
packed = [
    MeteoInputs.raster_folder_to_netcdf(
        f"{root}/{folder}", f"{root}/{folder}.nc", **READER
    )
    for folder in FOLDERS.values()
]
combined = MeteoInputs.combine_netcdf_files(*packed, f"{root}/meteo.nc")
print(f"written    : {', '.join(p.name for p in packed)}, {combined.name}")

# %% Check what was written still matches the rasters
merged = MeteoInputs.from_netcdf(
    combined,
    precipitation="precipitation",
    temperature="temperature",
    evapotranspiration="evapotranspiration",
)
from_rasters = MeteoInputs.from_rasters(
    *(f"{root}/{folder}" for folder in FOLDERS.values()), **READER
)
for name in METEO_VARIABLES:
    identical = np.array_equal(
        getattr(merged, name), getattr(from_rasters, name), equal_nan=True
    )
    print(f"  {name:20s} identical to {FOLDERS[name]}/: {identical}")
