r"""Regenerate the Coello NetCDF fixtures from the raster folders.

Two steps, four files: each of `prec/`, `temp/` and `evap/` is packed into its own NetCDF,
then the three are merged into `meteo.nc` with the variables named after the drivers.

Run from the repository root:

    pixi run -e dev python tests/rrm/data/coello/convert_and_combine_meteo_inputs_to_netcdf.py

The Coello rasters are named `0_Tair2m_..._2009.01.01.tif` -- a `%Y.%m.%d` date with dots.
Rhine's are `0_Temp_..._1979_1_1.tif`, so they need `r"\d{4}_\d{1,2}_\d{1,2}"` and
`"%Y_%m_%d"` instead.
"""

from __future__ import annotations

# %% Setup
import numpy as np

from hapi.inputs import METEO_VARIABLES, MeteoInputs

root = "tests/rrm/data/coello"

# Driver -> its raster folder; each NetCDF takes the folder's name.
FOLDERS = {"precipitation": "prec", "temperature": "temp", "evapotranspiration": "evap"}

# Where the date sits in the file names, and how to parse it.
READER = dict(regex_string=r"\d{4}.\d{2}.\d{2}", file_name_data_fmt="%Y.%m.%d")

# On a NAS add gdal_env={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"} to READER -- GDAL
# otherwise lists the directory per open, 369 ms a raster instead of 18 ms. It also hides
# .aux.xml / world files / .ovr, so only where the folder has none.

# %% Convert: one NetCDF per driver
packed = [
    MeteoInputs.raster_folder_to_netcdf(
        f"{root}/{folder}", f"{root}/{folder}.nc", **READER
    )
    for folder in FOLDERS.values()
]

# %% Combine: the three into one, variables named after the drivers
combined = MeteoInputs.combine_netcdf_files(*packed, f"{root}/meteo.nc")
print(f"written    : {', '.join(p.name for p in packed)}, {combined.name}")

# %% Check the merged file still holds what the rasters do
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
