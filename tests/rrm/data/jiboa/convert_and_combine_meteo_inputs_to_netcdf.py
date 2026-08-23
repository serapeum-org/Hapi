r"""Regenerate the Jiboa NetCDF fixtures from the raster folders.

Two steps, four files: each of `prec/`, `temp/` and `evap/` is packed into its own NetCDF,
then the three are merged into `meteo.nc` with the variables named after the drivers. The
Coello counterpart is `tests/rrm/data/coello/convert_and_combine_meteo_inputs_to_netcdf.py`.

Run from the repository root:

    pixi run -e dev python tests/rrm/data/jiboa/convert_and_combine_meteo_inputs_to_netcdf.py

NOTE: `meteo_inputs/prec`, `temp` and `evap` are **empty in the repository** -- the Jiboa
rasters are not committed, which is why no distributed Jiboa run is exercised anywhere. Point
`root` at a copy that has them; the example data set is one:

    root = "examples/hydrological-model/jiboa/data/meteo-data"

Unlike Coello's daily `..._2009.01.01.tif`, Jiboa is **hourly** and its dates are underscore
separated and not zero-padded -- `Rain_ISDW_2012_6_14_19_4000.tif` is 19:00 on 14 June 2012.
The trailing `_4000` is the cell size, not part of the date; `\d{4}_\d{1,2}_\d{1,2}_\d{1,2}`
cannot match it because nothing follows it.
"""

from __future__ import annotations

# %% Setup
import numpy as np

from hapi.inputs import METEO_VARIABLES, MeteoInputs

root = "tests/rrm/data/jiboa/meteo_inputs"

# Driver -> its raster folder; each NetCDF takes the folder's name.
FOLDERS = {"precipitation": "prec", "temperature": "temp", "evapotranspiration": "evap"}

# Hourly, underscore separated, not zero-padded. The format cannot be inferred from names
# shaped like this, so it has to be given.
READER = dict(
    regex_string=r"\d{4}_\d{1,2}_\d{1,2}_\d{1,2}", file_name_data_fmt="%Y_%m_%d_%H"
)

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
print(
    f"steps      : {merged.time_steps} hourly, {merged.time[0]:%Y-%m-%d %H:%M} -> "
    f"{merged.time[-1]:%Y-%m-%d %H:%M}"
)
for name in METEO_VARIABLES:
    identical = np.array_equal(
        getattr(merged, name), getattr(from_rasters, name), equal_nan=True
    )
    print(f"  {name:20s} identical to {FOLDERS[name]}/: {identical}")
