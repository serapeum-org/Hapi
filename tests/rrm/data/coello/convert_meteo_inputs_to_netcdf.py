r"""Pack the three meteorological driver folders into one NetCDF each.

Thin wrapper over `MeteoInputs.raster_folder_to_netcdf`, which does the reading, the ordering
and the write. This file only holds the paths and prints what came out, so it can be stepped
through cell by cell.

Produces `prec.nc`, `temp.nc` and `evap.nc`. `combine_netcdf.py` is the next step: it merges
those three into `meteo.nc`.

The Coello fixtures are named `0_Tair2m_..._2009.01.01.tif` -- a `%Y.%m.%d` date with dots.
Rhine's are `0_Temp_ECMWF_ERA_Interim_C_daily_1979_1_1.tif`: underscores, and a month and day
that are not zero-padded, so both the regex and the format have to say so (see RHINE below).
"""

from __future__ import annotations

# %% Setup
import numpy as np

from hapi.inputs import METEO_VARIABLES, MeteoInputs

root_dir = r"tests/rrm/data/coello"

#: driver -> the folder of rasters holding it. The NetCDF takes the folder's own name.
FOLDERS = {
    "precipitation": "prec",
    "temperature": "temp",
    "evapotranspiration": "evap",
}

# How the date sits in the file names. The format is inferred from the names when omitted,
# which covers `2009.01.01` and `20090101`; pass it for anything else.
DATE_REGEX = r"\d{4}.\d{2}.\d{2}"
DATE_FMT = "%Y.%m.%d"

# For the Rhine rasters instead:
# DATE_REGEX, DATE_FMT = r"\d{4}_\d{1,2}_\d{1,2}", "%Y_%m_%d"

# Optional window; None converts the whole folder.
START = None
END = None

# GDAL lists the directory on every open to look for sidecars. Against 14,823 files on the
# NAS that is a remote listing per raster: 369 ms instead of 18 ms, i.e. 90 min instead of 4.
# It also stops GDAL finding .aux.xml / world files / .ovr, so enable it only when the folder
# has none -- which is why it is not the library default.
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_PAM_ENABLED": "NO"}

READER = dict(
    regex_string=DATE_REGEX,
    file_name_data_fmt=DATE_FMT,
    start=START,
    end=END,
    gdal_env=GDAL_ENV,
)

raster_dirs = {name: f"{root_dir}/{folder}" for name, folder in FOLDERS.items()}

# %% Convert each driver
# Refuses to write if the rasters yield no calendar, or if they come back out of order.
written = {
    name: MeteoInputs.raster_folder_to_netcdf(
        raster_dirs[name], f"{root_dir}/{folder}.nc", **READER
    )
    for name, folder in FOLDERS.items()
}
for name, path in written.items():
    print(f"packed     : {name:20s} -> {path.name}")

# %% Read the three back and check they round-trip to the cubes the rasters hold
packed = MeteoInputs.from_netcdf_files(*written.values())
from_rasters = MeteoInputs.from_rasters(*raster_dirs.values(), **READER)

print(f"shape      : {packed.shape}  (rows, cols, time)")
print(f"calendar   : {packed.time[0]:%Y-%m-%d} -> {packed.time[-1]:%Y-%m-%d}")
for name in METEO_VARIABLES:
    identical = np.array_equal(
        getattr(packed, name), getattr(from_rasters, name), equal_nan=True
    )
    print(f"  {name:20s} identical to its rasters: {identical}")

# %% Next: merge the three into one file
# python tests/rrm/data/coello/combine_netcdf.py
