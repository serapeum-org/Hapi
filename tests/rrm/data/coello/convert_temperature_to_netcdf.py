r"""Convert the Rhine temperature rasters into a single NetCDF file.

The rasters are named ``0_Temp_ECMWF_ERA_Interim_C_daily_1979_1_1.tif``: an ordering index,
then a ``%Y_%m_%d`` date with non zero-padded month and day.

Needs pyramids >= 0.50.
"""

from datetime import datetime as dt

import numpy as np
from pyramids.dataset import DatasetCollection
from pyramids.netcdf import NetCDF

# %% Paths
""
root_dir = r"tests/rrm/data/coello"

var = "temp"

temp_path = f"{root_dir}/{var}"
out_path = f"{root_dir}/{var}.nc"

DATE_REGEX = r"\d{4}.\d{1,2}.\d{1,2}"
DATE_FMT = "%Y.%m.%d"

# Optional window, e.g. "1979-01-01" -> "1979-01-31". None converts every raster.
START = None
END = None
DATE_WINDOW_FMT = "%Y.%m.%d"

# GDAL lists the directory on every open to look for sidecars. Against 14,823 files on the NAS
# that is a remote listing per raster: 369 ms instead of 18 ms, i.e. 90 min instead of 4.
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_PAM_ENABLED": "NO"}

# %% Read the rasters as a temporal collection
collection = DatasetCollection.from_files(
    temp_path,
    glob="*.tif",
    date_format=DATE_FMT,
    date_regex=DATE_REGEX,
    start=dt.strptime(START, DATE_WINDOW_FMT) if START else None,  # noqa: DTZ007
    end=dt.strptime(END, DATE_WINDOW_FMT) if END else None,  # noqa: DTZ007
    gdal_env=GDAL_ENV,
)

time_stamps = list(collection.time)
print(f"time steps : {collection.time_length}")
print(f"band names : {collection.meta.band_names}")
print(f"shape      : {collection.meta.shape}  (bands, rows, cols)")
print(f"calendar   : {time_stamps[0]:%Y-%m-%d} -> {time_stamps[-1]:%Y-%m-%d}")
assert time_stamps == sorted(time_stamps), "the calendar axis is not chronological"

# %% Write the NetCDF
collection.to_netcdf(out_path)
print(f"written    : {out_path}")

# %% Verify the file round-trips
nc = NetCDF.read_file(out_path)
print(f"variables  : {nc.variable_names}")
print(f"dimensions : {nc.dimension_sizes}")
print(f"time stamps: {nc.time_stamp[0]} -> {nc.time_stamp[-1]}")

var = nc.get_variable(nc.variable_names[0])
values = np.asarray(var.read_array())
print(f"array      : {values.shape}  (time, rows, cols)")

# %% Plot
# var.plot(animate=True) paints the no-data cells and ignores exclude_value (pyramids #1013),
# so drive cleopatra directly. read_array is eager, hence the window: the full cube is 657 MiB
# and 14,823 frames is not a watchable animation.
