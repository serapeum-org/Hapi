r"""Pack one driver's folder of dated rasters into a single NetCDF.

Thin wrapper over `MeteoInputs.raster_folder_to_netcdf`, which does the reading, the
ordering and the write. This file only holds the paths and prints what came out, so it can
be stepped through cell by cell.

Set `VAR` to the folder to convert. The Coello fixtures are named
`0_Temp_..._2009.01.01.tif` -- a `%Y.%m.%d` date with dots. Rhine's are
`0_Temp_ECMWF_ERA_Interim_C_daily_1979_1_1.tif`: underscores, and a month and day that are
not zero-padded, so both the regex and the format have to say so (see RHINE below).
"""

from __future__ import annotations

# %% Setup
import numpy as np
from pyramids.netcdf import NetCDF

from hapi.inputs import MeteoInputs

root_dir = r"tests/rrm/data/coello"

VAR = "temp"
raster_dir = f"{root_dir}/{VAR}"
out_path = f"{root_dir}/{VAR}.nc"

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

# %% Convert
# Refuses to write if the rasters yield no calendar, or if they come back out of order.
written = MeteoInputs.raster_folder_to_netcdf(
    raster_dir,
    out_path,
    regex_string=DATE_REGEX,
    file_name_data_fmt=DATE_FMT,
    start=START,
    end=END,
    gdal_env=GDAL_ENV,
)
print(f"written    : {written}")

# %% Inspect what was written
nc = NetCDF.read_file(str(written))
print(f"variables  : {nc.variable_names}")
print(f"dimensions : {nc.dimension_sizes}")

values = np.asarray(nc.get_variable(nc.variable_names[0]).read_array())
print(f"array      : {values.shape}  (time, rows, cols)")

# %% Check it round-trips to the same cube the rasters hold
packed = MeteoInputs.from_netcdf_files(written, written, written)
from_rasters = MeteoInputs.from_rasters(
    raster_dir,
    raster_dir,
    raster_dir,
    regex_string=DATE_REGEX,
    file_name_data_fmt=DATE_FMT,
    start=START,
    end=END,
    gdal_env=GDAL_ENV,
)
identical = np.array_equal(packed.temperature, from_rasters.temperature, equal_nan=True)
print(f"calendar   : {packed.time[0]:%Y-%m-%d} -> {packed.time[-1]:%Y-%m-%d}")
print(f"round-trip : identical to the rasters: {identical}")

# %% Combine three of them into one file, once each driver has been packed
# MeteoInputs.combine_netcdf_files(
#     f"{root_dir}/prec.nc", f"{root_dir}/temp.nc", f"{root_dir}/evap.nc",
#     f"{root_dir}/meteo.nc",
# )
