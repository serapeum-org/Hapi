r"""Combine the per-variable NetCDFs into one file holding all three drivers.

Reads the files `convert_temperature_to_netcdf.py` produces one at a time -- `prec.nc`,
`temp.nc`, `evap.nc` -- and writes `meteo.nc`, whose variables are named
`precipitation` / `temperature` / `evapotranspiration`.

The first file seeds the container; the other two are pulled in with `NetCDF.add_variable`,
which copies the MDArray across, and each is renamed on arrival. Nothing touches disk until
`to_file`, so the source files are left as they are.
"""

from pathlib import Path

import numpy as np
from pyramids.netcdf import NetCDF

# %% Paths
root_dir = r"tests/rrm/data/coello"
out_path = f"{root_dir}/meteo.nc"

# output variable name -> the single-variable NetCDF holding it
SOURCES = {
    "precipitation": "prec.nc",
    "temperature": "temp.nc",
    "evapotranspiration": "evap.nc",
}

# %% Combine
(seed_name, seed_file), *rest = SOURCES.items()
combined = NetCDF.read_file(f"{root_dir}/{seed_file}")
combined.rename_variable(combined.variable_names[0], seed_name)

for name, file_name in rest:
    source = NetCDF.read_file(f"{root_dir}/{file_name}")
    combined.add_variable(source)
    combined.rename_variable(source.variable_names[0], name)

Path(out_path).unlink(missing_ok=True)
combined.to_file(out_path)
print(f"written    : {out_path}")

# %% Verify the file round-trips
nc = NetCDF.read_file(out_path)
print(f"variables  : {sorted(nc.variable_names)}")
print(f"dimensions : {nc.dimension_sizes}  epsg={nc.epsg}")
print(f"geo        : {nc.global_attributes['GeoTransform']}")
for name, file_name in SOURCES.items():
    written = np.asarray(nc.get_variable(name).read_array())
    source = NetCDF.read_file(f"{root_dir}/{file_name}")
    expected = np.asarray(source.get_variable(source.variable_names[0]).read_array())
    print(f"  {name:20s} identical to {file_name}: {np.array_equal(written, expected)}")
