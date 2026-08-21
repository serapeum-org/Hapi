r"""Combine the per-variable NetCDFs into one file holding all three drivers.

Reads the files ``convert_temperature_to_netcdf.py`` produces one at a time -- ``prec.nc``,
``temp.nc``, ``evap.nc`` -- and writes ``meteo.nc``, whose variables are named
``precipitation`` / ``temperature`` / ``evapotranspiration``.

``to_netcdf`` names one variable per band, after the band name, and band names survive a GeoTIFF
round-trip as band descriptions. So the three cubes are stacked into a three-band raster per
timestep and the bands are named; that is what gives the variables their names instead of
``Band_1``. The staging rasters are written to a temporary folder and deleted afterwards.

Rerunning this is byte-for-byte idempotent.
"""

import shutil
import tempfile
from datetime import datetime as dt
from pathlib import Path

import numpy as np
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.netcdf import NetCDF

# %% Paths
root_dir = r"tests/rrm/data/coello"

# output variable name -> the single-variable NetCDF holding it
SOURCES = {
    "precipitation": "prec.nc",
    "temperature": "temp.nc",
    "evapotranspiration": "evap.nc",
}
out_path = f"{root_dir}/meteo.nc"

DATE_FMT = "%Y.%m.%d"

# %% Read the three cubes and the georeferencing
cubes, calendar, geo, epsg, nodata = [], None, None, None, None
for name, file_name in SOURCES.items():
    nc = NetCDF.read_file(f"{root_dir}/{file_name}")
    if len(nc.variable_names) != 1:
        raise ValueError(
            f"{file_name} holds {nc.variable_names}; expected one variable"
        )
    cubes.append(np.asarray(nc.get_variable(nc.variable_names[0]).read_array()))
    if calendar is None:
        calendar = [dt.strptime(str(s)[:10], "%Y-%m-%d") for s in nc.time_stamp]
        # The GeoTransform root attribute, not nc.geotransform: that property reports an x
        # pixel size of 1.0 and a half-pixel-shifted origin (pyramids #1014).
        geo = tuple(float(v) for v in nc.global_attributes["GeoTransform"].split())
        epsg = int(nc.global_attributes["epsg"])
        nodata = float(nc.global_attributes["nodata"])

shapes = {n: c.shape for n, c in zip(SOURCES, cubes)}
if len(set(shapes.values())) != 1:
    raise ValueError(f"the three files must share a shape, got {shapes}")
print(f"cubes      : {shapes}  (time, rows, cols)")
print(f"calendar   : {calendar[0]:%Y-%m-%d} -> {calendar[-1]:%Y-%m-%d}")
print(f"geo        : {geo}  epsg={epsg}  nodata={nodata}")

# %% Stack into named bands and write the combined NetCDF
staging = Path(tempfile.mkdtemp(prefix="combine-netcdf-"))
try:
    for step, stamp in enumerate(calendar):
        bands = np.stack([cube[step] for cube in cubes]).astype("float32")
        raster = Dataset.create_from_array(
            bands, geo=geo, epsg=epsg, no_data_value=nodata
        )
        raster.band_names = list(SOURCES)
        raster.to_file(str(staging / f"{step}_meteo_{stamp:{DATE_FMT}}.tif"))

    collection = DatasetCollection.from_files(
        staging, glob="*.tif", date_format=DATE_FMT, date_regex=r"\d{4}.\d{2}.\d{2}"
    )
    Path(out_path).unlink(missing_ok=True)
    collection.to_netcdf(out_path)
finally:
    shutil.rmtree(staging, ignore_errors=True)
print(f"written    : {out_path}")

# %% Verify the file round-trips
nc = NetCDF.read_file(out_path)
print(f"variables  : {nc.variable_names}")
print(f"dimensions : {nc.dimension_sizes}  epsg={nc.epsg}")
for name, cube in zip(SOURCES, cubes):
    back = np.asarray(nc.get_variable(name).read_array())
    same = np.array_equal(back, cube)
    print(f"  {name:20s} identical to {SOURCES[name]}: {same}")
