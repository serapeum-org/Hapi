"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example

if the raster files in the rpath have names that starts with a number like
    0_Tair2m_ECMWF_ERA-Interim_C_daily_2009.01.01.tif
    1_Tair2m_ECMWF_ERA-Interim_C_daily_2009.01.02.tif
    2_Tair2m_ECMWF_ERA-Interim_C_daily_2009.01.03.tif

these are the new names as the renaming function renames the raster names in place

for the exercise to practice renaming the files delete these already renamed rasters
and upzip the temp-lumped-example.zip file then you will get rasters with names that looks like those you will get
from the download script from CHIRPS or ECMWF.
"""

from hapi.rrm.inputs import Inputs as IN

path = (
    "examples/hydrological-model/data/meteo_data/meteodata_prepared/temp-rename-example"
)

IN.renameFiles(
    path, prefix="Tair2m_ECMWF_ERA-Interim_C_daily", fmt="%Y.%m.%d", freq="daily"
)
