"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi
"""

from hapi.rrm.inputs import Inputs

rpath = "examples/hydrological-model/data"
"""
this function prepare downloaded raster data to have the same align and
nodatavalue from a GIS raster (DEM, flow accumulation, flow direction raster)
and return a folder with the output rasters with a name “New_Rasters”
"""
src_path = f"{rpath}/gis_data/acc4000.tif"
outputpath = f"{rpath}/meteo_data/meteodata_prepared/prec"
prec_in_path = f"{rpath}/meteo_data/raw_data/prec/"

Inputs.prepareInputs(src_path, prec_in_path, outputpath)
