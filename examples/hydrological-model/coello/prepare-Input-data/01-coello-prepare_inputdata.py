"""Prepare the Coello meteorological and GIS input rasters.

Make sure the working directory is set to the root of the Hapi repo:
current_work_directory = Hapi/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyramids.dataset import Dataset

from hapi.inputs import Inputs

metao_data_path = "examples/hydrological-model/data/meteo_data"
gis_data_path = "examples/hydrological-model/data/gis_data"
"""
prepare_inputs aligns and crops the downloaded raster data to have the same
alignment and no-data value as a GIS raster (DEM, flow accumulation, or flow
direction raster) and writes the prepared rasters to the given output folder.
"""

dem_path = f"{gis_data_path}/acc4000.tif"

inputs = Inputs(dem_path)
outputpath = f"{metao_data_path}/meteodata_prepared/"

# prec
prec_in_path = f"{metao_data_path}/raw_data/prec/"
inputs.prepare_inputs(prec_in_path, outputpath + "prec0")

# evap
evap_in_path = f"{metao_data_path}/raw_data/evap/"
inputs.prepare_inputs(evap_in_path, outputpath + "evap0")
# temp
temp_in_path = f"{metao_data_path}/raw_data/temp/"
inputs.prepare_inputs(temp_in_path, outputpath + "temp0")

"""
in case you want to manipulate the values in all the rasters of one of the inputs,
for example evapotranspiration values in rasters downloaded from ECMWF are -ve,
and to change them to +ve in all rasters (or apply any kind of function to all
input rasters in the same folder), loop over the files and use `Dataset.apply`.

"How can evaporation have both positive and negative values?
Evaporation is normally negative due to the convention for fluxes.
The meteorological convention for all vertical fluxes is that downwards is positive.
Positive evaporation represents condensation'.
Link: https://confluence.ecmwf.int/pages/viewpage.action?pageId=111155327
"""
evap_out_path = f"{metao_data_path}/meteodata_prepared/evap/"
new_folder_path = f"{metao_data_path}/meteodata_prepared/new_evap/"

Path(new_folder_path).mkdir(parents=True, exist_ok=True)
for file in sorted(Path(evap_out_path).glob("*.tif")):
    dataset = Dataset.read_file(str(file))
    # apply operates on the domain cells only and keeps the no-data cells
    dataset.apply(np.abs).to_file(f"{new_folder_path}/{file.name}")

"""
in order to run the model all inputs have to have the same number of rows and columns.
`Dataset.align` resamples a raster and copies the coordinate system, the number of
rows/columns, and the cell size from a source raster (the DEM raster).
"""

soil_path = f"{gis_data_path}/soil_classes.tif"
dem = Dataset.read_file(dem_path)
soil = Dataset.read_file(soil_path)

# align the soil raster to the DEM alignment
aligned_soil = soil.align(dem)

# the no-data cells are still different: some cells are no-data in the soil
# raster but not in the DEM raster. Crop with the DEM as a raster mask so both
# rasters share the same no-data cells.
masked_soil = aligned_soil.crop(dem)

# save the new raster
masked_soil.to_file(f"{gis_data_path}/soil_classes_aligned.tif")
