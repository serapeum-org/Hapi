"""Created on Sat Mar 27 19:09:20 2021.

@author: mofarrag

Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example
"""

from __future__ import annotations

from pyramids.dataset import Dataset

dem_path = "Data/GIS/Hapi_GIS_Data/acc4000.tif"
SaveTo = "data/parameters/"
# %%
# create a raster typical to the DEM and fill its domain cells with one value
dem = Dataset.read_file(dem_path)

K = 1
dem.fill(K, path=SaveTo + "11_K_muskingum.tif")
# %%
X = 0.2

dem.fill(X, path=SaveTo + "12_X_muskingum.tif")
