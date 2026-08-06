"""Created on Wed Jul 04 23:03:55 2018.

@author: Mostafa
This function is used durin the calibration of the model to distribute generated parameters by the calibation
algorithm into a defined HRUs by a classified raster
"""

# %library
import numpy as np
from hapi.rrm.parameters import Parameters as DP
from osgeo import gdal

# data path
path = "/data/"
# %% Two Lumped Parameter [K1, Perc]
# number of parameters in the rainfall runoff model
no_parameters = 12

soil_type = gdal.Open(path + "soil_classes.tif")
soil_A = soil_type.ReadAsArray()

no_lumped_par = 2
lumped_par_pos = [6, 8]

rows = soil_type.RasterYSize
cols = soil_type.RasterXSize
noval = np.float32(soil_type.GetRasterBand(1).GetNoDataValue())

values = list(
    set(
        [
            int(soil_A[i, j])
            for i in range(rows)
            for j in range(cols)
            if soil_A[i, j] != noval
        ]
    )
)
no_elem = len(values)
# generate no of parameters equals to model par* no of soil types
par_g = np.random.random(no_elem * (no_parameters - no_lumped_par))
par_g = np.append(par_g, 55)
par_g = np.append(par_g, 66)

par_2lumped = DP.hydrologic_response_units(
    par_g, soil_type, no_parameters, no_lumped_par, lumped_par_pos, kub=1, klb=0.5
)

# %% One Lumped Parameter [K1]

no_lumped_par = 1
lumped_par_pos = [6]

# generate no of parameters equals to model par* no of soil types
par_g = np.random.random(no_elem * (no_parameters - no_lumped_par))
par_g = np.append(par_g, 55)

par_1lump = DP.hydrologic_response_units(
    par_g, soil_type, no_parameters, no_lumped_par, lumped_par_pos, kub=1, klb=0.5
)

# %% HRU without lumped Parameter

no_lumped_par = 0
lumped_par_pos = []

# generate no of parameters equals to model par* no of soil types
par_g = np.random.random(no_elem * (no_parameters - no_lumped_par))

par_tot = DP.hydrologic_response_units(
    par_g, soil_type, no_parameters, no_lumped_par, lumped_par_pos, kub=1, klb=0.5
)
