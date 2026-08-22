"""Created on Wed Jul 04 23:03:55 2018.

@author: Mostafa
This function is used during the calibration of the model to distribute generated parameters by the calibration
algorithm into a defined HRUs by a classified raster
"""

# %library
import numpy as np
from pyramids.dataset import Dataset

from hapi.rrm.parameters import Parameters as DP

# data path
path = "/data/"
# %% Two Lumped Parameter [K1, Perc]
# number of parameters in the rainfall runoff model
no_parameters = 12

# The classified raster defining the HRUs. `Parameters` takes a pyramids Dataset, and the
# HRU shape is read off it, so it is handed to the constructor rather than to the method.
soil_type = Dataset.read_file(path + "soil_classes.tif")
soil_A = soil_type.read_array()

no_lumped_par = 2
lumped_par_pos = [6, 8]

rows = soil_type.rows
cols = soil_type.columns
noval = np.float32(soil_type.no_data_value[0])

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

# The raster, the counts and the K bounds are constructor arguments; the method takes only
# the flat parameter vector the calibration produced.
distributor = DP(
    soil_type,
    no_parameters,
    no_lumped_par=no_lumped_par,
    lumped_par_pos=lumped_par_pos,
    hru=True,
    k_upper_bound=1,
    k_lower_bound=50,
)
par_2lumped = distributor.hydrologic_response_units(par_g)

# %% One Lumped Parameter [K1]

no_lumped_par = 1
lumped_par_pos = [6]

# generate no of parameters equals to model par* no of soil types
par_g = np.random.random(no_elem * (no_parameters - no_lumped_par))
par_g = np.append(par_g, 55)

distributor = DP(
    soil_type,
    no_parameters,
    no_lumped_par=no_lumped_par,
    lumped_par_pos=lumped_par_pos,
    hru=True,
    k_upper_bound=1,
    k_lower_bound=50,
)
par_1lump = distributor.hydrologic_response_units(par_g)

# %% HRU without lumped Parameter

no_lumped_par = 0
lumped_par_pos = []

# generate no of parameters equals to model par* no of soil types
par_g = np.random.random(no_elem * (no_parameters - no_lumped_par))

distributor = DP(
    soil_type,
    no_parameters,
    no_lumped_par=no_lumped_par,
    lumped_par_pos=lumped_par_pos,
    hru=True,
    k_upper_bound=1,
    k_lower_bound=50,
)
par_tot = distributor.hydrologic_response_units(par_g)
