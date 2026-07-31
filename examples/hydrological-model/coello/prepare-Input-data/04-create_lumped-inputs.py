"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example
"""

import numpy as np
from Hapi.rrm.inputs import Inputs as IN

rpath = "examples/hydrological-model/data/meteo_data/meteodata_prepared/"
Path = f"{rpath}/temp-lumped-example"
SaveTo = f"{rpath}/lumped_temp.txt"

data = IN.createLumpedInputs(Path)
np.savetxt(SaveTo, data, fmt="%7.2f")
