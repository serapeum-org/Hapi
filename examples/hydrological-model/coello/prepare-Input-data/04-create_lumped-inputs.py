"""Create lumped (catchment-average) inputs from distributed rasters.

Make sure the working directory is set to the root of the Hapi repo:
current_work_directory = Hapi/
"""

from __future__ import annotations

import numpy as np

from hapi.inputs import Inputs

rpath = "examples/hydrological-model/data/meteo_data/meteodata_prepared/"
Path = f"{rpath}/temp-lumped-example"
SaveTo = f"{rpath}/lumped_temp.txt"

data = Inputs.create_lumped_inputs(Path)
np.savetxt(SaveTo, data, fmt="%7.2f")
