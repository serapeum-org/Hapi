"""Created on Wed Jul 11 05:32:05 2018.

@author: Mostafa
"""

# %links
# from IPython import get_ipython  # to reset the variable explorer each time

# get_ipython().magic("reset -f")
import os

# Comp =
os.chdir(
    "F:/04Private/02Research/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model/HBV_distributed/function/examples"
)

# import sys
# sys.path.append("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model/HBV_distributed/function")

# %library
import numpy as np
from pyramids.dataset import Dataset
from scipy.stats import norm

# functions
from hapi.rrm.parameters import Parameters as DP

# from hapi.catchment import GISCatchment as gc

# import matplotlib.pyplot as plt


# %% to modify the basins raster
# path="C:/Users/Mostafa/Desktop/delineation/Clipped/proj/basins.tif"
# pathout="mask.tif"
# basins=Dataset.read_file(path)
# gc.DeleteBasins(basins,pathout)
# %%
path = "C:/Users/Mostafa/Desktop/delineation/HRU/HAND/"
DEM = Dataset.read_file(path + "DEM.tif")
# dem_A=DEM.ReadAsArray()
# no_val=np.float32(DEM.GetRasterBand(1).GetNoDataValue())
# rows=DEM.RasterYSize
# cols=DEM.RasterXSize
FD = Dataset.read_file(path + "FD.tif")
# fd_A=fd.ReadAsArray()
# fd_val=[int(fd_A[i,j]) for i in range(rows) for j in range(cols) if fd_A[i,j] != no_val]
FPL = Dataset.read_file(path + "fpl.tif")
# fpl_A=FPL.ReadAsArray()
River = Dataset.read_file(path + "river.tif")
# river_A=River.ReadAsArray()
# slope
Slope = Dataset.read_file(path + "slope.tif")
Slope_A = Slope.read_array(band=0)
no_val_slope = np.float32(Slope.no_data_value[0])
# %% calculate HAND and DTND
HAND, DTND = DP.hru_hand(DEM, FD, FPL, River)
# %% calculate the cdf for
mean_slope = np.mean(Slope_A[Slope_A != no_val_slope])
stv_slope = np.std(Slope_A[Slope_A != no_val_slope])

mean_HAND = np.nanmean(HAND)
stv_HAND = np.nanstd(HAND)

vals = norm.pdf([0.001, 0.5, 0.999])
