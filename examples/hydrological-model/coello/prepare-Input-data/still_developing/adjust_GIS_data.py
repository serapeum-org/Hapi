"""Created on Tue Jul 17 06:17:00 2018.

@author: Mostafa
"""

# %links
from IPython import get_ipython  # to reset the variable explorer each time

get_ipython().magic("reset -f")
import os

os.chdir("F:/02Case studies/Coello/HAPI/Data")

# import sys
# sys.path.append("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model/HBV_distributed/function")

import gdal

# functions
import hapi.gis.raster as GIS

# %library
# import numpy as np


# %%
FD = gdal.Open("00inputs/GIS/4000/fd4000.tif")
fd_A = FD.ReadAsArray()
# manual adjusting
fd_adjusted = fd_A
# create the raster
# src=FD
# array=fd_adjusted
# path="00inputs/GIS/4000/fd_adj4000.tif"
#
# prj=src.GetProjection()
# cols=src.RasterXSize
# rows=src.RasterYSize
# gt=src.GetGeoTransform()
# noval=src.GetRasterBand(1).GetNoDataValue()
# if pixel_type==1:
#    outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_Float32)
# elif pixel_type==2:
#    outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_UInt16)
#
# outputraster.SetGeoTransform(gt)
# outputraster.SetProjection(prj)
# outputraster.GetRasterBand(1).SetNoDataValue(noval)
# outputraster.GetRasterBand(1).Fill(noval)
# outputraster.GetRasterBand(1).WriteArray(array)
# outputraster.FlushCache()
# outputraster = None

GIS.RasterLike(FD, fd_adjusted, "00inputs/GIS/4000/fd_adj4000.tif", 3)
