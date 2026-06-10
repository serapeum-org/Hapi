"""Plot a raster with pyramids/cleopatra.

The old ``Hapi.plot`` module moved to the ``cleopatra`` package. Rasters are
read with ``pyramids.dataset.Dataset`` and plotted either with the
``Dataset.plot`` facade or directly with ``cleopatra.array_glyph.ArrayGlyph``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from pyramids.dataset import Dataset

# %% Paths
RasterAPath = "examples/data/GIS/Hapi_GIS_Data/dem_100_f.tif"
RasterBPath = "examples/data/GIS/Hapi_GIS_Data/acc4000.tif"
pointsPath = "examples/GIS/data/points.csv"
# %%
# read the raster with pyramids
src = Dataset.read_file(RasterAPath)
# using all the default parameters, you can directly plot the Dataset
src.plot(band=0)
# %% figure options
src.plot(band=0, figsize=(8, 8), title="DEM", title_size=15)
# %% color bar options
src.plot(
    band=0,
    cbar_length=0.75,
    cbar_orientation="vertical",
    cbar_label_size=12,
    cbar_label="Elevation",
    cbar_label_rotation=-80,
    ticks_spacing=500,
)
# %% color scales
# linear scale
src.plot(band=0, color_scale="linear", cmap="terrain", ticks_spacing=500)
# %% power scale
# the lower the gamma, the more of the color bar is given to the low values
for gamma in [0.5, 0.4, 0.2]:
    src.plot(
        band=0,
        color_scale="power",
        gamma=gamma,
        cmap="terrain",
        ticks_spacing=500,
        title=f"gamma = {gamma}",
    )
# %% SymLogNorm scale
src.plot(
    band=0,
    color_scale="sym-lognorm",
    line_scale=0.001,
    line_threshold=0.0001,
    cmap="terrain",
    ticks_spacing=500,
)
# %% midpoint scale
src.plot(band=0, color_scale="midpoint", midpoint=20, cmap="terrain", ticks_spacing=500)
# %%
src = Dataset.read_file(RasterBPath)
arr = src.read_array(band=0)
# %% cell value labels
src.plot(
    band=0,
    display_cell_value=True,
    num_size=8,
    background_color_threshold=None,
    ticks_spacing=10,
)
# %% display points on the map
# read the points (x/y coordinates in the same CRS as the raster), convert
# them to array indices, and pass them as a [value, row, col] array
points = pd.read_csv(pointsPath)
loc = src.map_to_array_coordinates(points)
points_arr = np.column_stack([points["id"].to_numpy(), loc])
src.plot(
    band=0,
    points=points_arr,
    point_color="blue",
    point_size=100,
    pid_color="green",
    pid_size=20,
    display_cell_value=True,
    num_size=8,
    ticks_spacing=10,
)
