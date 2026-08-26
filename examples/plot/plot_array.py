"""Plot a raster with pyramids/cleopatra.

The old ``hapi.plot`` module moved to the ``cleopatra`` package. Rasters are
read with ``pyramids.dataset.Dataset`` and plotted either with the
``Dataset.plot`` facade or directly with
``cleopatra.glyphs.gridded.array_glyph.ArrayGlyph``.

cleopatra 0.30 replaced the loose styling keywords with typed group objects, so
``color_scale``/``gamma`` are now ``color=ColorScaling...``, ``display_cell_value``/
``num_size``/``background_color_threshold`` are ``cells=CellValues(...)``, and the
point-overlay keywords are ``points=PointOverlay(...)``. The figure, colormap and
colour-bar keywords (``figsize``, ``cmap``, ``vmin``/``vmax``, ``cbar_*``,
``ticks_spacing``) are unchanged.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from cleopatra.glyphs.gridded.array_glyph import PointOverlay
from cleopatra.styling.params import CellValues
from cleopatra.styling.scaling import ColorScaling
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
src.plot(band=0, color=ColorScaling.linear(), cmap="terrain", ticks_spacing=500)
# %% power scale
# the lower the gamma, the more of the color bar is given to the low values
for gamma in [0.5, 0.4, 0.2]:
    src.plot(
        band=0,
        color=ColorScaling.power(gamma=gamma),
        cmap="terrain",
        ticks_spacing=500,
        title=f"gamma = {gamma}",
    )
# %% SymLogNorm scale
src.plot(
    band=0,
    color=ColorScaling.sym_log(threshold=0.0001, scale=0.001),
    cmap="terrain",
    ticks_spacing=500,
)
# %% midpoint scale
src.plot(
    band=0,
    color=ColorScaling.midpoint(at=20),
    cmap="terrain",
    ticks_spacing=500,
)
# %%
src = Dataset.read_file(RasterBPath)
arr = src.read_array(band=0)
# %% cell value labels
src.plot(
    band=0,
    cells=CellValues(show=True, size=8, background_threshold=None),
    ticks_spacing=10,
)
# %% display points on the map
# read the points (x/y coordinates in the same CRS as the raster), convert
# them to array indices, and pass them as a [value, row, col] array wrapped in
# a PointOverlay, which also carries the marker and value-label styling
points = pd.read_csv(pointsPath)
loc = src.map_to_array_coordinates(points)
points_arr = np.column_stack([points["id"].to_numpy(), loc])
src.plot(
    band=0,
    points=PointOverlay(
        points_arr,
        color="blue",
        size=100,
        label_color="green",
        label_size=20,
    ),
    cells=CellValues(show=True, size=8),
    ticks_spacing=10,
)
