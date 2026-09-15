"""What the model reads, and the readers that build it.

Five concerns that shared one 2,200-line module: `rasters` decides the order a folder is read
in, `meteo` holds the three drivers, `network` the routing grid, `geometry` the channel
rasters the flood model needs, and `preparation` the utilities that align rasters to a DEM
and pull parameters from the global datasets. `dem` is here as a way station -- flow-direction
work belongs in `digital-rivers`.

Every public name is re-exported, so `from hapi.inputs import MeteoInputs` reads the same as
it did when this was a single module.
"""

from __future__ import annotations

from hapi.inputs.dem import DEM
from hapi.inputs.geometry import RIVER_GEOMETRY_RASTERS, RiverGeometry
from hapi.inputs.meteo import METEO_VARIABLES, MeteoInputs
from hapi.inputs.network import D8_CODES, FlowNetwork
from hapi.inputs.preparation import PARAMETERS_LIST, Inputs
from hapi.inputs.rasters import read_rasters

__all__ = [
    "D8_CODES",
    "DEM",
    "METEO_VARIABLES",
    "PARAMETERS_LIST",
    "RIVER_GEOMETRY_RASTERS",
    "FlowNetwork",
    "Inputs",
    "MeteoInputs",
    "RiverGeometry",
    "read_rasters",
]
