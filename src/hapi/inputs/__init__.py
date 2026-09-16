"""What the model reads, and the readers that build it.

Five concerns that shared one 2,200-line module: `rasters` decides the order a folder is read
in, `meteo` holds the three drivers, `network` the routing grid, `geometry` the channel
rasters the flood model needs, and `preparation` the utilities that align rasters to a DEM
and pull parameters from the global datasets. `dem` is here as a way station -- flow-direction
work belongs in `digital-rivers`.

Every public name is re-exported, so `from hapi.inputs import MeteoInputs` reads the same as
it did when this was a single module. The re-exports are resolved on first use rather than at
import: `import hapi.inputs.dem` runs no reader it does not need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hapi.inputs.dem import DEM
    from hapi.inputs.geometry import RIVER_GEOMETRY_RASTERS, RiverGeometry
    from hapi.inputs.meteo import METEO_VARIABLES, MeteoInputs
    from hapi.inputs.network import D8_CODES, FlowNetwork
    from hapi.inputs.preparation import PARAMETERS_LIST, Inputs
    from hapi.inputs.rasters import read_rasters

#: Each re-exported name and the module that defines it.
_EXPORTS: dict[str, str] = {
    "D8_CODES": "hapi.inputs.network",
    "DEM": "hapi.inputs.dem",
    "METEO_VARIABLES": "hapi.inputs.meteo",
    "PARAMETERS_LIST": "hapi.inputs.preparation",
    "RIVER_GEOMETRY_RASTERS": "hapi.inputs.geometry",
    "FlowNetwork": "hapi.inputs.network",
    "Inputs": "hapi.inputs.preparation",
    "MeteoInputs": "hapi.inputs.meteo",
    "RiverGeometry": "hapi.inputs.geometry",
    "read_rasters": "hapi.inputs.rasters",
}

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

if not TYPE_CHECKING:
    # Defined only at run time: a module-level `__getattr__` visible to mypy would type every
    # unknown name as `Any` and hide a typo in `from hapi.inputs import ...`.
    import importlib

    def __getattr__(name: str) -> object:
        """Import a re-exported name from the module that defines it, on first access."""
        try:
            source = _EXPORTS[name]
        except KeyError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
        value = getattr(importlib.import_module(source), name)
        globals()[name] = value
        return value

    def __dir__() -> list[str]:
        """List the re-exported names alongside everything already in the namespace."""
        return sorted({*globals(), *__all__})
