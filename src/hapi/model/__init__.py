"""Assembly: the objects a run is built from, before it is validated.

`Catchment` is assembled by successive `read_*()` calls rather than through its constructor;
`Lake` is the second lumped model the lake-aware entry points take beside it.

Nothing in `hapi.engine` imports this package. The engine states its requirements through
`hapi.simulation.protocols`, which `Catchment` satisfies structurally, so the builder and the
engine are siblings rather than a chain -- and the two can be read, tested and changed apart.
"""

from __future__ import annotations

from hapi.model.catchment import Catchment
from hapi.model.lake import Lake

__all__ = ["Catchment", "Lake"]
