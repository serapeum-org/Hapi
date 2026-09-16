"""Assembly: the objects a run is built from, before it is validated.

`Catchment` is assembled by successive `read_*()` calls rather than through its constructor;
`Lake` is the second lumped model the lake-aware entry points take beside it.

Nothing in `hapi.engine` imports this package. The engine states its requirements through
`hapi.simulation.protocols`, which `Catchment` satisfies structurally, so the builder and the
engine are siblings rather than a chain -- and the two can be read, tested and changed apart.

Both classes are re-exported, and resolved on first use rather than at import, so
`import hapi.model.lake` does not load the builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hapi.model.catchment import Catchment
    from hapi.model.lake import Lake

#: Each re-exported name and the module that defines it.
_EXPORTS: dict[str, str] = {
    "Catchment": "hapi.model.catchment",
    "Lake": "hapi.model.lake",
}

__all__ = ["Catchment", "Lake"]

if not TYPE_CHECKING:
    # Defined only at run time: a module-level `__getattr__` visible to mypy would type every
    # unknown name as `Any` and hide a typo in `from hapi.model import ...`.
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
