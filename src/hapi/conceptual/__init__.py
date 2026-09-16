"""The lumped conceptual models, and the objects that configure one.

`base` declares the surface every model implements; `hbv`, `hbv_bergestrom92` and `hbv_lake`
implement it. `setup` holds what a run has to supply alongside the model -- the parameter
vector, the initial conditions, the catchment area -- and validates the combination before a
run starts.

The names `setup` defines are re-exported here, so `from hapi.conceptual import ParameterSet`
reads the same as it did when this was a single module. They are resolved on first use rather
than at import, so `import hapi.conceptual.base` does not run `setup`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hapi.conceptual.setup import (
        PARAMETER_COUNTS,
        ConceptualModelSetup,
        ParameterBounds,
        ParameterSet,
        parameter_count,
        validate_initial_cond,
        validate_parameter_count,
        validate_q_init,
    )

#: Each re-exported name and the module that defines it.
_EXPORTS: dict[str, str] = {
    "PARAMETER_COUNTS": "hapi.conceptual.setup",
    "ConceptualModelSetup": "hapi.conceptual.setup",
    "ParameterBounds": "hapi.conceptual.setup",
    "ParameterSet": "hapi.conceptual.setup",
    "parameter_count": "hapi.conceptual.setup",
    "validate_initial_cond": "hapi.conceptual.setup",
    "validate_parameter_count": "hapi.conceptual.setup",
    "validate_q_init": "hapi.conceptual.setup",
}

__all__ = [
    "PARAMETER_COUNTS",
    "ConceptualModelSetup",
    "ParameterBounds",
    "ParameterSet",
    "parameter_count",
    "validate_initial_cond",
    "validate_parameter_count",
    "validate_q_init",
]

if not TYPE_CHECKING:
    # Defined only at run time: a module-level `__getattr__` visible to mypy would type every
    # unknown name as `Any` and hide a typo in `from hapi.conceptual import ...`.
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
