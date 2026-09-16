"""Fitting a model to observations.

`search` drives the Harmony Search optimiser and scores each trial against the gauge record.
`distribution` is what stands between the two: it maps the flat parameter vector the optimiser
produces onto the 3D per-cell array the model reads, honouring lumped parameters and HRUs.

`distribution` was `hapi.rrm.parameters`, which shared a name with the FigShare downloader
now at `hapi.data.figshare` and nothing else. The names say which is which.

Every public name the old `hapi/calibration.py` module had -- `Calibration`, the arity error,
and the three message constants -- is re-exported here, so imports written against that
module still work. They are resolved on first use rather than at import: `search` imports
the builder and the engine, and `import hapi.calibration.distribution`, which needs
neither, must not pay for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hapi.calibration.search import (
        COLUMNS_MISMATCH_ERROR,
        OBJECTIVE_FN_ARGS_ERROR,
        ROWS_MISMATCH_ERROR,
        Calibration,
        ObjectiveFunctionArityError,
    )

#: Each re-exported name and the module that defines it.
_EXPORTS: dict[str, str] = {
    "COLUMNS_MISMATCH_ERROR": "hapi.calibration.search",
    "OBJECTIVE_FN_ARGS_ERROR": "hapi.calibration.search",
    "ROWS_MISMATCH_ERROR": "hapi.calibration.search",
    "Calibration": "hapi.calibration.search",
    "ObjectiveFunctionArityError": "hapi.calibration.search",
}

__all__ = [
    "COLUMNS_MISMATCH_ERROR",
    "OBJECTIVE_FN_ARGS_ERROR",
    "ROWS_MISMATCH_ERROR",
    "Calibration",
    "ObjectiveFunctionArityError",
]

if not TYPE_CHECKING:
    # Defined only at run time: a module-level `__getattr__` visible to mypy would type every
    # unknown name as `Any` and hide a typo in `from hapi.calibration import ...`.
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
