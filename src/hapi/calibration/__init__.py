"""Fitting a model to observations.

`search` drives the Harmony Search optimiser and scores each trial against the gauge record.
`distribution` is what stands between the two: it maps the flat parameter vector the optimiser
produces onto the 3D per-cell array the model reads, honouring lumped parameters and HRUs.

`distribution` was `hapi.rrm.parameters`, which shared a name with the FigShare downloader
now at `hapi.data.figshare` and nothing else. The names say which is which.

Every public name the old `hapi/calibration.py` module had -- `Calibration`, the arity error,
and the three message constants -- is re-exported here, so imports written against that
module still work.
"""

from __future__ import annotations

from hapi.calibration.search import (
    COLUMNS_MISMATCH_ERROR,
    OBJECTIVE_FN_ARGS_ERROR,
    ROWS_MISMATCH_ERROR,
    Calibration,
    ObjectiveFunctionArityError,
)

__all__ = [
    "COLUMNS_MISMATCH_ERROR",
    "OBJECTIVE_FN_ARGS_ERROR",
    "ROWS_MISMATCH_ERROR",
    "Calibration",
    "ObjectiveFunctionArityError",
]
