"""Fitting a model to observations.

`search` drives the Harmony Search optimiser and scores each trial against the gauge record.
`distribution` is what stands between the two: it maps the flat parameter vector the optimiser
produces onto the 3D per-cell array the model reads, honouring lumped parameters and HRUs.

`distribution` was `hapi.rrm.parameters`, which shared a name with the FigShare downloader
now at `hapi.data.figshare` and nothing else. The names say which is which.

`search`'s public names are re-exported here, so `from hapi.calibration import Calibration`
reads the same as it did when this was a single module.
"""

from __future__ import annotations

from hapi.calibration.search import Calibration, ObjectiveFunctionArityError

__all__ = ["Calibration", "ObjectiveFunctionArityError"]
