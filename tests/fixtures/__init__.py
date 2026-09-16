"""Shared pytest fixtures.

Not tests. `tests/conftest.py` star-imports `coello`, which star-imports `catchment` and
`calibration`, so every fixture defined here is visible to every test.
"""

from __future__ import annotations
