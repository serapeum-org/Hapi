"""Hapi - Hydrological library for Python.

**Hapi** is a Python package providing fast and flexible, way to build distributed
hydrological model using lumped conceptual model

Main Features
-------------
Here are just a few of the things that pandas does well:

  - Easy handling of rasters data downloaded from global data and easy way to
    manipulate the data to arrange it to run the model
  - Easy calibration of the model using Harmony search method and Genetic Algorithms
  - flexible GIS function to process rasters interpolate values and georeference
   calculated discharge values to the correct place
"""
from __future__ import annotations

import importlib
import sys
import warnings
from importlib.abc import MetaPathFinder
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hapi-nile")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


class _HapiBackwardCompatFinder(MetaPathFinder):
    """Allow ``import Hapi`` / ``from Hapi.x import y`` on case-sensitive systems.

    Redirects any ``Hapi`` or ``Hapi.*`` import to ``hapi`` / ``hapi.*``
    and emits a DeprecationWarning so users know to update their code.
    """

    _migrating: bool = False

    def find_module(self, fullname: str, path: object = None) -> _HapiBackwardCompatFinder | None:
        if self._migrating:
            return None
        if fullname == "Hapi" or fullname.startswith("Hapi."):
            return self
        return None

    def load_module(self, fullname: str) -> object:
        if fullname in sys.modules:
            return sys.modules[fullname]

        new_name = "hapi" + fullname[4:]  # replace leading "Hapi"
        warnings.warn(
            f"Importing from '{fullname}' is deprecated. "
            f"Use '{new_name}' instead. "
            "The 'Hapi' package name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._migrating = True
        try:
            module = importlib.import_module(new_name)
        finally:
            self._migrating = False

        sys.modules[fullname] = module
        return module


sys.meta_path.insert(0, _HapiBackwardCompatFinder())
