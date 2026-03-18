"""Hapi."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError  # type: ignore
from importlib.metadata import version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__author__ = "Mostafa Farrag"
__email__ = "moah.farag@gmail.com"
__docformat__ = "restructuredtext"


__doc__ = """
Hapi - Hydrological library for Python
=====================================================================

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
