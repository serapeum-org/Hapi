# Hapi - Hydrological library for Python

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4686056.svg)](https://doi.org/10.5281/zenodo.4686056)
[![PyPI](https://img.shields.io/pypi/v/Hapi-nile)](https://pypi.org/project/HAPI-Nile/)
[![Conda](https://img.shields.io/conda/v/conda-forge/hapi?label=conda-forge)](https://anaconda.org/conda-forge/hapi)
[![Python](https://img.shields.io/pypi/pyversions/hapi-nile)](https://pypi.org/project/HAPI-Nile/)
[![Downloads](https://static.pepy.tech/badge/hapi-nile)](https://pepy.tech/project/hapi-nile)
[![Platforms](https://anaconda.org/conda-forge/hapi/badges/platforms.svg)](https://anaconda.org/conda-forge/hapi)

![Hapi](img/Hapi.png){ width="400" }

**Hapi** is a Python package providing a fast and flexible way to build hydrological models with
different spatial representations (lumped, semi-distributed, and conceptual distributed) using HBV96.
The package allows developers to change the structure of the defined conceptual model or to provide
their own model. It contains two routing functions: Muskingum-Cunge and MAXBAS triangular function.

![Model Structure](img/Picture1.png){ width="400" }

## Main Features

- Modified version of HBV96 hydrological model (Bergstrom, 1992) with 15 parameters when
  considering snow processes, and 10 parameters without snow, plus 2 Muskingum routing parameters
- GIS modules to prepare meteorological inputs and perform preprocessing (align rasters with the
  DEM), plus methods to manipulate distributed data (rasters, NetCDF, shapefiles)
- Sensitivity analysis module based on One-At-a-Time (OAT) and Sobol interaction analysis
  (Rusli et al., 2015)
- Statistical module with interpolation methods, frequency analysis distributions, and Maximum
  Likelihood parameter estimation
- Visualization module for animating distributed model results and meteorological inputs
- Optimization module for calibrating the model using Harmony Search

The recent version integrates global hydrological parameters from Beck et al. (2016) to reduce
model complexity and parameter uncertainty.

## IHE-Delft Sessions

- April 14-15: Two-day session for Masters and PhD students at IHE-Delft —
  [Day 1](https://youtu.be/HbmUdN9ehSo), [Day 2](https://youtu.be/m7kHdOFQFIY)

## Citation

For using Hapi please cite Farrag et al. (2021) and Farrag & Corzo (2021).

## References

Farrag, M. & Corzo, G. (2021) MAfarrag/Hapi: Hapi. doi:10.5281/ZENODO.4662170

Farrag, M., Perez, G. C. & Solomatine, D. (2021) Spatio-Temporal Hydrological Model Structure
and Parametrization Analysis. J. Mar. Sci. Eng. 9(5), 467. doi:10.3390/jmse9050467

Beck, H. E. et al. (2016) Global-scale regionalization of hydrologic model parameters.
doi:10.1002/2015WR018247

Bergstrom, S. (1992) The HBV model - its structure and applications. SMHI RH 4(4), 35.

Rusli, S. R., Yudianto, D. & Liu, J. (2015) Effects of temporal variability on HBV model
calibration. Water Sci. Eng. 8(4), 291-300. doi:10.1016/j.wse.2015.12.002
