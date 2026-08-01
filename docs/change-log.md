# Change Log

## Unreleased

### Breaking Changes

- Remove all direct `gdal`/`osgeo` usage from the package in favour of the pyramids `Dataset` API:
  - `Parameters` (spatial parameter distribution) now requires a pyramids `Dataset` instead of a
    `gdal.Dataset` and raises `TypeError` otherwise.
  - `DistributedRRM.Dist_HBV2` and `Parameters.hru_hand` take pyramids `Dataset` objects.
- `Catchment.plot_distributed_results` forwards the cleopatra `ArrayGlyph.animate` keyword
  arguments (`figsize`, `display_cell_value`, `ticks_spacing`, `interval`, `frame_label`,
  `cell_value_text_colors`, `color_scale="linear"|"power"|…`) — the old CamelCase kwargs
  (`Figsize`, `PlotNumbers`, `TicksSpacing`, `Gaugecolor`, `IDcolor`, `ColorScale=1`, …) are no
  longer accepted.
- `Catchment.save_animation(path, fps=2)` replaces the old
  `save_animation(video_format=, path=, save_frames=)` signature; the format is inferred from the
  file extension.
- Dependency floors raised: `pyramids-gis>=0.46.0`, `cleopatra>=0.26.1`, `statista>=0.8.0`,
  `pandas>=3.0.0`, `scipy>=1.17.0`, `matplotlib>=3.11.0`. The direct conda `gdal` pin is gone —
  GDAL is vendored inside the pyramids-gis wheel.
- `dev` and `docs` are PEP 735 `[dependency-groups]` and are no longer installable as pip extras
  (`pip install hapi-nile[dev]`).
- `HBV.simulate` now takes its arguments in the `BaseConceptualModel` order
  (`prec, temp, et, ll_temp, par, init_st, q_init, snow`) — previously `par` came before `ll_temp`,
  which also made `HBV` unusable through `Wrapper.Lumped`'s positional call.
- Attributes renamed to snake_case: `Catchment.FlowDirArr` → `flow_dir_arr`,
  `Catchment.FPLArr` → `fpl_arr`, and `Parameters.raster_A` → `raster_array`.

### Dependencies

- Migrate to pyramids-gis 0.46 (`MultiDataset` → `DatasetCollection`, exact band statistics,
  `align`/`crop` `inplace=` semantics) and cleopatra 0.26 (first-axis animation, `points` arrays,
  `ArrayGlyph.save_animation`).
- Migrate to statista 0.8 (`pearson_corr_coeff`, `r2`, `Sensitivity.one_at_a_time`/`sobol`).
- Replace black + isort + flake8 + pydocstyle with ruff; add bandit, gitleaks, checkov, nbstripout,
  shellcheck, package-wide mypy, a coverage floor, and a pixi lock staleness gate to pre-commit.

### Fixed

- Fix the no-data masks in `plot_distributed_results` (`np.isnan` instead of comparing against the
  no-data value, which never matched after `read_flow_acc` converts no-data cells to NaN); the
  masking now happens on a copy, so plotting no longer mutates the model result arrays.
- Fix `Parameters.hru_hand`: river cells are their own nearest drainage (HAND = 0), and the
  flow-tracing no longer crashes on the removed legacy `dem = dem(flow_direction)` call.

### Dev

- Add conda/pypi workflow
- Add github release workflow
- Add pypi release workflow
- Remove coverall
- Remove flake8 separate config file
- Move the main package files inside src directory
- Move the hydrodynamic model to separate repo (serapis)
- Move the plot module to the cleopatra package
- Replace the setup.py by pyproject.toml

### Parameters

- Remove the parameters from the package and retrieve them with the
  parameter package.
- Redesign the parameters module to separate the responsibility of each
  class (`Parameter`, `ParameterManager`, `FileManager`,
  `FigshareAPIClient`).
- Add CLI to download the parameters from the FigShare server
  (`list-parameter-names`, `download-parameter-set`,
  `download-parameters`).

### Conceptual Models

- Refactor the HBV Bergestrom 92, HBV Lake, and the HBV conceptual
  models into classes.
- Move unused HBV variants to the examples folder.

## 1.6.0 (2023-02-03)

- All attributes follow snake case naming convention
- Refactor all modules with pre-commit
- Add smoothDikeLevel, getReach and updateReach
- Bump up dependencies versions
- Move unnecessary functions to serapeum-utils

## 1.5.0 (2023-01-10)

- Hydraulic model can read chunked big zip file
- Fix CI
- Fix missing module (saint venant script and module)

## 1.4.0 (2022-12-27)

- Remove fiona and the reading file exception using fiona
- Unify reading results of rainfall-runoff model in the readRRMResults,
  ReadLaterals, ReadUSHydrographs
- Refactor code and change methods to camelCase
- Add hydrodynamic model 1D config file read function
- Simplify functions with too many parameters using decorator
- Add automatic PyPI build and publish GitHub Actions

## 1.3.5 (2022-12-27)

- Fix PyPI package names in the requirements.txt file for all internal
  packages
- Fix python version number
- Tests are all passing

## 1.3.4 (2022-12-27)

- Merge two functions readLaterals and readRRMProgression, rename
  RRMProgression to routedRRM

## 1.3.3 (2022-12-27)

- Use joblib to parallelize reading laterals in hydraulic model

## 1.3.2 (2022-12-26)

- Remove parameters from the package and retrieve them with the
  parameter package.
