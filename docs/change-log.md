# Change Log

## 2.0.0 (Unreleased)

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
