# Installation

## Stable release

Please install Hapi in a virtual environment so that its requirements do not tamper with your system's Python.
Hapi requires **Python 3.11 or newer**.

The distribution is published as `HAPI-Nile`, and the import package is `hapi`.

## pip

```shell
pip install HAPI-Nile
```

To install a specific release:

```shell
pip install HAPI-Nile=={release}
```

## conda

`Hapi` is also available in the [conda-forge](https://conda-forge.org/) channel:

```shell
conda install -c conda-forge hapi
```

## Dependencies

Hapi installs its dependencies automatically. GDAL does **not** need to be installed
separately: it ships vendored inside the `pyramids-gis` wheel, which Hapi depends on.

You can check the versions of the libraries Hapi depends on at
[libraries.io](https://libraries.io/github/serapeum-org/Hapi).

## Install from GitHub

To install the latest development version (the HEAD of the `main` branch):

```shell
pip install git+https://github.com/serapeum-org/Hapi.git
```

Or a specific release:

```shell
pip install git+https://github.com/serapeum-org/Hapi.git@{release}
```

## From sources

The sources can be downloaded from the [GitHub repo](https://github.com/serapeum-org/Hapi).

Clone the public repository:

```shell
git clone https://github.com/serapeum-org/Hapi.git
```

Or download the [tarball](https://github.com/serapeum-org/Hapi/tarball/main):

```shell
curl -OJL https://github.com/serapeum-org/Hapi/tarball/main
```

Once you have a copy of the source, install it with:

```shell
cd Hapi
pip install .
```

## Development install

If you are planning to make changes and contribute to the development of Hapi, make a
git clone of the repository and do an editable install, so that any change you make is
directly reflected in your environment:

```shell
git clone https://github.com/serapeum-org/Hapi.git
cd Hapi
pip install -e .
```

### Using pixi

The repository is managed with [pixi](https://pixi.sh), which resolves the whole
environment (including the test and documentation tooling) from `pyproject.toml` and
`pixi.lock`. This is what CI runs, so it is the most reliable way to reproduce a
development environment:

```shell
pixi install -e dev
pixi run -e dev test-all
```

Useful tasks:

| Task | Command |
| --- | --- |
| Run the main test suite | `pixi run -e dev main` |
| Run the whole test suite | `pixi run -e dev test-all` |
| Type check | `pixi run -e dev mypy` |
| Serve the documentation locally | `pixi run -e docs mkdocs serve` |

## Check if the installation is successful

```shell
python -c "import hapi; print(hapi.__name__)"
```

This should run without errors.


> **Note:**

      The documentation is built with MkDocs and published to GitHub Pages:
      https://serapeum-org.github.io/Hapi
