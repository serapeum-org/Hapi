# Installation

## Stable release

Please install Hapi in a Virtual environment so that its requirements don't tamper with your system's python
`Hapi` works with all Python versions

## conda
the easiest way to install `Hapi` is using `conda` package manager. `Hapi` is available in the [conda-forge](https://conda-forge.org/) channel. To install
you can use the following command:

```shell
conda install -c conda-forge hapi
```
If this works it will install Hapi with all dependencies including Python and gdal,
and you skip the rest of the installation instructions.


## Installing Python and gdal dependencies

The main dependencies for Hapi are an installation of Python 2.7+, and gdal

## Installing Python

For Python we recommend using the Anaconda Distribution for Python 3, which is available
for download from https://www.anaconda.com/download/. The installer gives the option to
add `python` to your `PATH` environment variable. We will assume in the instructions
below that it is available in the path, such that `python`, `pip`, and `conda` are
all available from the command line.

Note that there is no hard requirement specifically for Anaconda's Python, but often it
makes installation of required dependencies easier using the conda package manager.

## Install as a conda environment

The easiest and most robust way to install Hapi is by installing it in a separate
conda environment. In the root repository directory there is an `environment.yml` file.
This file lists all dependencies. Either use the `environment.yml` file from the master branch
(please note that the master branch can change rapidly and break functionality without warning),
or from one of the releases {release}.

Run this command to start installing all Hapi dependencies:

```shell
conda env create -f environment.yml
```

This creates a new environment with the name `hapi`. To activate this environment in
a session, run:

```shell
activate hapi
```
For the installation of Hapi there are two options (from the Python Package Index (PyPI)
or from Github). To install a release of Hapi from the PyPI (available from release 2018.1):

```shell
pip install HAPI-Nile=={release}
```

## From sources

The sources for HapiSM can be downloaded from the
[Github repo](https://github.com/serapeum-org/Hapi).

You can either clone the public repository:

```shell
$ git clone git://github.com/serapeum-org/Hapi
```
Or download the [tarball](https://github.com/serapeum-org/Hapi/tarball/master):

```shell
$ curl -OJL https://github.com/serapeum-org/Hapi/tarball/master
```
Once you have a copy of the source, you can install it with:

```shell
$ python setup.py install
```


To install directly from GitHub (from the HEAD of the master branch):

+ `pip install git+https://github.com/serapeum-org/Hapi.git`

or from Github from a specific release:

+ `pip install git+https://github.com/serapeum-org/Hapi.git@{release}`

Now you should be able to start this environment's Python with `python`, try
`import Hapi` to see if the package is installed.


More details on how to work with conda environments can be found here:
https://conda.io/docs/user-guide/tasks/manage-environments.html


If you are planning to make changes and contribute to the development of Hapi, it is
best to make a git clone of the repository, and do a editable install in the location
of you clone. This will not move a copy to your Python installation directory, but
instead create a link in your Python installation pointing to the folder you installed
it from, such that any changes you make there are directly reflected in your install.

+ `git clone https://github.com/serapeum-org/Hapi.git`
+ `cd Hapi`
+ `activate Hapi`
+ `pip install -e .`

Alternatively, if you want to avoid using `git` and simply want to test the latest
version from the `master` branch, you can replace the first line with downloading
a zip archive from GitHub: https://github.com/serapeum-org/Hapi/archive/master.zip
[libraries.io](https://libraries.io/github/serapeum-org/Hapi).

## Install using pip

Besides the recommended conda environment setup described above, you can also install
Hapi with `pip`. For the more difficult to install Python dependencies, it is best to
use the conda package manager:

```shell
conda install numpy scipy gdal netcdf4 pyproj
```

you can check [libraries.io](https://libraries.io/github/serapeum-org/Hapi). to check versions of the libraries


Then install a release {release} of Hapi (available from release 2018.1) with pip:

```shell
pip install HAPI-Nile=={release}
````

## Check if the installation is successful

To check it the install is successful, go to the examples directory and run the following command:

```shell
`python -m Hapi.*******`
```

This should run without errors.


> **Note:**

      This documentation was generated on today

      Documentation for the development version:
      https://Hapi.readthedocs.org/en/latest/

      Documentation for the stable version:
      https://Hapi.readthedocs.org/en/stable/
