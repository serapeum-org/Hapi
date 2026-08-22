r"""Combine the per-variable NetCDFs into one file holding all three drivers.

Reads `prec.nc`, `temp.nc` and `evap.nc` one at a time and writes `meteo.nc`, whose variables
are named `precipitation` / `temperature` / `evapotranspiration`.

The first file seeds the container; the other two are pulled in with `NetCDF.add_variable`,
which copies the MDArray across, and each is renamed on arrival. Nothing touches disk until
`to_file`, so the source files are left as they are.

`meteo.nc` is a committed fixture, so this only runs when invoked directly:

    pixi run -e dev python tests/rrm/data/coello/combine_netcdf.py

Run it from the repository root, or pass `--root` to point at the directory holding the four
files. It exits non-zero if the written file does not match its sources.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pyramids.netcdf import NetCDF

DEFAULT_ROOT = Path("tests/rrm/data/coello")

#: output variable name -> the single-variable NetCDF holding it
SOURCES = {
    "precipitation": "prec.nc",
    "temperature": "temp.nc",
    "evapotranspiration": "evap.nc",
}


def combine(root: Path, out_path: Path) -> None:
    """Write the three per-variable files into one multi-variable NetCDF.

    Args:
        root: Directory holding `prec.nc`, `temp.nc` and `evap.nc`.
        out_path: File to write.
    """
    (seed_name, seed_file), *rest = SOURCES.items()
    combined = NetCDF.read_file(str(root / seed_file))
    combined.rename_variable(combined.variable_names[0], seed_name)

    for name, file_name in rest:
        source = NetCDF.read_file(str(root / file_name))
        combined.add_variable(source)
        combined.rename_variable(source.variable_names[0], name)

    out_path.unlink(missing_ok=True)
    combined.to_file(str(out_path))
    print(f"written    : {out_path}")


def verify(root: Path, out_path: Path) -> bool:
    """Check the written file carries the source arrays unchanged.

    Args:
        root: Directory holding the per-variable sources.
        out_path: The combined file to check.

    Returns:
        bool: True when every variable matches its source element for element.
    """
    nc = NetCDF.read_file(str(out_path))
    print(f"variables  : {sorted(nc.variable_names)}")
    print(f"dimensions : {nc.dimension_sizes}  epsg={nc.epsg}")
    print(f"geo        : {nc.global_attributes['GeoTransform']}")

    ok = True
    for name, file_name in SOURCES.items():
        written = np.asarray(nc.get_variable(name).read_array())
        source = NetCDF.read_file(str(root / file_name))
        expected = np.asarray(
            source.get_variable(source.variable_names[0]).read_array()
        )
        identical = np.array_equal(written, expected)
        ok &= identical
        print(f"  {name:20s} identical to {file_name}: {identical}")
    return bool(ok)


def main(argv: list[str] | None = None) -> int:
    """Combine the fixtures and verify the result.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        int: 0 when the written file matches its sources, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"directory holding the per-variable NetCDFs (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="file to write (default: <root>/meteo.nc)",
    )
    args = parser.parse_args(argv)
    out_path = args.out or args.root / "meteo.nc"

    combine(args.root, out_path)
    return 0 if verify(args.root, out_path) else 1


if __name__ == "__main__":
    sys.exit(main())
