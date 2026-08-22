r"""Merge the three per-driver NetCDFs into one file holding all three.

Thin wrapper over `MeteoInputs.combine_netcdf_files`. Reads `prec.nc`, `temp.nc` and
`evap.nc` -- which `convert_meteo_inputs_to_netcdf.py` produces from the raster folders --
and writes `meteo.nc`, whose variables are named `precipitation` / `temperature` /
`evapotranspiration` so a reader can ask for them by name.

All four files are committed, so this only runs when invoked directly:

    pixi run -e dev python tests/rrm/data/coello/combine_netcdf.py

Run it from the repository root, or pass `--root` to point at the Coello data directory. It
exits non-zero if the merged file does not carry what its sources hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hapi.inputs import METEO_VARIABLES, MeteoInputs

DEFAULT_ROOT = Path("tests/rrm/data/coello")

#: driver -> the single-variable NetCDF holding it
SOURCES = {
    "precipitation": "prec.nc",
    "temperature": "temp.nc",
    "evapotranspiration": "evap.nc",
}


def combine(root: Path) -> Path:
    """Merge the three per-driver files into one.

    Args:
        root: Directory holding the per-driver NetCDFs.

    Returns:
        Path: The merged file.

    Raises:
        FileNotFoundError: One of the sources has not been produced yet.
    """
    sources = [root / name for name in SOURCES.values()]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not found; run convert_meteo_inputs_to_netcdf.py first to pack the "
            "raster folders"
        )

    combined = MeteoInputs.combine_netcdf_files(*sources, root / "meteo.nc")
    print(f"combined   : {', '.join(SOURCES.values())} -> {combined.name}")
    return combined


def verify(root: Path, combined: Path) -> bool:
    """Check the merged file carries what its three sources hold.

    Args:
        root: Directory holding the per-driver NetCDFs.
        combined: The merged file to check.

    Returns:
        bool: True when every driver survived the merge unchanged.
    """
    sources = MeteoInputs.from_netcdf_files(*(root / name for name in SOURCES.values()))
    merged = MeteoInputs.from_netcdf(
        combined,
        precipitation="precipitation",
        temperature="temperature",
        evapotranspiration="evapotranspiration",
    )

    ok = True
    for name in METEO_VARIABLES:
        identical = np.array_equal(
            getattr(merged, name), getattr(sources, name), equal_nan=True
        )
        ok &= identical
        print(f"  {name:20s} identical to {SOURCES[name]}: {identical}")
    return bool(ok)


def main(argv: list[str] | None = None) -> int:
    """Merge the per-driver files and verify the result.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        int: 0 when the merged file matches its sources, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"directory holding the per-driver NetCDFs (default: {DEFAULT_ROOT})",
    )
    args = parser.parse_args(argv)

    combined = combine(args.root)
    return 0 if verify(args.root, combined) else 1


if __name__ == "__main__":
    sys.exit(main())
