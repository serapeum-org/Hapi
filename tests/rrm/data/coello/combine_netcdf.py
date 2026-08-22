r"""Regenerate the Coello NetCDF fixtures from the raster folders.

Packs each driver's folder of dated GeoTIFFs into its own NetCDF, then merges the three into
`meteo.nc`, whose variables are named `precipitation` / `temperature` / `evapotranspiration`.

All four files are committed, so this only runs when invoked directly:

    pixi run -e dev python tests/rrm/data/coello/combine_netcdf.py

Run it from the repository root, or pass `--root` to point at the Coello data directory. It
exits non-zero if the regenerated files do not match the rasters they came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hapi.inputs import METEO_VARIABLES, MeteoInputs

DEFAULT_ROOT = Path("tests/rrm/data/coello")

#: driver name -> the raster folder holding it
SOURCES = {
    "precipitation": "prec",
    "temperature": "temp",
    "evapotranspiration": "evap",
}

RASTER_KWARGS = dict(regex_string=r"\d{4}.\d{2}.\d{2}", file_name_data_fmt="%Y.%m.%d")


def regenerate(root: Path) -> Path:
    """Pack each raster folder into a NetCDF, then merge the three.

    Args:
        root: Directory holding the `prec` / `temp` / `evap` folders.

    Returns:
        Path: The combined file.
    """
    for name, folder in SOURCES.items():
        out = MeteoInputs.raster_folder_to_netcdf(
            root / folder, root / f"{folder}.nc", **RASTER_KWARGS
        )
        print(f"packed     : {folder}/ -> {out.name}  ({name})")

    combined = MeteoInputs.combine_netcdf_files(
        root / "prec.nc", root / "temp.nc", root / "evap.nc", root / "meteo.nc"
    )
    print(f"combined   : {combined.name}")
    return combined


def verify(root: Path, combined: Path) -> bool:
    """Check the regenerated files still carry what the rasters hold.

    Args:
        root: Directory holding the raster folders.
        combined: The merged file to check.

    Returns:
        bool: True when every driver matches the folder it was packed from.
    """
    from_rasters = MeteoInputs.from_rasters(
        root / "prec", root / "temp", root / "evap", **RASTER_KWARGS
    )
    from_file = MeteoInputs.from_netcdf(
        combined,
        precipitation="precipitation",
        temperature="temperature",
        evapotranspiration="evapotranspiration",
    )

    ok = True
    for name in METEO_VARIABLES:
        identical = np.array_equal(
            getattr(from_file, name), getattr(from_rasters, name), equal_nan=True
        )
        ok &= identical
        print(f"  {name:20s} identical to its rasters: {identical}")
    return bool(ok)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the fixtures and verify them.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        int: 0 when the regenerated files match their rasters, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"directory holding the raster folders (default: {DEFAULT_ROOT})",
    )
    args = parser.parse_args(argv)

    combined = regenerate(args.root)
    return 0 if verify(args.root, combined) else 1


if __name__ == "__main__":
    sys.exit(main())
