r"""Regenerate the Coello NetCDF fixtures from the raster folders.

Packs `prec/`, `temp/` and `evap/` into one NetCDF each, then merges the three into
`meteo.nc`, whose variables are named `precipitation` / `temperature` /
`evapotranspiration` so a reader can ask for them by name.

All four files are committed, so this only runs when invoked directly:

    pixi run -e dev python tests/rrm/data/coello/convert_meteo_inputs_to_netcdf.py

Run it from the repository root, or pass `--root` to point elsewhere. It exits non-zero if
what it wrote does not match the rasters it came from.

The Coello rasters are named `0_Tair2m_..._2009.01.01.tif` -- a `%Y.%m.%d` date with dots.
Rhine's are `0_Temp_ECMWF_ERA_Interim_C_daily_1979_1_1.tif`: underscores, and a month and
day that are not zero-padded, so both the regex and the format have to say so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hapi.inputs import METEO_VARIABLES, MeteoInputs

DEFAULT_ROOT = Path("tests/rrm/data/coello")

#: driver -> the folder of rasters holding it. The NetCDF takes the folder's name.
FOLDERS = {
    "precipitation": "prec",
    "temperature": "temp",
    "evapotranspiration": "evap",
}

#: How the date sits in the file names. The format is inferred when omitted, which covers
#: `2009.01.01` and `20090101`; pass it for anything else, e.g. Rhine's
#: `r"\d{4}_\d{1,2}_\d{1,2}"` with `"%Y_%m_%d"`.
READER = dict(regex_string=r"\d{4}.\d{2}.\d{2}", file_name_data_fmt="%Y.%m.%d")

#: GDAL lists the directory on every open to look for sidecars. Against 14,823 files on a
#: NAS that is a remote listing per raster -- 369 ms instead of 18 ms. It also stops GDAL
#: finding .aux.xml / world files / .ovr, so it is opt-in rather than the library default.
#: Add `gdal_env=FAST_GDAL` to READER when the folder has no sidecars.
FAST_GDAL = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_PAM_ENABLED": "NO"}


def regenerate(root: Path) -> Path:
    """Pack each raster folder, then merge the three into one file.

    Args:
        root: Directory holding the `prec` / `temp` / `evap` folders.

    Returns:
        Path: The merged file.
    """
    packed = {}
    for name, folder in FOLDERS.items():
        packed[name] = MeteoInputs.raster_folder_to_netcdf(
            root / folder, root / f"{folder}.nc", **READER
        )
        print(f"packed     : {folder}/ -> {packed[name].name}  ({name})")

    combined = MeteoInputs.combine_netcdf_files(*packed.values(), root / "meteo.nc")
    print(
        f"combined   : {', '.join(p.name for p in packed.values())} -> {combined.name}"
    )
    return combined


def verify(root: Path, combined: Path) -> bool:
    """Check the merged file carries what the rasters hold.

    Args:
        root: Directory holding the raster folders.
        combined: The merged file to check.

    Returns:
        bool: True when every driver survived packing and merging unchanged.
    """
    from_rasters = MeteoInputs.from_rasters(
        *(root / folder for folder in FOLDERS.values()), **READER
    )
    merged = MeteoInputs.from_netcdf(
        combined,
        precipitation="precipitation",
        temperature="temperature",
        evapotranspiration="evapotranspiration",
    )

    ok = True
    for name in METEO_VARIABLES:
        identical = np.array_equal(
            getattr(merged, name), getattr(from_rasters, name), equal_nan=True
        )
        ok &= identical
        print(f"  {name:20s} identical to {FOLDERS[name]}/: {identical}")
    return bool(ok)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the fixtures and verify them.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        int: 0 when what was written matches the rasters, 1 otherwise.
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
