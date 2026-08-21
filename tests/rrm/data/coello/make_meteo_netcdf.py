r"""Build ``meteo.nc``: the three Coello drivers in one NetCDF, one variable each.

Run once from the repo root to regenerate the fixture:

    pixi run -e dev python tests/rrm/data/coello/make_meteo_netcdf.py

The cubes come from the raster folders, which are the source of truth for the Coello fixture, so
``meteo.nc`` carries the same values as ``prec.nc`` / ``temp.nc`` / ``evap.nc`` and the same
georeferencing as the rasters themselves.

``DatasetCollection.to_netcdf`` names each variable after the band it came from, and band names
survive a GeoTIFF round-trip as band descriptions. So the drivers are stacked into a three-band
raster per timestep, the bands are named, and the collection is written out -- which is what makes
the variables ``precipitation`` / ``temperature`` / ``evapotranspiration`` rather than ``Band_1``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from pyramids.dataset import Dataset, DatasetCollection

from hapi.inputs import METEO_VARIABLES, MeteoInputs

HERE = Path(__file__).parent
OUT = HERE / "meteo.nc"
READ_KWARGS = dict(
    start="2009-01-01",
    end="2009-01-10",
    regex_string=r"\d{4}.\d{2}.\d{2}",
    date=True,
    file_name_data_fmt="%Y.%m.%d",
)


def main() -> None:
    """Stack the three raster folders into one multi-variable NetCDF."""
    inputs = MeteoInputs.from_rasters(
        HERE / "prec", HERE / "temp", HERE / "evap", **READ_KWARGS
    )
    template = Dataset.read_file(str(sorted((HERE / "prec").glob("*.tif"))[0]))
    geo, epsg, nodata = template.geotransform, template.epsg, template.no_data_value[0]

    staging = Path(tempfile.mkdtemp(prefix="coello-meteo-"))
    try:
        cubes = [getattr(inputs, name) for name in METEO_VARIABLES]
        for step, stamp in enumerate(inputs.time):
            bands = np.stack([cube[:, :, step] for cube in cubes]).astype("float32")
            raster = Dataset.create_from_array(
                bands, geo=geo, epsg=epsg, no_data_value=nodata
            )
            # to_netcdf names one variable per band, after the band name.
            raster.band_names = list(METEO_VARIABLES)
            raster.to_file(str(staging / f"{step}_meteo_{stamp:%Y.%m.%d}.tif"))

        collection = DatasetCollection.from_files(
            staging,
            glob="*.tif",
            date_format="%Y.%m.%d",
            date_regex=r"\d{4}.\d{2}.\d{2}",
        )
        if OUT.exists():
            OUT.unlink()
        collection.to_netcdf(str(OUT))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
