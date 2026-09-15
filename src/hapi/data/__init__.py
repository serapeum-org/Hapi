"""Access to the global parameter datasets the model can be initialised from.

Downloading only: nothing here reads a raster into the model. `figshare` fetches the Beck et
al. (2016) parameter sets into `HAPI_DATA_DIR`; the readers that then consume them live in
`hapi.inputs`.
"""

from __future__ import annotations
