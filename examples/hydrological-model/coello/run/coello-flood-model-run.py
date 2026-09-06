"""Distributed model on the flood-model path, with kinematic-wave routing declared.

Ported from `tests/FloodModel.py`, which left this repository in commit `733957be`
("move flood model to serapis", 2024-01-03) and still sits unchanged in Serapis. That
script could not run here any more: it pointed at `F:/02Case-studies/`, imported the
pre-rename `Hapi` package, and called five readers that no longer exist --
`read_flow_acc`, `read_flow_dir`, `read_rainfall`, `read_temperature` and `read_et`,
all of which moved into `FlowNetwork` and `MeteoInputs`.

What the flood path does, and does not, do:

`Run.run_flood` validates the four river-geometry rasters against the catchment grid and
then runs the ordinary Muskingum distributed model. Declaring `routing_method="Kinematic"`
tells it that a 1D hydraulic model routes the river cells, so the Muskingum pass leaves
them alone -- that is the handoff the original design intended, and the hydraulic half
(`SaintVenant.KinematicRaster`) went to Serapis in the same commit. Nothing in Hapi picks
those cells up, so **with `"Kinematic"` the river cells carry no routed discharge**. Run it
with `"Muskingum"` -- the default below -- to route every cell here, and use `"Kinematic"`
only when Serapis will take the river cells from you.

The paths are written from the repo root, so run this from there:
`python examples/hydrological-model/coello/run/coello-flood-model-run.py`.
"""

from __future__ import annotations

import numpy as np

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
from hapi.run import Run

# %% Paths
DATA = "examples/hydrological-model/data/distributed_model"
GIS = f"{DATA}/GIS"

PREC_PATH = f"{DATA}/prec"
EVAP_PATH = f"{DATA}/evap"
TEMP_PATH = f"{DATA}/temp"
FLOW_ACC_PATH = f"{GIS}/acc4000.tif"
FLOW_DIR_PATH = f"{GIS}/fd4000.tif"
PARAMETERS_PATH = f"{DATA}/Parameter set-Avg"

# %% Flood-model rasters
DEM_FILE = f"{GIS}/dem4000.tif"
BANKFULL_DEPTH_FILE = f"{GIS}/bankfulldepth.tif"
RIVER_WIDTH_FILE = f"{GIS}/river_width.tif"
RIVER_ROUGHNESS_FILE = f"{GIS}/channel_roughness.tif"
FLOODPLAIN_ROUGHNESS_FILE = f"{GIS}/floodplain_roughness.tif"

# "Muskingum" routes every cell here. Use "Kinematic" only when a hydraulic model
# downstream will route the river cells -- see the module docstring.
ROUTING_METHOD = "Muskingum"

# %% Model configuration
AREA = 1530
INITIAL_COND = [0, 5, 5, 5, 0]
SNOW = False
START = "2009-01-01"
END = "2009-01-10"

# %% Build the model
Coello = Catchment(
    "Coello",
    START,
    END,
    spatial_resolution="Distributed",
    routing_method=ROUTING_METHOD,
)

Coello.meteo = MeteoInputs.from_rasters(
    PREC_PATH,
    TEMP_PATH,
    EVAP_PATH,
    start=START,
    end=END,
    regex_string=r"\d{4}.\d{2}.\d{2}",
    date=True,
    file_name_data_fmt="%Y.%m.%d",
)
Coello.flow_network = FlowNetwork.from_rasters(FLOW_ACC_PATH, FLOW_DIR_PATH)
Coello.read_river_geometry(
    DEM_FILE,
    BANKFULL_DEPTH_FILE,
    RIVER_WIDTH_FILE,
    RIVER_ROUGHNESS_FILE,
    FLOODPLAIN_ROUGHNESS_FILE,
)
Coello.read_parameters(PARAMETERS_PATH, SNOW)
Coello.read_lumped_model(HBVBergestrom92, AREA, INITIAL_COND)

# %% Run the flood model
results = Run.run_flood(Coello)

# The domain is the flow-accumulation mask, not the whole grid: `q_total` is allocated
# with zeros everywhere, so counting its non-NaN cells would just report rows x cols.
inside = ~np.isnan(Coello.flow_network.flow_acc_arr)
river_cells = int(
    np.count_nonzero((np.nan_to_num(Coello.river_geometry.bankfull_depth) > 0) & inside)
)

print(f"routing            : {results.routing.value}")
print(f"q_total            : {results.q_total.shape}")
print(f"catchment cells    : {int(np.count_nonzero(inside))}")
print(
    f"river cells        : {river_cells} (left to a hydraulic model under 'Kinematic')"
)
print(f"basin-wide q_total : {float(np.nansum(results.q_total)):.1f}")
