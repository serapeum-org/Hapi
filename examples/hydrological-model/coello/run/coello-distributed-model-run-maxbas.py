"""Distributed model with a maxbas routing scheme, built from YAML.

Everything that used to be a "Paths" block of hardcoded assignments now lives in
`coello-distributed-model-run-maxbas.yaml`, next to this script -- `Catchment.from_yaml` reads
it and assembles the model. Running it stays here, as in any hand-wired script.

MAXBAS sends every cell straight to the outlet, so the config loads no flow-direction raster and
`extract_discharge` needs `frame_work_1=True`: a cell of `Qtot` is that cell's contribution to
the outlet rather than the discharge at it, which makes the per-gauge shortcut invalid.
"""

from __future__ import annotations

from hapi.catchment import Catchment
from hapi.run import Run

# %% Load the configuration and build the model
Coello = Catchment.from_yaml("coello-distributed-model-run-maxbas.yaml")

# %% Run the model
"""
Outputs:
    ----------
    1-state_variables: [numpy attribute]
        4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
    2-qlz: [numpy attribute]
        3D array of the lower zone discharge
    3-quz: [numpy attribute]
        3D array of the upper zone discharge
    4-qout: [numpy attribute]
        1D timeseries of discharge at the outlet of the catchment
        of unit m3/sec
    5-quz_routed: [numpy attribute]
        3D array of the upper zone discharge  accumulated and
        routed at each time step
    6-qlz_translated: [numpy attribute]
        3D array of the lower zone discharge translated at each time step
"""
Run.runFW1(Coello)

# %% calculate performance criteria
Coello.extract_discharge(calculate_metrics=True, frame_work_1=True)

gaugeid = Coello.GaugesTable.loc[Coello.GaugesTable.index[-1], "id"]
print("----------------------------------")
print("Gauge - " + str(gaugeid))
print("RMSE= " + str(round(Coello.metrics.loc["RMSE", gaugeid], 2)))
print("NSE= " + str(round(Coello.metrics.loc["NSE", gaugeid], 2)))
print("NSEhf= " + str(round(Coello.metrics.loc["NSEhf", gaugeid], 2)))
print("KGE= " + str(round(Coello.metrics.loc["KGE", gaugeid], 2)))
print("WB= " + str(round(Coello.metrics.loc["WB", gaugeid], 2)))

# %% plot the hydrograph at the outlet gauge (row position, not the gauge id)
Coello.plot_hydrograph(Coello.start, Coello.end, Coello.GaugesTable.index[-1])
