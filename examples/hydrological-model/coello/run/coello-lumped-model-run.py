"""Lumped model with Muskingum routing, built from YAML.

Everything that used to be a "Paths" block of hardcoded assignments now lives in
`coello-lumped-model-run.yaml`, next to this script -- `Catchment.from_yaml` reads it and
assembles the model. Running it stays here, as in any hand-wired script: the routing function is
a run-time choice rather than an input, so it is picked below and handed to `Run.runLumped`.

Lumped mode reads one CSV of catchment-average drivers instead of a grid, and one discharge file
instead of a gauge table plus a folder -- see the config for both.
"""

from __future__ import annotations

import datetime as dt

import statista.descriptors as metrics

from hapi.catchment import Catchment
from hapi.routing import Routing
from hapi.run import Run

# %% Load the configuration and build the model
Coello = Catchment.from_yaml(
    "examples/hydrological-model/coello/run/coello-lumped-model-run.yaml"
)

# %% Routing
# RoutingFn = Routing.triangular_routing_2
RoutingFn = Routing.muskingum_v
Route = 1

# %% Run the model
Run.runLumped(Coello, Route, RoutingFn)

# %% Calculate performance criteria
scores = dict()

Qobs = Coello.QGauges["q"]

scores["RMSE"] = metrics.rmse(Qobs, Coello.Qsim["q"])
scores["NSE"] = metrics.nse(Qobs, Coello.Qsim["q"])
scores["NSEhf"] = metrics.nse_hf(Qobs, Coello.Qsim["q"])
scores["KGE"] = metrics.kge(Qobs, Coello.Qsim["q"])
scores["WB"] = metrics.wb(Qobs, Coello.Qsim["q"])

print("RMSE= " + str(round(scores["RMSE"], 2)))
print("NSE= " + str(round(scores["NSE"], 2)))
print("NSEhf= " + str(round(scores["NSEhf"], 2)))
print("KGE= " + str(round(scores["KGE"], 2)))
print("WB= " + str(round(scores["WB"], 2)))

# %% Plot Hydrograph
gaugei = 0
fig, ax = Coello.plot_hydrograph(Coello.start, Coello.end, gaugei, title="Lumped Model")

# %% Save Results
SaveTo = "examples/hydrological-model/data/lumped_model/"
StartDate = "2009-01-01"
EndDate = "2010-04-20"

path = f"{SaveTo}Results-Lumped-Model_{str(dt.datetime.now())[0:10]}.txt"
Coello.save_results(result=5, start=StartDate, end=EndDate, path=path)
print(f"results written to  : {path}")
