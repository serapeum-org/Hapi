"""Distributed Muskingum model driven from a single combined NetCDF, built from YAML.

Standalone version of the workflow behind
`tests/rrm/catchment/test_e2e_coello_from_netcdf.py::TestMuskingumPipeline::
test_the_drivers_come_from_the_file_and_cover_the_model`: one `MeteoInputs.from_netcdf` call
replaces the three raster-folder reads, so the model touches no meteorological raster at all.
`meteo.nc` packs the rainfall, temperature and evapotranspiration folders bundled under
`tests/rrm/data/coello/{prec,temp,evap}` into one file with the calendar inside it -- see
`tests/rrm/data/coello/convert_and_combine_meteo_inputs_to_netcdf.py` for how it was built.

Everything that used to be a "Paths" block of hardcoded assignments now lives in
`coello-distributed-model-run-netcdf.yaml`, next to this script -- `Catchment.from_yaml` reads
it and assembles the `Catchment` the same way `_build` did in the e2e test.
"""

from __future__ import annotations

import numpy as np

from hapi.catchment import Catchment
from hapi.run import Run

# %% Load the configuration and build the model
Coello = Catchment.from_yaml(__file__.removesuffix(".py") + ".yaml")

# %% Check the drivers actually came from the file and cover the model
print(f"meteo grid + steps : {Coello.meteo.shape}")
print(f"model steps        : {len(Coello.date_index)}")
print(f"meteo period       : {Coello.meteo.time[0]} -> {Coello.meteo.time[-1]}")
print(f"model period       : {Coello.date_index[0]} -> {Coello.date_index[-1]}")
if Coello.meteo.time_steps != len(Coello.date_index):
    raise ValueError("the drivers must hold exactly as many steps as the model spans")
if Coello.meteo.time[0] != Coello.date_index[0]:
    raise ValueError("the drivers must start where the model does")
if Coello.meteo.time[-1] != Coello.date_index[-1]:
    raise ValueError("the drivers must end where the model does")

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
        3D array of the upper zone discharge accumulated and
        routed at each time step
    6-qlz_translated: [numpy attribute]
        3D array of the lower zone discharge translated at each time step
"""
Run.RunHapi(Coello)

# %% Routed fields cover the grid, finite inside the catchment
inside = ~np.isnan(Coello.flow_network.flow_acc_arr)
for field_name in ("Qtot", "quz_routed", "qlz_translated"):
    field = getattr(Coello, field_name)
    print(
        f"{field_name:15s} shape {field.shape}, finite inside: {np.isfinite(field[inside]).all()}"
    )

# %% Extract discharge at every gauge and score against the observations
Coello.extract_discharge(calculate_metrics=True)

for gauge_id in Coello.GaugesTable["id"]:
    print("----------------------------------")
    print(f"Gauge - {gauge_id}")
    print(f"RMSE=    {Coello.metrics.loc['RMSE', gauge_id]:.2f}")
    print(f"NSE=     {Coello.metrics.loc['NSE', gauge_id]:.2f}")
    print(f"NSEhf=   {Coello.metrics.loc['NSEhf', gauge_id]:.2f}")
    print(f"KGE=     {Coello.metrics.loc['KGE', gauge_id]:.2f}")
    print(f"WB=      {Coello.metrics.loc['WB', gauge_id]:.2f}")

# %% Save the routed discharge to rasters, one per time step
# Both paths come from the configuration rather than being restated here: `save_results`
# re-reads the flow-accumulation raster for georeferencing (FlowNetwork keeps only the arrays,
# not the source path), and `outputs.results_dir` says where the rasters go. The block is
# optional, so a configuration without one writes beside the script rather than failing on a
# missing attribute.
outputs = Coello.config.outputs
save_to = (outputs.results_dir if outputs is not None else None) or ""
Coello.save_results(
    flow_acc_path=Coello.config.flow_network.flow_accumulation,
    result=1,
    path=save_to,
)
print(f"rasters written to  : {save_to}")

# %% Plot the hydrograph at the outlet gauge (row position, not the gauge id)
Coello.plot_hydrograph(Coello.start, Coello.end, Coello.GaugesTable.index[-1])
