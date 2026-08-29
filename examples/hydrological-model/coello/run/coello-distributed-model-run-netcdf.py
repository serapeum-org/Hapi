"""Distributed Muskingum model driven from a single combined NetCDF.

Standalone version of the workflow behind
`tests/rrm/catchment/test_e2e_coello_from_netcdf.py::TestMuskingumPipeline::
test_the_drivers_come_from_the_file_and_cover_the_model`: one `MeteoInputs.from_netcdf` call
replaces the three raster-folder reads, so the model touches no meteorological raster at all.
`meteo.nc` packs the rainfall, temperature and evapotranspiration folders bundled under
`tests/rrm/data/coello/{prec,temp,evap}` into one file with the calendar inside it -- see
`tests/rrm/data/coello/convert_and_combine_meteo_inputs_to_netcdf.py` for how it was built.
"""

from __future__ import annotations

import numpy as np

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBV
from hapi.run import Run

# %% Paths
Path = "tests/rrm/data/coello"
MeteoPath = f"{Path}/meteo.nc"
FlowAccPath = f"{Path}/gis/acc4000.tif"
FlowDPath = f"{Path}/gis/fd4000.tif"
ParPath = f"{Path}/parameters/muskingum"
GaugesTablePath = f"{Path}/calibration/gauges.csv"
GaugesPath = f"{Path}/calibration"

# %% Meteorological data -- the whole point: one file, no raster reads
AreaCoeff = 1530
InitialCond = [0, 5, 5, 5, 0]
Snow = 0

start = "2009-01-01"
end = "2009-01-10"
name = "Coello"

Coello = Catchment(name, start, end, spatial_resolution="Distributed")
Coello.meteo = MeteoInputs.from_netcdf(
    MeteoPath,
    precipitation="precipitation",
    temperature="temperature",
    evapotranspiration="evapotranspiration",
)

Coello.flow_network = FlowNetwork.from_rasters(FlowAccPath, FlowDPath)
Coello.read_parameters(ParPath, Snow, maxbas=False)
Coello.read_lumped_model(HBV, AreaCoeff, InitialCond)

# %% Gauges
Coello.read_gauge_table(GaugesTablePath, FlowAccPath)
Coello.read_discharge_gauges(GaugesPath, column="id", fmt="%Y-%m-%d")

# %% Check the drivers actually came from the file and cover the model
print(f"meteo grid + steps : {Coello.meteo.shape}")
print(f"model steps        : {len(Coello.date_index)}")
print(f"meteo period       : {Coello.meteo.time[0]} -> {Coello.meteo.time[-1]}")
print(f"model period       : {Coello.date_index[0]} -> {Coello.date_index[-1]}")
assert Coello.meteo.time_steps == len(Coello.date_index), (
    "the drivers must hold exactly as many steps as the model spans"
)
assert Coello.meteo.time[0] == Coello.date_index[0], "the drivers must start where the model does"
assert Coello.meteo.time[-1] == Coello.date_index[-1], "the drivers must end where the model does"

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
    print(f"{field_name:15s} shape {field.shape}, finite inside: {np.isfinite(field[inside]).all()}")

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
SaveTo = "results/saved rasters/"
Coello.save_results(flow_acc_path=FlowAccPath, result=1, path=SaveTo)
print(f"rasters written to  : {SaveTo}")

# %% Plot the hydrograph at the outlet gauge (row position, not the gauge id)
Coello.plot_hydrograph(start, end, Coello.GaugesTable.index[-1])
