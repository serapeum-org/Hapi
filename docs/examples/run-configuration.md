# Run Configuration (YAML)

The other example pages assemble a model in Python: construct a `Catchment`, then call
`read_lumped_inputs`, `read_parameters`, `read_lumped_model` and the rest in the right order.
That works, but it puts every path, date and area inside the script, so a script is only ever
about one catchment — and the call order is something you have to know.

A run configuration moves all of it into a YAML file that sits beside the data it names.
`Catchment.from_yaml` reads the file, validates it, and makes the same `read_*` calls in the
same order:

```python
from hapi.catchment import Catchment
from hapi.run import Run

Coello = Catchment.from_yaml("coello-lumped-model-run.yaml")
Run.runLumped(Coello, Routing.triangular_routing_1)
```

The four shipped examples under `examples/hydrological-model/coello/run/` are each a pair — a
`.py` that runs the model and a `.yaml` beside it holding everything the run needs.

## A complete lumped configuration

```yaml
# Paths are relative to this file, so the run works from any working directory.
catchment:
  name: Coello
  start: "2009-01-01"
  end: "2011-12-31"
  spatial_resolution: lumped
  temporal_resolution: daily

# Lumped mode reads one CSV of catchment-average drivers, not a grid: columns are
# [date, precipitation, ET, temperature], optionally followed by the long-term average.
meteo:
  path: ../../data/lumped_model/meteo_data-MSWEP.csv

parameters:
  path: ../../data/lumped_model/Coello_Lumped2021-03-08_muskingum.txt
  snow: false
  maxbas: false

conceptual_model:
  model_class: HBVBergestrom92
  catchment_area: 1530
  initial_condition: [0, 10, 10, 10, 0]

# One discharge file, and no gauge table: locating gauges on a grid is a distributed concern.
gauges:
  discharge: ../../data/lumped_model/Qout_c.csv
  fmt: "%Y-%m-%d"

outputs:
  results_dir: ../../data/lumped_model
```

`catchment`, `meteo` and `conceptual_model` are required. `parameters`, `gauges` and `outputs`
are optional: omit `parameters` for a calibration, which derives them from the bounds given to
`read_parameters_bound`, and omit `gauges` for a run that is not scored against observations.

## What changes for a distributed run

`spatial_resolution: distributed` changes the shape of two blocks and requires a third. `meteo`
becomes a grid, described by its `source`:

```yaml
catchment:
  name: Coello
  start: "2009-01-01"
  end: "2009-04-10"
  spatial_resolution: distributed
  routing_method: maxbas

meteo:
  source: rasters
  precipitation: ../../data/distributed_model/prec
  temperature: ../../data/distributed_model/temp
  evapotranspiration: ../../data/distributed_model/evap
  file_name_data_fmt: "%Y.%m.%d"

# MAXBAS sends every cell straight to the outlet, so no flow-direction raster is read.
flow_network:
  flow_accumulation: ../../data/distributed_model/GIS/acc4000.tif

gauges:
  table: ../../data/distributed_model/stations/gauges.csv
  discharge: ../../data/distributed_model/stations/
```

`meteo.source` picks which `MeteoInputs` loader builds the grid, and what the three driver
fields mean:

| `source` | The three driver fields name | Reads |
|---|---|---|
| `rasters` (default) | A folder of dated GeoTIFFs each | `MeteoInputs.from_rasters` |
| `netcdf_files` | One NetCDF each | `MeteoInputs.from_netcdf_files` |
| `netcdf` | A variable inside `meteo.path` | `MeteoInputs.from_netcdf` |

The last is the fastest: one file, opened once, with the calendar inside it. See
[Meteorological inputs](meteo-inputs.md) for how to pack a folder of rasters into one.

## Paths are relative to the file

A relative path in a configuration is resolved against the configuration's own directory, not
against whatever directory you happen to run from. That is what makes a configuration portable:
it travels with the data it names, and the run works from anywhere. Absolute paths are used as
written.

The example scripts rely on this — each loads the YAML sitting next to it:

```python
Coello = Catchment.from_yaml(__file__.removesuffix(".py") + ".yaml")
```

## What the file is checked for

The file is validated in full before anything is opened, so a mistake is reported as a mistake
in the file rather than as a failure deep inside a reader:

- **Unknown keys are refused.** A misspelled `precipitaton` fails at parse time instead of being
  dropped and reappearing as a missing input.
- **So are keys that do not apply.** A `flow_network` block on a lumped run, `glob` under
  `source: netcdf`, `gauges.table` on a lumped run — each is a line that would do nothing, and
  each is named in the error. Only keys you actually wrote count; defaults are never held
  against you.
- **Required blocks are checked per shape.** A distributed run needs `flow_network` and all
  three drivers; Muskingum additionally needs `flow_network.flow_direction`, which MAXBAS never
  reads. A lumped run needs `meteo.path`.
- **`routing_method` must agree with `parameters.maxbas`.** The two parameter counts differ by
  one and `maxbas` selects which is expected, so a disagreeing pair still counts correctly and
  then reads the wrong parameter as the routing one. Leave `routing_method` out and it is
  derived from the parameter set.
- **Every date is parsed against its own `fmt`**, and the period must run forwards — including
  the meteorological window, whose bounds fall back to the catchment's when unstated.
- **Every path is checked for existence** before the first reader runs, and all the missing ones
  are reported together.

Dates may be quoted or not: `start: 2009-01-01` is a date to YAML, and it is written back out in
the block's `fmt`.

## Reading the configuration back

The parsed configuration stays on the model as `model.config`, so the blocks the build does not
itself consume remain reachable — `outputs` above all:

```python
outputs = Coello.config.outputs
save_to = (outputs.results_dir if outputs is not None else None) or ""
Coello.save_results(
    flow_acc_path=Coello.config.flow_network.flow_accumulation,
    result=1,
    path=save_to,
)
```

## Out of scope

The schema describes a `Catchment` run. It carries no field for a lake record, a river geometry,
or a flow-path-length raster, so lake-aware runs (`Run.RunHapiwithLake`), the flood model
(`Run.RunFloodModel`) and `DistMaxbas2` are still assembled in Python.

`Calibration.from_yaml` works — it takes the same constructor arguments — and gives back a
`Calibration` to call the calibration methods on. `Run.from_yaml` does not: `Run` holds entry
points called on a model built elsewhere, so it refuses and says so.

The full field-by-field reference is on the [Config API page](../api/config.md).
