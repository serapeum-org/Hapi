# Results

Every `Run.*` entry point returns a `SimulationResults` and assigns it to `Catchment.results`. That
object is the only home for the arrays a run produced — the catchment carries no result attributes
of its own — and it is also what renders and writes them.

## Reading a run

```python
results = Run.run_distributed(model)   # also assigned to model.results

results.q_total          # (rows, cols, time) total discharge
results.routing          # RoutingKind.MUSKINGUM
results.run.period       # the calendar the arrays are indexed by
```

`routing` is not decoration: it decides how one cell of `q_total` may be read. Under Muskingum the
discharge accumulates downstream, so a cell *is* the discharge at that cell. Under MAXBAS every cell
is routed straight to the outlet, so a cell is only that cell's contribution and the hydrograph is
the sum over the domain. Ask `results.outlet_shortcut_valid` rather than assuming.

## Viewing and saving

| Call | Does |
|---|---|
| `results.animate(start, end, option=1)` | Animates a result array or a driver over the grid. |
| `results.save_animation(path, fps=2)` | Writes the animation `animate` built. |
| `results.save(path, result=1, flow_acc_path=...)` | One GeoTIFF per step, or a CSV for a lumped run. |

`animate` and `save` need the run behind the arrays — the calendar to index them by and the grid to
mask them with — which is why `SimulationResults` carries the `DistributedRun` or `LumpedRun` that
produced it. A results object built by hand rather than by a run says so instead of failing on
`None`.

`save` chooses rasters or CSV from `routing`: a lumped run has no grid to write rasters on, and that
is a property of the results rather than something the caller restates. The raster branch needs
`flow_acc_path` because `FlowNetwork` keeps the accumulation *array* but not its projection, so the
georeferencing has to be read back from the file.

Importing the run layer does not import matplotlib or cleopatra: `animate` imports them itself, so a
model run never pays for a plotting stack it does not use.

`Catchment.plot_hydrograph` stayed on the catchment. It reads no result array — it compares `Qsim`
against the observed gauge record, which is an analysis input, not something a run produced.

## SimulationResults
::: hapi.results.SimulationResults

## RoutingKind
::: hapi.results.RoutingKind
