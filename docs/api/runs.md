# Runs

A `Catchment` is a **builder**: its inputs are `X | None` until the matching `read_*` call has
run, and that is honest. The engines need the opposite — a catchment that is finished.
`DistributedRun` and `LumpedRun` are that finished form.

`from_model` is the single validation seam. Constructing a run *is* the validation: it resolves
every optional input, checks the drivers, the parameter cube, the river geometry and the
flow-path-length raster against the catchment grid, and refuses a combination the engines cannot
run. Every engine entry point takes one of these types, so the checks are enforced by the
signatures rather than by remembering to call them — which is how `Calibration`, going straight
to `Wrapper`, used to skip all of them.

```python
run = DistributedRun.from_model(model)          # checked here, once
results = Wrapper.run_muskingum(run)            # nothing left to re-check
```

Both are frozen. The checks happen at construction, so a mutable run would let a caller swap an
input in afterwards and reach an engine with something never validated.

## DistributedRun
::: hapi.runs.DistributedRun

## LumpedRun
::: hapi.runs.LumpedRun
