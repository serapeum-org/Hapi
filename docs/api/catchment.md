# Catchment

## Routing methods

`Catchment` accepts exactly three routing methods, matched case-insensitively and stored in one
spelling:

| Written as | Stored as | Routes |
|---|---|---|
| `muskingum` | `Muskingum` | Cell to cell along the flow-direction network. |
| `maxbas` | `MAXBAS` | Every cell straight to the outlet through a triangular function. |
| `kinematic` | `Kinematic` | The flood model's own path (`Run.run_flood`). |

`Calibration` does not take one at all: it holds a `Catchment`, and reads the method off the model
it was given.

Anything else raises a `ValueError` naming the three. Before this check the constructor stored
whatever string it was handed, so a run configured as `"Max_bas"` — or as a descriptive label such
as `"Muskingum-Cunge"` — was accepted and then silently routed with Muskingum, because the routing
loop compared against `"Muskingum"` exactly. That comparison is gone: which router runs is decided
by the entry point you call, and the stored method is read by `Run.run_flood`, which derives
`skip_hydraulic_cells` from `"Kinematic"`, and by the cross-check against `parameters.maxbas`. One
spelling is what keeps both honest; a script passing a spelling outside the table has to be updated
to one of the three.

A YAML run configuration reaches only the first two: `kinematic` selects the flood model, which
[`hapi.core.config`](config.md) does not describe.

## Catchment
::: hapi.model.catchment.Catchment
