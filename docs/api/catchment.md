# Catchment

## Routing methods

`Catchment` and `Calibration` accept exactly three routing methods, matched case-insensitively
and stored in the one spelling the internals compare against:

| Written as | Stored as | Routes |
|---|---|---|
| `muskingum` | `Muskingum` | Cell to cell along the flow-direction network. |
| `maxbas` | `MAXBAS` | Every cell straight to the outlet through a triangular function. |
| `kinematic` | `Kinematic` | The flood model's own path (`Run.RunFloodModel`). |

Anything else raises a `ValueError` naming the three. Up to and including version 1.7.0 the
constructor stored whatever string it was handed, so a run configured as `"Max_bas"` — or as a
descriptive label such
as `"Muskingum-Cunge"` — was accepted and then silently routed with Muskingum, because
`distrrm.SpatialRouting` compares against `"Muskingum"` exactly. Rejecting the spelling is what
makes that comparison trustworthy; a script passing a spelling outside the table has to be updated
to one of the three.

A YAML run configuration reaches only the first two: `kinematic` selects the flood model, which
[`hapi.config`](config.md) does not describe.

## Catchment
::: hapi.catchment.Catchment
