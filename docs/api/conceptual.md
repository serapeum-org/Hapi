# Conceptual model inputs

The conceptual model's inputs used to be six loose attributes on `Catchment` whose rules were
enforced nowhere in particular. They are three value objects now, each checking its own invariant
in `__post_init__`, so a bad combination is refused where it is made rather than several frames
into a run.

- `ParameterSet` — the parameter values plus the `(snow, maxbas)` pair that fixes their width.
  Every route to a parameter set goes through the same width rule, including the per-trial
  replacements a calibration makes. It is frozen; use `with_values` to derive a new set from an
  optimiser's vector.
- `ConceptualModelSetup` — the model, the catchment area, the initial condition and the initial
  discharge, as `read_lumped_model` produces them.
- `ParameterBounds` — the calibration's search space, held to the same width rule as the trial
  vectors it bounds.

## ParameterSet
::: hapi.conceptual.ParameterSet

## ConceptualModelSetup
::: hapi.conceptual.ConceptualModelSetup

## ParameterBounds
::: hapi.conceptual.ParameterBounds

## Parameter-count helpers
::: hapi.conceptual.parameter_count

::: hapi.conceptual.validate_parameter_count
