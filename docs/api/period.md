# Simulation period

Six attributes on `Catchment` used to describe one thing: `start`, `end` and
`temporal_resolution` were given, and `date_index`, `dt` and `conversion_factor` were derived
from them in the constructor and then stored beside them as if they were independent. Storing a
derivation is how the three drift apart — reassigning `end` left `date_index` describing the old
span, with nothing to notice — and it is why the same `pd.date_range` branch was written out four
times across the package.

`SimulationPeriod` holds the three inputs and derives the rest on read, so they cannot disagree.
It is frozen: a run covers the period it was built for, and a model that needs a different one
gets a new period rather than a mutated one.

```python
model.period.date_index      # one entry per step
model.period.days            # how many steps
len(model.period)            # the same number
```

## SimulationPeriod
::: hapi.period.SimulationPeriod
