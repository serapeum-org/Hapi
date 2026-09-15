"""The vocabulary a run is stated in, shared by the builder and the engine.

`validated` holds the model that passed validation -- `DistributedRun`, `LumpedRun` -- which
is what an engine entry point actually consumes. `results` holds what one produced.
`protocols` states structurally what a run requires, so the engine never has to import the
builder to say it.

These sit below both `hapi.model` and `hapi.engine` for that reason: put them in the engine
and the builder would have to reach upwards for its own output type.
"""

from __future__ import annotations
