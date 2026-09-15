"""The lumped conceptual models, and the objects that configure one.

`base` declares the surface every model implements; `hbv`, `hbv_bergestrom92` and `hbv_lake`
implement it. `setup` holds what a run has to supply alongside the model -- the parameter
vector, the initial conditions, the catchment area -- and validates the combination before a
run starts.

The names `setup` defines are re-exported here, so `from hapi.conceptual import ParameterSet`
reads the same as it did when this was a single module.
"""

from __future__ import annotations

from hapi.conceptual.setup import (
    PARAMETER_COUNTS,
    ConceptualModelSetup,
    ParameterBounds,
    ParameterSet,
    parameter_count,
    validate_initial_cond,
    validate_parameter_count,
    validate_q_init,
)

__all__ = [
    "PARAMETER_COUNTS",
    "ConceptualModelSetup",
    "ParameterBounds",
    "ParameterSet",
    "parameter_count",
    "validate_initial_cond",
    "validate_parameter_count",
    "validate_q_init",
]
