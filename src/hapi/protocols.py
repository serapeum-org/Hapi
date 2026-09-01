"""What the run layer requires of the model it is handed.

The entry points in :mod:`hapi.run` and the wiring in :mod:`hapi.wrapper` used to declare
their argument as :class:`~hapi.catchment.Catchment` -- a class of 40-odd attributes, of which
any one run touches a dozen. That named the wrong thing: it over-stated the requirement, and
it pointed the dependency at a concrete class, so the run layer could not be reasoned about
without the class it runs.

These protocols state the requirement instead. `Catchment` satisfies them structurally without
inheriting anything, so neither `hapi.run` nor `hapi.wrapper` imports it at runtime, and any
other object carrying the same attributes runs too.

They live in their own module rather than in `hapi.run` because `hapi.run` imports
`hapi.wrapper`, and `hapi.wrapper` needs the same protocols -- a shared home is what keeps
that from being a cycle.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from hapi.conceptual import ConceptualModelSetup, ParameterSet
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.period import SimulationPeriod
from hapi.results import SimulationResults


class ConceptualModelInputs(Protocol):
    """What the per-cell conceptual model needs, whatever routes its output.

    The part of the contract the lumped and distributed paths share. Split out so that
    :class:`LumpedModel` does not advertise a need for a flow network it never touches.

    Attributes:
        parameters: The parameter array together with the `(snow, maxbas)` pair that fixes
            its width -- one object, so the width rule is checked on every route to a set.
        model_setup: The conceptual model instance, the catchment area and the state the
            run starts from.
        period: The span the run covers, and the calendar, `dt` and `conversion_factor`
            it implies. One object rather than six loose fields, so a derived value can
            never describe a different span from the one the model is set to.
        results: Where the run writes its output. None before the first run.
    """

    period: SimulationPeriod
    parameters: ParameterSet
    model_setup: ConceptualModelSetup
    results: SimulationResults | None


class DistributedModel(ConceptualModelInputs, Protocol):
    """What a distributed run requires on top of the conceptual model's own inputs.

    Attributes:
        meteo: The three driver cubes and the calendar they cover.
        flow_network: The routing network and the grid it defines.
    """

    meteo: MeteoInputs
    flow_network: FlowNetwork


class LumpedModelInputs(ConceptualModelInputs, Protocol):
    """What a lumped run requires: one column per variable rather than a grid.

    Attributes:
        data: `(time, 4)` array of precipitation, ET, temperature and the long-term average.
        Qsim: Where the routed hydrograph lands. Whether MAXBAS routing applies is read off
            `parameters.maxbas`, since that is what fixes the vector's width too.
    """

    data: np.ndarray
    Qsim: Any


class FloodModel(DistributedModel, Protocol):
    """A distributed model that also carries the river geometry the flood model reads.

    Attributes:
        routing_method: `"Kinematic"` when the kinematic-wave model routes the river cells,
            which is what `run_flood` reads to decide whether the Muskingum pass skips them.
        bankfull_depth: `(rows, cols)` bankfull depth. A positive value marks a river cell,
            which the flood model can leave for a 1D hydraulic model to route.
        river_width: `(rows, cols)` channel width.
        river_roughness: `(rows, cols)` channel roughness.
        flood_plain_roughness: `(rows, cols)` floodplain roughness.
    """

    routing_method: str
    bankfull_depth: np.ndarray
    river_width: np.ndarray
    river_roughness: np.ndarray
    flood_plain_roughness: np.ndarray
