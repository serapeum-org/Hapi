"""What the run layer accepts from a caller, before it has been narrowed.

The entry points in :mod:`hapi.run` used to declare their argument as
:class:`~hapi.catchment.Catchment` -- a class of 40-odd attributes, of which any one run touches
a dozen. That named the wrong thing: it over-stated the requirement, and pointed the dependency
at a concrete class, so the run layer could not be reasoned about without the class it runs.

:class:`CatchmentLike` states the requirement instead, and states it *honestly*: a catchment is
a builder, so its inputs really are `X | None` until the matching `read_*` call has run. An
entry point accepts one of these and immediately narrows it with
:meth:`~hapi.runs.DistributedRun.from_model`, which is where the optionality is resolved and
every cross-input check happens. Past that seam the engines see
:class:`~hapi.runs.DistributedRun` or :class:`~hapi.runs.LumpedRun`, whose fields are not
optional at all.

So there are two types on purpose, and the split is the point: this one describes what a caller
can hand over, the run types describe what an engine is allowed to receive. `Catchment`
satisfies this protocol structurally, so neither :mod:`hapi.run` nor :mod:`hapi.wrapper` imports
it at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from hapi.conceptual import ConceptualModelSetup, ParameterSet
from hapi.inputs import FlowNetwork, MeteoInputs, RiverGeometry
from hapi.period import SimulationPeriod
from hapi.results import SimulationResults


class CatchmentLike(Protocol):
    """A catchment under assembly: the period is settled, the inputs may not be.

    Every field but `period` is optional, because that is the truth about a builder -- and it
    is why an entry point narrows before it runs anything, rather than dereferencing these.

    Attributes:
        period: The span the model covers. Settled at construction, so never `None`.
        meteo: The three driver cubes, once assigned.
        flow_network: The routing network and grid, once assigned.
        parameters: The parameter set, once `read_parameters` has run.
        model_setup: The conceptual model, once `read_lumped_model` has run.
        data: The lumped driver record, once `read_lumped_inputs` has run.
        river_geometry: The hydraulic rasters, once `read_river_geometry` has run.
        flow_path_length_arr: The flow-path-length raster, once read.
        routing_method: Which routing the parameter set was calibrated for.
        results: Where a run's output lands. `None` before the first run.
    """

    period: SimulationPeriod
    meteo: MeteoInputs | None
    flow_network: FlowNetwork | None
    parameters: ParameterSet | None
    model_setup: ConceptualModelSetup | None
    data: np.ndarray | None
    river_geometry: RiverGeometry | None
    flow_path_length_arr: np.ndarray | None
    routing_method: str
    results: SimulationResults | None


class SupportsQsim(CatchmentLike, Protocol):
    """A catchment that also has somewhere for a lumped hydrograph to land.

    Attributes:
        Qsim: Where `Run.run_lumped` puts the routed series, as a frame indexed by the period.
    """

    Qsim: Any


class SpatialDistribution(Protocol):
    """What a calibration needs of the thing that maps a flat vector onto the model's grid.

    The optimiser searches over a flat vector; the model runs on a `(rows, cols, n)` array. A
    spatial-distribution object is what converts one into the other, and
    :class:`hapi.rrm.parameters.Parameters` is the implementation that ships here.

    The calibration entry points used to type this argument `Callable[..., Any]`, which was
    doubly wrong: a calibration never calls it, and what it actually does is read four members
    off it. So the annotation described a function while the code used an object, and mypy had
    nothing to check the four accesses against.

    Attributes:
        Function: The distribution strategy, chosen by `Parameters.__init__` from the `function`
            argument -- an attribute holding a callable rather than a method, which is why it is
            declared as one. A calibration calls it with the trial vector alone; callers outside
            may pass `kub` / `klb` too, hence the open signature.
        Par3d: The `(rows, cols, no_parameters)` array `Function` fills in. Read straight after
            each call, so the two are a pair: calling `Function` is what makes this current.
        no_parameters: Parameters per cell. Strides the Muskingum K/X pairs out of the trial
            vector when the constraints are built.
        no_elem: Cells inside the domain. Sizes the two inequality constraints per cell.
    """

    Function: Callable[..., Any]
    Par3d: np.ndarray
    no_parameters: int
    no_elem: int
