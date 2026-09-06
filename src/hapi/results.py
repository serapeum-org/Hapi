"""The arrays a model run produces, and the routing that produced them.

Running a model used to leave its output as nine separate attributes on the
:class:`~hapi.catchment.Catchment` it was handed, with a private boolean recording which
routing scheme had written them. That made a catchment's state unknowable between runs --
the fields of a finished run and the fields of a half-finished one look the same -- and it
put the interpretation of the arrays (`_maxbas_routed`) on the input object rather than on
the arrays themselves.

:class:`SimulationResults` holds them together instead, with the routing scheme as a field.
A run assigns one to `Catchment.results`, and that is the only place the arrays live -- read
them as `model.results.q_total`. The catchment carries no result attributes of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RoutingKind(Enum):
    """Which routing scheme produced a set of results.

    The distinction is not cosmetic: it decides how a single cell of
    :attr:`SimulationResults.q_total` should be read. Under Muskingum the discharge accumulates
    downstream, so a cell *is* the discharge at that cell and the outlet cell carries the
    outlet hydrograph. Under MAXBAS every cell is routed straight to the outlet with its own
    `maxbas`, so a cell is only that cell's *contribution* and the hydrograph is the sum over
    the domain.

    Attributes:
        UNROUTED: The per-cell conceptual model has run, but no routing has been applied yet.
            The state every distributed run passes through between
            :meth:`~hapi.rrm.distrrm.DistributedRRM.run_lumped_model` and its routing step.
        MUSKINGUM: Cell-to-cell Muskingum routing along the flow network.
        MAXBAS: Triangular (MAXBAS) routing of each cell straight to the outlet.
        LUMPED: No spatial routing -- the catchment was run as a single unit.
    """

    UNROUTED = "unrouted"
    MUSKINGUM = "muskingum"
    MAXBAS = "maxbas"
    LUMPED = "lumped"


@dataclass
class SimulationResults:
    """The arrays one model run produced, and the routing that produced them.

    Built by the run layer and assigned to `Catchment.results`. Mutable, because the run
    fills it in stages: the per-cell model writes :attr:`quz`, :attr:`qlz` and
    :attr:`state_variables`, and the routing step then adds the routed fields and sets
    :attr:`routing`.

    Attributes:
        routing: Which scheme routed these arrays. See :class:`RoutingKind`.
        quz: `(rows, cols, time)` upper-zone discharge in m3/s. For a lumped run, a 1D series.
        qlz: `(rows, cols, time)` lower-zone discharge in m3/s. For a lumped run, a 1D series.
        state_variables: `(rows, cols, time, 5)` state array, the states being
            `[sp, sm, uz, lz, wc]`. For a lumped run, `(time, 5)`. `None` when a distributed
            run was asked not to keep them -- it is five times the size of every other field
            put together and nothing but `save_results` and `plot_distributed_results` reads
            it, so a run that will not look at it need not pay for it. See
            :attr:`~hapi.runs.DistributedRun.keep_state_variables`.
        quz_routed: Upper-zone discharge after routing. `None` until a routing step runs.
        qlz_translated: Lower-zone discharge after translation. `None` until then.
        q_total: `quz_routed + qlz_translated`. Read it through
            :attr:`outlet_shortcut_valid` rather than assuming what a cell means.
        qout: The outlet hydrograph, when the run computed one. The MAXBAS paths sum over the
            domain and set it directly; the Muskingum paths leave it `None` for
            :meth:`~hapi.catchment.Catchment.extract_discharge` to read off the outlet cell,
            which needs the gauge table the engine does not have.

    Examples:
        - A freshly run, unrouted set knows it is not yet interpretable at the outlet:
            ```python
            >>> import numpy as np
            >>> from hapi.results import RoutingKind, SimulationResults
            >>> cube = np.zeros((2, 3, 4), dtype="float32")
            >>> results = SimulationResults(
            ...     routing=RoutingKind.UNROUTED, quz=cube, qlz=cube,
            ...     state_variables=np.zeros((2, 3, 4, 5), dtype="float32"),
            ... )
            >>> results.routing.value
            'unrouted'
            >>> results.q_total is None
            True

            ```
        - The outlet-cell shortcut is valid under Muskingum and not under MAXBAS:
            ```python
            >>> import numpy as np
            >>> from hapi.results import RoutingKind, SimulationResults
            >>> cube = np.zeros((2, 3, 4), dtype="float32")
            >>> states = np.zeros((2, 3, 4, 5), dtype="float32")
            >>> muskingum = SimulationResults(
            ...     RoutingKind.MUSKINGUM, cube, cube, states
            ... )
            >>> maxbas = SimulationResults(RoutingKind.MAXBAS, cube, cube, states)
            >>> muskingum.outlet_shortcut_valid, maxbas.outlet_shortcut_valid
            (True, False)

            ```
    """

    routing: RoutingKind
    quz: np.ndarray
    qlz: np.ndarray
    state_variables: np.ndarray | None
    quz_routed: np.ndarray | None = None
    qlz_translated: np.ndarray | None = None
    q_total: np.ndarray | None = None
    qout: np.ndarray | None = None

    @property
    def outlet_shortcut_valid(self) -> bool:
        """bool: Whether a single cell of :attr:`q_total` is the discharge *at* that cell.

        True for every scheme except MAXBAS, which routes each cell straight to the outlet
        and so makes a cell a contribution rather than a discharge. Reading the outlet cell
        of a MAXBAS run under-reports the hydrograph, which is what this guards.
        """
        return self.routing is not RoutingKind.MAXBAS
