"""Execution: what turns a validated model into results.

`run` is the entry-point namespace a caller reaches for; `wrapper` wires each entry point to
the spatial machinery; `distributed` runs the conceptual model cell by cell and accumulates
downstream; `routing` holds the routing functions both paths use.

Nothing here imports `hapi.model`. The engine states what it needs through
`hapi.simulation.protocols`, which `Catchment` satisfies structurally -- so the builder and
the engine are siblings rather than a chain.
"""

from __future__ import annotations
