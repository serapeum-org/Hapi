"""What the rest of the package is built on, and what it imports nothing to provide.

Nothing here imports from any other `hapi` subpackage, which is the property that makes it
the bottom of the layering: the simulation calendar, the declarative run configuration, and
the warning-silencing helpers.
"""

from __future__ import annotations
