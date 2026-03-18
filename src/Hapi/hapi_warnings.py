"""Custom warning classes and utilities for silencing warnings.

The ``hapi_warnings`` module provides custom warning types used
throughout the Hapi package and helper functions to suppress
specific warning categories during runtime.

Examples
--------
    >>> from Hapi.hapi_warnings import InstabilityWarning
    >>> import warnings
    >>> warnings.warn(
    ...     "Simulation diverged at step 5",
    ...     InstabilityWarning,
    ... )
"""
from __future__ import annotations

import warnings


class InstabilityWarning(UserWarning):
    """Warning issued when numerical results may be unstable.

    ``InstabilityWarning`` is a subclass of ``UserWarning`` that
    flags potential numerical instability in hydrological simulation
    results, such as diverging flows or extreme parameter values.

    Examples:
        >>> from Hapi.hapi_warnings import InstabilityWarning
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     warnings.warn("unstable result", InstabilityWarning)
        ...     len(w)
        1
    """

    pass


warnings.simplefilter("always", InstabilityWarning)
warnings.simplefilter("always", UserWarning)


def SilencePandasWarning():
    """Silence pandas ``FutureWarning`` messages.

    Configures the warnings filter to ignore all
    ``FutureWarning`` instances, which are commonly emitted by
    pandas when deprecated APIs are used.

    Examples:
        >>> from Hapi.hapi_warnings import SilencePandasWarning
        >>> SilencePandasWarning()
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
