"""The conceptual model a run executes, and the parameters it is configured with.

Seven attributes on :class:`~hapi.catchment.Catchment` described one thing -- the
rainfall-runoff model, configured and ready to run -- and were set by two readers that did
not know about each other: `read_parameters` set `parameters`, `snow` and `maxbas`, while
`read_lumped_model` set `lumped_model`, `area`, `initial_cond` and `q_init`.

That mattered because `snow` and `maxbas` are not tags. They *determine how many parameters
there must be* (:data:`PARAMETER_COUNTS`), and that rule was enforced in exactly one place:
inside `read_parameters`, on the one assignment a calibration never makes. A calibration
replaces the parameter array once per trial vector, and none of those replacements were
checked -- so a spatial-distribution function producing the wrong width reached the per-cell
loop, where it fails as an index error far from the call that caused it.

:class:`ParameterSet` holds the array with the `(snow, maxbas)` pair that fixes its width and
enforces the rule in its constructor, so every route to a parameter set goes through the same
check. :class:`ConceptualModelSetup` holds the other half. They are two objects rather than
one because they are read independently, in either order -- pairing them would force a
half-built object to exist, which is the thing being removed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hapi.rrm.base_model import BaseConceptualModel

#: (snow, maxbas) -> how many parameters the conceptual model reads in that configuration.
#: The snow routine adds five; MAXBAS replaces the two Muskingum parameters with one.
PARAMETER_COUNTS: dict[tuple[bool, bool], int] = {
    (True, True): 16,
    (False, True): 11,
    (True, False): 17,
    (False, False): 12,
}


def parameter_count(parameters: np.ndarray | list) -> int:
    """Count the parameters a set carries, whatever shape it is stored in.

    A distributed set is a `(rows, cols, n)` array and a lumped one a flat sequence, so the
    count is read from the shape rather than from a mode flag the caller has to supply --
    which is how the check used to depend on `spatial_resolution`.

    Args:
        parameters: The parameter set, 3D for distributed or 1D for lumped.

    Returns:
        int: Number of parameters per cell (distributed) or in total (lumped).

    Examples:
        ```python
        >>> import numpy as np
        >>> from hapi.conceptual import parameter_count
        >>> parameter_count(np.zeros((13, 14, 12)))
        12
        >>> parameter_count([1.0] * 12)
        12

        ```
    """
    array = np.asarray(parameters)
    return array.shape[2] if array.ndim == 3 else len(array)


def validate_parameter_count(
    parameters: np.ndarray | list, snow: bool, maxbas: bool
) -> None:
    """Check a parameter set carries the count `(snow, maxbas)` requires.

    Split out of the spec's constructor so a builder can run it the moment the parameters and
    the configuration are both known -- which is inside `read_parameters`, before the
    conceptual model itself has been read. Waiting for the whole spec would move a
    wrong-width error to whichever later call completed it.

    Args:
        parameters: The parameter set.
        snow: Whether the snow routine runs.
        maxbas: Whether the set carries a MAXBAS value.

    Raises:
        ValueError: The count does not match.

    Examples:
        ```python
        >>> import numpy as np
        >>> from hapi.conceptual import validate_parameter_count
        >>> validate_parameter_count(np.zeros((2, 2, 5)), snow=False, maxbas=False)
        Traceback (most recent call last):
            ...
        ValueError: a model with snow=False, maxbas=False takes 12 parameters, got 5

        ```
    """
    expected = PARAMETER_COUNTS[(bool(snow), bool(maxbas))]
    actual = parameter_count(parameters)
    if actual != expected:
        raise ValueError(
            f"a model with snow={bool(snow)}, maxbas={bool(maxbas)} takes {expected} "
            f"parameters, got {actual}"
        )


def validate_initial_cond(initial_cond: Any) -> list:
    """Check the initial state is the five values the conceptual models read.

    Exposed as a function, not only as part of the spec's constructor, so a builder can check
    this half as it arrives rather than waiting for the other half to turn up -- which is what
    keeps the error at the `read_lumped_model` call that supplied it.

    Args:
        initial_cond: The candidate initial state.

    Returns:
        list: `initial_cond`, unchanged.

    Raises:
        TypeError: It is not a list.
        ValueError: It does not hold exactly five values.
    """
    if not isinstance(initial_cond, list):
        raise TypeError(
            f"init_st should be of type list, got {type(initial_cond).__name__}"
        )
    if len(initial_cond) != 5:
        raise ValueError(
            f"state variables are 5 and the given initial values are {len(initial_cond)}"
        )
    return initial_cond


def validate_q_init(q_init: Any) -> float | None:
    """Check the initial discharge is a float or absent.

    Args:
        q_init: The candidate initial discharge.

    Returns:
        float | None: `q_init`, unchanged.

    Raises:
        TypeError: It is neither None nor a float.
    """
    if q_init is not None and not isinstance(q_init, float):
        raise TypeError(f"q_init should be of type float, got {type(q_init).__name__}")
    return q_init


@dataclass(frozen=True)
class ParameterSet:
    """A parameter set together with the configuration that fixes its width.

    Exactly what `read_parameters` produces. `snow` and `maxbas` are not tags travelling
    beside the array -- they *determine how many parameters there must be*
    (:data:`PARAMETER_COUNTS`), so holding the three together is what lets the rule be checked
    on construction instead of at one call site.

    Frozen: a calibration explores parameter sets by the thousand, and each is a different
    set rather than a mutation of the last. :meth:`with_values` returns a new one and re-runs
    the check, so a distribution function producing the wrong width fails on the trial that
    produced it rather than as an index error inside the per-cell loop.

    Attributes:
        values: `(rows, cols, n)` for a distributed run, a flat sequence for a lumped one.
        snow: Whether the snow routine runs.
        maxbas: Whether the set carries a MAXBAS value instead of Muskingum's two.

    Examples:
        - The width has to match the configuration:
            ```python
            >>> import numpy as np
            >>> from hapi.conceptual import ParameterSet
            >>> ParameterSet(np.zeros((2, 2, 12))).count
            12
            >>> ParameterSet(np.zeros((2, 2, 5)))
            Traceback (most recent call last):
                ...
            ValueError: a model with snow=False, maxbas=False takes 12 parameters, got 5

            ```
    """

    values: np.ndarray | list
    snow: bool = False
    maxbas: bool = False

    def __post_init__(self):
        """Check the width matches `(snow, maxbas)`.

        Raises:
            ValueError: The count does not match.
        """
        validate_parameter_count(self.values, self.snow, self.maxbas)

    @property
    def count(self) -> int:
        """int: Number of parameters the set carries. See :func:`parameter_count`."""
        return parameter_count(self.values)

    def with_values(self, values: np.ndarray | list) -> ParameterSet:
        """Return the same configuration with a different parameter array.

        Args:
            values: The replacement parameter array.

        Returns:
            ParameterSet: A new set carrying `values`, width already checked.

        Raises:
            ValueError: The replacement does not carry the required count.
        """
        return replace(self, values=values)


@dataclass(frozen=True)
class ConceptualModelSetup:
    """The conceptual model instance and the state it starts from.

    Exactly what `read_lumped_model` produces. Kept apart from :class:`ParameterSet` because
    the two are read independently and either may be read first -- pairing them would force a
    half-built object to exist, which is the thing this refactor is removing.

    Attributes:
        model: The conceptual model instance whose `simulate` is called per cell.
        area: Catchment area in km2.
        initial_cond: Initial state values `[sp, sm, uz, lz, wc]`.
        q_init: Initial discharge in m3/s, or None to let the model choose.

    Examples:
        ```python
        >>> from hapi.conceptual import ConceptualModelSetup
        >>> from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
        >>> setup = ConceptualModelSetup(
        ...     HBVBergestrom92(), 1530.0, [0, 10, 10, 10, 0], q_init=5.0
        ... )
        >>> setup.area, setup.q_init
        (1530.0, 5.0)

        ```
    """

    model: BaseConceptualModel
    area: float | int
    initial_cond: list
    q_init: float | None = None

    def __post_init__(self):
        """Check the initial state and discharge.

        Raises:
            TypeError: `initial_cond` is not a list, or `q_init` is neither None nor a float.
            ValueError: `initial_cond` does not hold five values.
        """
        validate_initial_cond(self.initial_cond)
        validate_q_init(self.q_init)


@dataclass(frozen=True)
class ParameterBounds:
    """The search space a calibration explores, and the configuration it explores it under.

    Exactly what `read_parameters_bound` produces. It carries the same `(snow, maxbas)` pair
    as :class:`ParameterSet` because a calibration has no parameter file to read it from --
    the bounds are where the configuration enters, and every trial vector the optimiser
    produces is checked against it.

    Attributes:
        lower: Lower bound per parameter.
        upper: Upper bound per parameter.
        snow: Whether the snow routine runs.
        maxbas: Whether the parameter vector carries a MAXBAS value.

    Examples:
        ```python
        >>> from hapi.conceptual import ParameterBounds
        >>> bounds = ParameterBounds([0.0] * 12, [1.0] * 12)
        >>> len(bounds)
        12

        ```
    """

    lower: np.ndarray | list
    upper: np.ndarray | list
    snow: bool = False
    maxbas: bool = False

    def __post_init__(self):
        """Check the two bounds describe the same parameters.

        Raises:
            ValueError: The bounds are different lengths.
        """
        if len(self.lower) != len(self.upper):
            raise ValueError(
                f"the length of UB should be the same as LB, got {len(self.upper)} and "
                f"{len(self.lower)}"
            )
        object.__setattr__(self, "lower", np.array(self.lower))
        object.__setattr__(self, "upper", np.array(self.upper))

    def __len__(self) -> int:
        """int: Number of parameters being calibrated."""
        return len(self.lower)
