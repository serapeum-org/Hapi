"""The span of time a model run covers, and everything the temporal resolution implies.

Six attributes on :class:`~hapi.catchment.Catchment` used to describe one thing: `start`,
`end` and `temporal_resolution` were given, and `date_index`, `dt` and `conversion_factor`
were derived from them in the constructor and then stored beside them as if they were
independent. Storing a derivation is how the three drift apart -- reassigning `end` left
`date_index` describing the old span, with nothing to notice -- and it is why the same
`pd.date_range` branch was written out four times across the package.

:class:`SimulationPeriod` holds the three inputs and derives the rest on read, so they cannot
disagree. It is frozen: a run covers the period it was built for, and a model that needs a
different one gets a new period rather than a mutated one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import pandas as pd

#: mm over a km2 in a day, expressed as m3/s: (1000 * 24 * 60 * 60) / (1000 ** 2).
CONVERSION_FACTOR = (1000 * 24 * 60 * 60) / (1000**2)

#: The temporal resolutions the model runs at, mapped to the pandas offset alias each uses.
#: Adding one means deciding its `conversion_factor` too -- see :attr:`SimulationPeriod.freq`.
RESOLUTIONS: dict[str, str] = {"daily": "D", "hourly": "h"}

TemporalResolution = Literal["daily", "hourly"]


@dataclass(frozen=True)
class SimulationPeriod:
    """The span a run covers, and the calendar and unit factors it implies.

    Attributes:
        start: First step of the simulation.
        end: Last step of the simulation.
        temporal_resolution: `"daily"` or `"hourly"`, lower-cased on construction.

    Examples:
        - The calendar is derived, so it always matches the span:
            ```python
            >>> from hapi.period import SimulationPeriod
            >>> period = SimulationPeriod.parse("2009-01-01", "2009-01-10")
            >>> len(period)
            10
            >>> period.date_index[0].strftime("%Y-%m-%d")
            '2009-01-01'

            ```
        - An hourly period covers the same span with a different step:
            ```python
            >>> from hapi.period import SimulationPeriod
            >>> hourly = SimulationPeriod.parse(
            ...     "2009-01-01", "2009-01-02", temporal_resolution="Hourly"
            ... )
            >>> len(hourly)
            25
            >>> round(hourly.conversion_factor, 1)
            3.6

            ```
        - It is frozen, so a derived value can never be left describing a different span:
            ```python
            >>> from hapi.period import SimulationPeriod
            >>> period = SimulationPeriod.parse("2009-01-01", "2009-01-10")
            >>> period.end = "2010-01-01"  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            dataclasses.FrozenInstanceError: cannot assign to field 'end'

            ```
    """

    start: dt.datetime
    end: dt.datetime
    temporal_resolution: TemporalResolution = "daily"

    def __post_init__(self):
        """Normalise the resolution and check the span runs forwards.

        Raises:
            TypeError: `temporal_resolution` is not a string.
            ValueError: The resolution is not one of :data:`RESOLUTIONS`, or `end` is before
                `start`.
        """
        if not isinstance(self.temporal_resolution, str):
            raise TypeError(
                f"temporal_resolution must be a string, got "
                f"{type(self.temporal_resolution).__name__}"
            )
        resolution = self.temporal_resolution.lower()
        if resolution not in RESOLUTIONS:
            raise ValueError(
                f"available temporal resolutions are {', '.join(map(repr, RESOLUTIONS))}, "
                f"got {self.temporal_resolution!r}"
            )
        object.__setattr__(self, "temporal_resolution", resolution)

        # A backwards span produces an empty `date_index`, which then fails much later as a
        # zero-length driver mismatch that names neither date.
        if self.end < self.start:
            raise ValueError(
                f"the simulation ends before it starts: {self.start:%Y-%m-%d} to "
                f"{self.end:%Y-%m-%d}"
            )

    @classmethod
    def parse(
        cls,
        start: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        temporal_resolution: str = "Daily",
    ) -> SimulationPeriod:
        """Build a period from the string dates a configuration or a script supplies.

        Args:
            start: Start date.
            end: End date.
            fmt: `strptime` format both dates are read with.
            temporal_resolution: `"Daily"` or `"Hourly"`, matched case-insensitively.

        Returns:
            SimulationPeriod: The parsed period.

        Raises:
            ValueError: A date does not match `fmt`, or the span runs backwards.

        Examples:
            ```python
            >>> from hapi.period import SimulationPeriod
            >>> SimulationPeriod.parse("01/2009/01", "10/2009/01", fmt="%d/%Y/%m").days
            10

            ```
        """
        return cls(
            dt.datetime.strptime(start, fmt),
            dt.datetime.strptime(end, fmt),
            temporal_resolution,  # type: ignore[arg-type]
        )

    @property
    def freq(self) -> str:
        """str: The pandas offset alias for this resolution."""
        return RESOLUTIONS[self.temporal_resolution]

    @property
    def date_index(self) -> pd.DatetimeIndex:
        """pandas.DatetimeIndex: One entry per step, from :attr:`start` to :attr:`end`.

        Derived rather than stored: this is the value that used to be computed in the
        constructor and could then outlive a change to the span it described.
        """
        return pd.date_range(self.start, self.end, freq=self.freq)

    @property
    def days(self) -> int:
        """int: Number of steps the period covers."""
        return len(self.date_index)

    @property
    def conversion_factor(self) -> float:
        """float: Depth-to-discharge factor -- mm over the catchment to m3/s at this step."""
        return CONVERSION_FACTOR if self.temporal_resolution == "daily" else (
            CONVERSION_FACTOR / 24
        )

    @property
    def dt(self) -> float:
        """float: The routing time-step factor.

        One for both resolutions today. It is a property rather than a stored `1` so the
        Muskingum routing has a single place to read it from; whether an hourly run should
        route with a different value is an open question, recorded in the planning notes
        rather than silently answered here.
        """
        return 1.0

    def __len__(self) -> int:
        """int: Number of steps, so `len(period)` reads as the span."""
        return self.days
