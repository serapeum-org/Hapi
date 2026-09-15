"""The arrays a model run produces, the routing that produced them, and how to view them.

Running a model used to leave its output as nine separate attributes on the
:class:`~hapi.catchment.Catchment` it was handed, with a private boolean recording which
routing scheme had written them. That made a catchment's state unknowable between runs --
the fields of a finished run and the fields of a half-finished one look the same -- and it
put the interpretation of the arrays (`_maxbas_routed`) on the input object rather than on
the arrays themselves.

:class:`SimulationResults` holds them together instead, with the routing scheme as a field.
A run assigns one to `Catchment.results`, and that is the only place the arrays live -- read
them as `model.results.q_total`. The catchment carries no result attributes of its own.

It also renders and writes them. :meth:`~SimulationResults.animate`,
:meth:`~SimulationResults.save_animation` and :meth:`~SimulationResults.save` used to sit on
`Catchment` -- around three hundred lines of matplotlib, cleopatra and pyramids on the object
whose job is to *assemble inputs*, and the only reason a builder imported a plotting stack at
all. They read result arrays and the run that produced them and touch nothing a catchment
alone holds, so they live here, beside the arrays they render.

`Catchment.plot_hydrograph` deliberately stayed behind: it reads no result array at all. It
compares `Qsim` against the observed gauge record, which is an analysis input this object has
no claim to.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger
from pyramids.dataset import Dataset
from pyramids.dataset import DatasetCollection as Datacube

from hapi.runs import DistributedRun, LumpedRun

if TYPE_CHECKING:
    import matplotlib.animation
    from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

    from hapi.core.period import SimulationPeriod

#: The five per-cell states, in the order the last axis of `state_variables` carries them.
STATE_VARIABLES = ["SP", "SM", "UZ", "LZ", "WC"]

#: `animate` option -> (the attribute it reads, the default title). Options 1-3 are result
#: arrays, 4-8 are slices of the state array, and 9-11 are the meteorological drivers the run
#: was given -- animated on the same grid, which is why they live on the same switch.
_ANIMATION_OPTIONS: dict[int, tuple[str, str]] = {
    1: ("q_total", "Total Discharge"),
    2: ("quz_routed", "Surface Flow"),
    3: ("qlz_translated", "Ground Water Flow"),
    4: ("state:0", "Snow Pack"),
    5: ("state:1", "Soil Moisture"),
    6: ("state:2", "Upper Zone"),
    7: ("state:3", "Lower Zone"),
    8: ("state:4", "Water Content"),
    9: ("meteo:precipitation", "Precipitation"),
    10: ("meteo:evapotranspiration", "ET"),
    11: ("meteo:temperature", "Temperature"),
}

#: `save` option -> the result attribute it writes, for a distributed run. The state options
#: name the slice of `state_variables` rather than a field of their own.
_RASTER_OPTIONS: dict[int, str] = {
    1: "q_total",
    2: "quz_routed",
    3: "qlz_translated",
    4: "state:0",
    5: "state:1",
    6: "state:2",
    7: "state:3",
    8: "state:4",
}


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

    Examples:
        - The kind carries its own name, which is what a run records on its results:
            ```python
            >>> from hapi.results import RoutingKind
            >>> RoutingKind.MUSKINGUM.value
            'muskingum'
            >>> sorted(kind.value for kind in RoutingKind)
            ['lumped', 'maxbas', 'muskingum', 'unrouted']

            ```
    """

    UNROUTED = "unrouted"
    MUSKINGUM = "muskingum"
    MAXBAS = "maxbas"
    LUMPED = "lumped"


@dataclass
class SimulationResults:
    """The arrays one model run produced, the routing that produced them, and their views.

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
            put together and nothing but :meth:`save` and :meth:`animate` reads it, so a run
            that will not look at it need not pay for it. See
            :attr:`~hapi.runs.DistributedRun.keep_state_variables`.
        quz_routed: Upper-zone discharge after routing. `None` until a routing step runs.
            After a MAXBAS run this *is* :attr:`quz`, not a copy of it -- the triangular
            routing works in place and a copy would double the memory of a
            `(rows, cols, time)` array for nothing. Nothing in the package writes through
            the alias, but it is visible (`results.quz_routed is results.quz`), so editing
            one in place edits the other.
        qlz_translated: Lower-zone discharge after translation. `None` until then. Aliases
            :attr:`qlz` after a MAXBAS run, for the same reason as :attr:`quz_routed`.
        q_total: `quz_routed + qlz_translated`. Read it through
            :attr:`outlet_shortcut_valid` rather than assuming what a cell means.
        qout: The outlet hydrograph, when the run computed one, and always `len(period)`
            long. The conceptual model allocates one slot more than it fills, and every
            path that produces a series drops that unwritten trailing slot -- so index 0
            is the model's initial state, not a simulated step, which matters when scoring
            the first value against an observation. The MAXBAS paths sum over the domain and set it directly;
            the Muskingum paths leave it `None` for
            :meth:`~hapi.catchment.Catchment.extract_discharge` to read off the outlet cell,
            which needs the gauge table the engine does not have.
        run: The validated inputs these arrays came from, carried as provenance. It is what
            makes the arrays interpretable on their own: the calendar to index them by, the
            grid to mask them with, and the drivers the animation options can show beside
            them. `None` only for a results object built by hand rather than by a run, in
            which case the presentation methods say so rather than failing on `None`.
        anim: The animation :meth:`animate` last built, or `None`. Not a constructor
            argument.

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
        - Arrays with no run behind them say what is missing rather than failing on `None`:
            ```python
            >>> import numpy as np
            >>> from hapi.results import RoutingKind, SimulationResults
            >>> cube = np.zeros((2, 3, 4), dtype="float32")
            >>> orphan = SimulationResults(RoutingKind.MUSKINGUM, cube, cube, None)
            >>> orphan.save(path="out")
            Traceback (most recent call last):
                ...
            ValueError: these results carry no run...

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
    run: DistributedRun | LumpedRun | None = None
    anim: matplotlib.animation.FuncAnimation | None = field(
        default=None, init=False, repr=False
    )
    # The glyph, not just the animation: cleopatra writes the file through the object that
    # built the frames, so `save_animation` needs the glyph `animate` kept, not its return.
    _animation_glyph: ArrayGlyph | None = field(default=None, init=False, repr=False)

    @property
    def outlet_shortcut_valid(self) -> bool:
        """bool: Whether a single cell of :attr:`q_total` is the discharge *at* that cell.

        False for MAXBAS, which routes each cell straight to the outlet and so makes a cell
        a contribution rather than a discharge -- reading the outlet cell of a MAXBAS run
        under-reports the hydrograph, which is what this guards. False for UNROUTED too:
        there is no `q_total` yet, so there is no cell to read and no shortcut to take.

        Examples:
            - Muskingum accumulates downstream, so a cell is a discharge:
                ```python
                >>> import numpy as np
                >>> from hapi.results import RoutingKind, SimulationResults
                >>> cube = np.zeros((2, 3, 4), dtype="float32")
                >>> muskingum = SimulationResults(RoutingKind.MUSKINGUM, cube, cube, None)
                >>> muskingum.outlet_shortcut_valid
                True

                ```
            - MAXBAS and unrouted arrays do not support the shortcut:
                ```python
                >>> import numpy as np
                >>> from hapi.results import RoutingKind, SimulationResults
                >>> cube = np.zeros((2, 3, 4), dtype="float32")
                >>> [
                ...     SimulationResults(kind, cube, cube, None).outlet_shortcut_valid
                ...     for kind in (RoutingKind.MAXBAS, RoutingKind.UNROUTED)
                ... ]
                [False, False]

                ```
        """
        return self.routing not in (RoutingKind.MAXBAS, RoutingKind.UNROUTED)

    # ------------------------------------------------------------------ #
    # narrowing helpers
    # ------------------------------------------------------------------ #

    def _require_run(self) -> DistributedRun | LumpedRun:
        """Return the run behind these arrays, or say that there is none.

        Returns:
            DistributedRun | LumpedRun: The run that produced these results.

        Raises:
            ValueError: The results were built by hand rather than by a run.
        """
        if self.run is None:
            raise ValueError(
                "these results carry no run, so there is no calendar to index them by and "
                "no grid to write them on; they were built directly rather than by a "
                "`Run.*` entry point"
            )
        return self.run

    def _require_distributed_run(self) -> DistributedRun:
        """Return the run as a distributed one, or say that it is not.

        Returns:
            DistributedRun: The distributed run that produced these results.

        Raises:
            ValueError: There is no run, or it was a lumped one, which has neither a grid
                nor spatial drivers to render.
        """
        run = self._require_run()
        if not isinstance(run, DistributedRun):
            raise ValueError(
                "these results came from a lumped run, which has no grid to render or "
                "write rasters from; use `save` to write them as a CSV instead"
            )
        return run

    def _require_state_variables(self) -> np.ndarray:
        """Return the per-cell state array, or say why it is absent.

        It is `(rows, cols, time, 5)` -- as much memory as every other result field combined
        -- so a run can be asked not to keep it. Only these plotting and saving options read
        it, so the error belongs here, naming the switch rather than failing on `None` inside
        a slice.

        Returns:
            np.ndarray: The state array.

        Raises:
            ValueError: The run was asked not to keep the states.
        """
        if self.state_variables is None:
            raise ValueError(
                "this run did not keep the state variables, so no state option can be "
                "plotted or saved; run it with keep_state_variables=True (the default) if "
                "you need them"
            )
        return self.state_variables

    def _require_field(self, name: str) -> np.ndarray:
        """Return a routed result field, or say which step has not run.

        Args:
            name: The attribute to read.

        Returns:
            np.ndarray: The field.

        Raises:
            ValueError: The field is still `None` because no routing step has run.
        """
        value: np.ndarray | None = getattr(self, name)
        if value is None:
            raise ValueError(
                f"`{name}` is empty because no routing step has filled it; these results "
                f"are {self.routing.value}. `qout` is filled by the `Wrapper` entry points, "
                f"not by the routers -- summing the domain is not a routing step -- so "
                f"results routed by calling `DistributedRRM` directly do not carry one"
            )
        return value

    def _step_bounds(
        self,
        period: SimulationPeriod,
        start: str | dt.datetime,
        end: str | dt.datetime,
        fmt: str,
        inclusive: bool,
    ) -> tuple[int, int]:
        """Resolve two dates to positions in the run's calendar.

        Args:
            period: The span the run covered.
            start: First date, or `""` for the first step.
            end: Last date, or `""` for the last step.
            fmt: `strptime` format a string date is read with.
            inclusive: Whether `end` itself is included in the range.

        Returns:
            tuple[int, int]: The half-open `(start, end)` positions.

        Raises:
            ValueError: A date is not a step of the run's calendar.
        """
        index = period.date_index
        if start == "":
            start = index[0]
        elif isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)
        if end == "":
            end = index[-1]
        elif isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)

        for label, value in (("start", start), ("end", end)):
            if not (index == value).any():
                raise ValueError(
                    f"{label} date {value} is not a step of this run, which covers "
                    f"{index[0]} to {index[-1]}"
                )

        start_i = int(np.nonzero(index == start)[0][0])
        end_i = int(np.nonzero(index == end)[0][0]) + (1 if inclusive else 0)
        return start_i, end_i

    def _select(self, option: str, start_i: int, end_i: int) -> np.ndarray:
        """Slice the array an option names out of the results or the run's drivers.

        Args:
            option: Either an attribute name, `"state:<i>"` for a slice of the state array,
                or `"meteo:<name>"` for one of the run's drivers.
            start_i: First step.
            end_i: One past the last step.

        Returns:
            np.ndarray: The `(rows, cols, time)` slice.
        """
        if option.startswith("state:"):
            layer = int(option.split(":")[1])
            return self._require_state_variables()[:, :, start_i:end_i, layer]
        if option.startswith("meteo:"):
            run = self._require_distributed_run()
            driver: np.ndarray = getattr(run.meteo, option.split(":")[1])
            return driver[:, :, start_i:end_i]
        return self._require_field(option)[:, :, start_i:end_i]

    # ------------------------------------------------------------------ #
    # presentation
    # ------------------------------------------------------------------ #

    def animate(
        self,
        start: str | dt.datetime,
        end: str | dt.datetime,
        fmt: str = "%Y-%m-%d",
        option: int = 1,
        gauges: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> matplotlib.animation.FuncAnimation:
        """Animate a result array or one of the run's drivers over the spatial domain.

        Cells outside the catchment domain are masked on a copy of the data, so the arrays
        held here are never modified. The animation title defaults to the selected variable's
        name; an explicit `title=` keyword argument overrides it.

        Args:
            start: Starting date of the animation.
            end: End date of the animation.
            fmt: Format a string date is read with. Default is "%Y-%m-%d".
            option: Variable to animate. 1 - Total discharge, 2 - Surface flow (the routed
                upper zone), 3 - Ground water flow (the translated lower zone), 4 - Snow
                pack, 5 - Soil moisture, 6 - Upper zone, 7 - Lower zone, 8 - Water content,
                9 - Precipitation, 10 - ET, 11 - Temperature. Default is 1. Options 4-8 are
                the state variables and 9-11 are the run's own drivers.
            gauges: Gauge table to overlay, as `Catchment.GaugesTable`. It must carry `id`,
                `cell_row` and `cell_col` columns. `None`, the default, draws no gauges.
                This used to be a `bool` that reached back onto the catchment for the table;
                the table is an analysis input, so it is passed in.
            **kwargs: Additional keyword arguments passed to `ArrayGlyph.animate`. Loose
                styling keywords still accepted: title (str), title_size (int), cmap (str),
                vmin (float), vmax (float), figsize (tuple), ticks_spacing (int),
                cbar_label (str), cbar_label_size (int), cbar_length (float),
                cbar_orientation (str).
                Styling that cleopatra moved onto typed group objects is passed as those
                objects instead, and a keyword cleopatra no longer knows raises rather than
                being ignored: playback=`Animation` (0.38: was interval / frame_label /
                cell_value_text_colors / data_getter), color=`ColorScaling` (0.30: was
                color_scale / gamma / bounds / midpoint), cells=`CellValues` (was
                display_cell_value / num_size / background_color_threshold),
                contour=`Contour` (was levels), data_style=`DataStyle` (was style /
                hillshade). See
                `cleopatra.glyphs.gridded.array_glyph.ArrayGlyph.animate` for the full list.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object, also kept on
            :attr:`anim` so :meth:`save_animation` can write it.

        Raises:
            ValueError: `option` is not between 1 and 11, the results carry no distributed
                run, or a state option was asked for on a run that dropped the states.

        Examples:
            - An option outside the table is refused before any array is touched, and
              before the plotting stack is imported:
                ```python
                >>> import numpy as np
                >>> from hapi.results import RoutingKind, SimulationResults
                >>> cube = np.zeros((2, 3, 4), dtype="float32")
                >>> results = SimulationResults(
                ...     RoutingKind.MUSKINGUM, cube, cube, None, q_total=cube
                ... )
                >>> results.animate("2009-01-01", "2009-01-02", option=99)
                Traceback (most recent call last):
                    ...
                ValueError: the option parameter takes a value between 1 and 11, given: 99

                ```
            - Arrays with no run behind them have no calendar and no grid, so they say so
              rather than failing on `None` several frames in:
                ```python
                >>> import numpy as np
                >>> from hapi.results import RoutingKind, SimulationResults
                >>> cube = np.zeros((2, 3, 4), dtype="float32")
                >>> orphan = SimulationResults(RoutingKind.MUSKINGUM, cube, cube, None)
                >>> orphan.animate("2009-01-01", "2009-01-02", option=1)
                Traceback (most recent call last):
                    ...
                ValueError: these results carry no run...

                ```

        See Also:
            save_animation: Writes the animation this builds.
            save: Writes the same arrays as rasters or a CSV instead of rendering them.
        """
        if option not in _ANIMATION_OPTIONS:
            raise ValueError(
                f"the option parameter takes a value between 1 and "
                f"{max(_ANIMATION_OPTIONS)}, given: {option}"
            )

        run = self._require_distributed_run()
        start_i, end_i = self._step_bounds(run.period, start, end, fmt, inclusive=False)

        source, title = _ANIMATION_OPTIONS[option]
        arr = self._select(source, start_i, end_i)

        # Masked on a copy, so plotting never mutates the result arrays -- and on a float
        # copy, because the mask writes NaN and the meteo options read `MeteoInputs` cubes
        # "as stored", which an integer driver raster would make unassignable.
        arr = (
            arr.copy()
            if np.issubdtype(arr.dtype, np.floating)
            else arr.astype(np.float32)
        )
        arr[np.isnan(run.flow_network.flow_acc_arr), :] = np.nan

        time = run.period.date_index[start_i:end_i]

        # cleopatra pulls in matplotlib, and this module is imported by the engines
        # (`distrrm`, `wrapper`, `run`). Importing it here keeps a model run free of a
        # plotting stack it never uses -- which is the property that made moving these
        # methods off `Catchment` worth doing rather than just tidier. It sits below the
        # checks above so a rejected option or an out-of-range date does not pay for it
        # either.
        from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay

        if gauges is not None:
            # animate expects a 3-column array: [value to display, cell row, cell column].
            # cleopatra 0.30 stopped accepting a bare array; it must be wrapped in a
            # PointOverlay, which also carries the marker/label styling.
            kwargs["points"] = PointOverlay(
                gauges[["id", "cell_row", "cell_col"]].to_numpy()
            )

        # animate iterates over the first dimension, so move the time axis to the front
        array = ArrayGlyph(np.moveaxis(arr, -1, 0))
        # the option title is a default; an explicit title= kwarg wins
        kwargs.setdefault("title", title)
        # cleopatra is untyped, so name what it hands back rather than letting `Any` leak
        # out of a public signature.
        anim: matplotlib.animation.FuncAnimation = array.animate(time, **kwargs)

        self._animation_glyph = array
        self.anim = anim

        return anim

    def save_animation(self, path: str | Path, fps: int = 2) -> None:
        """Save the animation built by :meth:`animate`.

        The output format is determined by the file extension. GIF uses PillowWriter;
        mov/avi/mp4 require FFmpeg to be installed.

        Args:
            path: Output file path. The extension determines the format (gif, mov, avi, mp4).
            fps: Frames per second. Default is 2.

        Raises:
            ValueError: :meth:`animate` has not been called yet, or the file format is not
                supported.
            FileNotFoundError: A video format is requested but FFmpeg is not installed.

        Examples:
            - There is nothing to write until :meth:`animate` has built it:
                ```python
                >>> import numpy as np
                >>> from hapi.results import RoutingKind, SimulationResults
                >>> cube = np.zeros((2, 3, 4), dtype="float32")
                >>> results = SimulationResults(RoutingKind.MUSKINGUM, cube, cube, None)
                >>> results.anim is None
                True
                >>> results.save_animation("flow.gif")
                Traceback (most recent call last):
                    ...
                ValueError: There is no animation to save, call `animate` first

                ```

        See Also:
            animate: Builds the animation this writes.
        """
        if self._animation_glyph is None:
            raise ValueError("There is no animation to save, call `animate` first")
        self._animation_glyph.save_animation(path, fps=fps)

    def save(
        self,
        path: str | Path = "",
        result: int = 1,
        start: str | dt.datetime = "",
        end: str | dt.datetime = "",
        prefix: str = "",
        fmt: str = "%Y-%m-%d",
        flow_acc_path: str | Path = "",
    ) -> None:
        """Write the results to disk: one raster per step, or a CSV for a lumped run.

        Which of the two happens is read off :attr:`routing` rather than passed in -- a
        lumped run has no grid to write rasters on, and that is a property of the results.

        Args:
            path: Output directory for a distributed run (created if it does not exist), or
                the CSV file itself for a lumped one, as a `str` or a `Path`. Default is
                "", the working directory.
            result: What to write. Distributed: 1 - Total discharge, 2 - Surface flow (the
                routed upper zone), 3 - Ground water flow (the translated lower zone),
                4 - Snow pack, 5 - Soil moisture, 6 - Upper zone, 7 - Lower zone,
                8 - Water content. Lumped: 1 - simulated discharge, 2 - upper zone,
                3 - lower zone, 4 - the five states, 5 - all of them. Default is 1.
            start: Start of the output period. A string is parsed with `fmt`. If empty, the
                run's first step.
            end: End of the output period, inclusive. If empty, the run's last step.
            prefix: Prefix for the raster file names. Default is "Result_".
            fmt: Date format `start` and `end` are parsed with. Default is "%Y-%m-%d".
            flow_acc_path: The flow-accumulation raster, used as the georeferencing template
                for the written rasters. Required for a distributed run: `FlowNetwork` keeps
                the accumulation *array* but not its projection, so the grid has to be read
                back from the file.

        Raises:
            TypeError: `path` is neither a `str` nor a `Path`. `outputs.results_dir` is
                optional in a run configuration, so a caller forwarding it can hold None.
            ValueError: `result` is not a valid option, `flow_acc_path` is missing on a
                distributed run, or the results carry no run to date them by.

        Examples:
            - A lumped run writes a CSV, dated by the run's own calendar. Option 1 is the
              simulated discharge, which for a lumped run is `q_total` itself:
                ```python
                >>> import os, tempfile
                >>> from pathlib import Path
                >>> import numpy as np
                >>> from hapi.conceptual import ConceptualModelSetup, ParameterSet
                >>> from hapi.core.period import SimulationPeriod
                >>> from hapi.results import RoutingKind, SimulationResults
                >>> from hapi.conceptual.hbv_bergestrom92 import HBVBergestrom92
                >>> from hapi.runs import LumpedRun
                >>> period = SimulationPeriod.parse("2009-01-01", "2009-01-03")
                >>> run = LumpedRun(
                ...     period=period,
                ...     data=np.ones((len(period), 4)),
                ...     parameters=ParameterSet(np.ones(12), snow=False, maxbas=False),
                ...     model_setup=ConceptualModelSetup(
                ...         HBVBergestrom92(), 100.0, [0.0] * 5, 1.0
                ...     ),
                ... )
                >>> discharge = np.array([1.5, 2.5, 3.5])
                >>> results = SimulationResults(
                ...     RoutingKind.LUMPED, discharge, discharge, None,
                ...     q_total=discharge, run=run,
                ... )
                >>> path = os.path.join(tempfile.mkdtemp(), "q.csv")
                >>> results.save(path=path, result=1)
                >>> print(Path(path).read_text().strip())
                date,Qsim
                '2009-01-01',1.500
                '2009-01-02',2.500
                '2009-01-03',3.500

                ```
            - `path` is checked before anything else, because a run configuration's
              `outputs.results_dir` is optional and a caller can forward `None`:
                ```python
                >>> import numpy as np
                >>> from hapi.results import RoutingKind, SimulationResults
                >>> cube = np.zeros((2, 3, 4), dtype="float32")
                >>> results = SimulationResults(RoutingKind.MUSKINGUM, cube, cube, None)
                >>> results.save(path=None)
                Traceback (most recent call last):
                    ...
                TypeError: path must be a str or Path naming a directory (distributed) or a file (lumped), got NoneType

                ```

        See Also:
            animate: Renders the same arrays instead of writing them.
            hapi.runs.DistributedRun.keep_state_variables: Whether the state options have
                anything to write.
        """
        # Checked rather than delegated: `outputs.results_dir` is optional in a run
        # configuration, so a caller forwarding it can hold None, and the writers would
        # only fail much later. `Path` is as welcome as `str` -- Path("") is `.`, the
        # same working directory the empty default names.
        if not isinstance(path, str | Path):
            raise TypeError(
                f"path must be a str or Path naming a directory (distributed) or a "
                f"file (lumped), got {type(path).__name__}"
            )

        run = self._require_run()
        start_i, end_i = self._step_bounds(run.period, start, end, fmt, inclusive=True)

        if self.routing is RoutingKind.LUMPED:
            self._save_csv(run.period, path, result, start_i, end_i)
        else:
            self._save_rasters(
                run.period, path, result, start_i, end_i, prefix, flow_acc_path
            )

        logger.debug("Data is saved successfully")

    def _save_rasters(
        self,
        period: SimulationPeriod,
        path: str | Path,
        result: int,
        start_i: int,
        end_i: int,
        prefix: str,
        flow_acc_path: str | Path,
    ) -> None:
        """Write one GeoTIFF per step off the flow-accumulation raster's grid.

        Args:
            period: The run's calendar, which names the files.
            path: Destination directory, created if it does not exist.
            result: Which array to write. See :meth:`save`.
            start_i: First step.
            end_i: One past the last step.
            prefix: File-name prefix.
            flow_acc_path: The georeferencing template.

        Raises:
            ValueError: `flow_acc_path` is empty, or `result` is not between 1 and 8.
        """
        if not flow_acc_path:
            raise ValueError(
                "writing rasters needs a georeferencing template; pass flow_acc_path, the "
                "flow-accumulation raster the model was built on"
            )
        if result not in _RASTER_OPTIONS:
            raise ValueError(
                f" The result parameter takes a value between 1 and "
                f"{max(_RASTER_OPTIONS)}, given: {result}"
            )

        arr = self._select(_RASTER_OPTIONS[result], start_i, end_i)

        if prefix == "":
            prefix = "Result_"

        # `path` names a directory here, unlike the CSV branch where it is the file itself.
        # Joined rather than concatenated: the old `path + prefix` wrote
        # `some/dirResult_2009-01-01.tif` for any directory given without a trailing
        # separator, which is how a directory is normally written. `Path("")` is `.`,
        # so the empty default still writes into the working directory.
        destination = Path(path)
        if path and not destination.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        names = [
            destination / f"{prefix}{str(i)[:10]}.tif"
            for i in period.date_index[start_i:end_i]
        ]

        # Closed when the write finishes: on Windows an open GDAL handle keeps a lock on
        # the file, so a script that saves rasters and then moves or deletes the template
        # fails, and a repeated `save` accumulates handles.
        with Dataset.read_file(flow_acc_path) as src:
            # from_dataset is pyramids' named constructor for an in-memory scaffold off a
            # template raster; the bare Datacube(src, time_length=) form it replaced is
            # kept only as a legacy fallback upstream.
            cube = Datacube.from_dataset(src, arr.shape[2])
            # A copy, not the `moveaxis` view: `arr` is a slice of a result array, and
            # handing a view to a writer that may normalise no-data in place would edit
            # the results this call is only supposed to read. `np.array(copy=True)` rather
            # than `ascontiguousarray`, which returns the input untouched when it is
            # already contiguous -- and a single-step range is, because numpy ignores
            # size-1 dimensions when testing that. `save(start=d, end=d)` therefore kept
            # handing out a view, which is exactly the case this guards.
            cube.values = np.array(np.moveaxis(arr, -1, 0), order="C", copy=True)
            cube.to_file(names)

    def _save_csv(
        self,
        period: SimulationPeriod,
        path: str | Path,
        result: int,
        start_i: int,
        end_i: int,
    ) -> None:
        """Write a lumped run's series to a CSV.

        Args:
            period: The run's calendar, which indexes the frame.
            path: The CSV file to write.
            result: Which series to write. See :meth:`save`.
            start_i: First step.
            end_i: One past the last step.

        Raises:
            ValueError: `result` is not between 1 and 5.
        """
        if result not in (1, 2, 3, 4, 5):
            raise ValueError(
                f"in lumped mode the result parameter takes a value between 1 and 5, "
                f"given: {result}"
            )

        # The run's own calendar, not a fresh daily `date_range`: the old branch hard-coded
        # `freq="D"`, so an hourly lumped run wrote a daily index against hourly values.
        data = pd.DataFrame(index=period.date_index[start_i:end_i])
        data["date"] = ["'" + str(i)[:10] + "'" for i in data.index]

        if result in (1, 5):
            # For a lumped run the total discharge *is* `Qsim`; `Run.run_lumped` only wraps
            # this same array in a frame to put on the model.
            data["Qsim"] = self._require_field("q_total")[start_i:end_i]
        if result in (2, 5):
            data["Quz"] = self.quz[start_i:end_i]
        if result in (3, 5):
            data["Qlz"] = self.qlz[start_i:end_i]
        if result in (4, 5):
            data[STATE_VARIABLES] = self._require_state_variables()[start_i:end_i, :]

        data.to_csv(path, index=False, float_format="%.3f")
