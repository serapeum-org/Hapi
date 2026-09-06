"""The validated inputs one run needs, as a type rather than a convention.

:class:`~hapi.catchment.Catchment` is a *builder*: it is constructed empty and filled by
`read_*` calls that may run in any order, some of which a given run never needs. So every input
on it is declared `X | None`, and that is honest -- a catchment half-way through assembly really
does have no flow network.

The run layer needs the opposite thing: a catchment that is *finished*. Conflating the two is
what made the engines dereference `X | None` on every line, why four modules were excused from
mypy, and -- worse -- why "has this been validated?" was a question you answered by remembering
which entry point you came through. `Calibration` went straight to :class:`~hapi.wrapper.Wrapper`
and so skipped every check `Run` performed, on the one path that rebuilds the parameter array
thousands of times.

:class:`DistributedRun` and :class:`LumpedRun` are that finished thing. Their fields are not
optional, and :meth:`DistributedRun.from_model` is the only way to get one: constructing it *is*
the validation. Nothing reaches the engines without passing through it, so the question stops
being one of discipline.

They hold inputs only. Results come back as a return value -- see
:class:`~hapi.results.SimulationResults` -- so a run cannot half-overwrite what it read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hapi.conceptual import ConceptualModelSetup, ParameterSet
from hapi.inputs import FlowNetwork, MeteoInputs, RiverGeometry
from hapi.period import SimulationPeriod

if TYPE_CHECKING:
    from hapi.protocols import CatchmentLike

ROWS_MISMATCH_ERROR = "the parameters must have as many rows as the catchment grid"
COLS_MISMATCH_ERROR = "the parameters must have as many columns as the catchment grid"
GRID_MISMATCH_ERROR = "all input data should have the same number of rows and columns"


def _require(model: CatchmentLike, name: str, hint: str) -> Any:
    """Fetch an input a run cannot do without, or name the reader that supplies it.

    Args:
        model: The catchment being narrowed.
        name: Attribute to fetch.
        hint: How to supply it, quoted in the error.

    Returns:
        Any: The attribute's value.

    Raises:
        ValueError: The attribute is unset.
    """
    value = getattr(model, name, None)
    if value is None:
        raise ValueError(f"this run needs {name}, which is not set on the model; {hint}")
    return value


@dataclass(frozen=True)
class DistributedRun:
    """Everything a distributed run needs, checked and non-optional.

    Attributes:
        period: The span the run covers, and the calendar and factors it implies.
        meteo: The three driver cubes.
        flow_network: The routing network and the grid it defines.
        parameters: The parameter array plus the `(snow, maxbas)` pair fixing its width.
        model_setup: The conceptual model instance and the state it starts from.
        river_geometry: The five hydraulic rasters, when the flood path supplied them. `None`
            on an ordinary distributed run, which never reads them. Absent-or-complete, never
            half-filled -- that is what :class:`~hapi.inputs.RiverGeometry` guarantees.
        skip_hydraulic_cells: Leave river cells unrouted for a 1D hydraulic model. Needs
            `river_geometry` to identify them, checked here rather than in the routing loop.
        flow_path_length: Flow-path length raster, read only by
            :meth:`~hapi.rrm.distrrm.DistributedRRM.route_maxbas_by_path_length`.
    """

    period: SimulationPeriod
    meteo: MeteoInputs
    flow_network: FlowNetwork
    parameters: ParameterSet
    model_setup: ConceptualModelSetup
    river_geometry: RiverGeometry | None = None
    skip_hydraulic_cells: bool = False
    flow_path_length: np.ndarray | None = None

    def __post_init__(self):
        """Check the inputs agree with each other and with the grid.

        Raises:
            ValueError: The drivers or the parameters do not cover the grid, the river geometry
                does not, or a cell skip was asked for with no geometry to identify the river
                cells.
        """
        rows, cols = self.flow_network.rows, self.flow_network.cols

        # The three cubes already agree with each other (settled when MeteoInputs was built);
        # this is the other half -- that they cover the grid, and the period.
        self.meteo.validate_against(rows, cols, self.period.date_index)

        shape = np.asarray(self.parameters.values).shape
        if shape[0] != rows:
            raise ValueError(ROWS_MISMATCH_ERROR)
        if shape[1] != cols:
            raise ValueError(COLS_MISMATCH_ERROR)

        if self.river_geometry is not None and not self.river_geometry.covers(rows, cols):
            raise ValueError(GRID_MISMATCH_ERROR)

        if self.skip_hydraulic_cells and self.river_geometry is None:
            raise ValueError(
                "skipping the hydraulic cells needs the river geometry to identify them, "
                "but none is set; call read_river_geometry first"
            )

    @property
    def parameter_cube(self) -> np.ndarray:
        """np.ndarray: The parameters as the `(rows, cols, n)` cube a distributed run indexes.

        `ParameterSet.values` is a flat sequence for a lumped run and a cube for a distributed
        one, so it is typed as either. On this side of the narrowing it is always the cube --
        `__post_init__` has already read its first two axes -- and saying so here means the
        engines index a real array instead of a union.
        """
        return np.asarray(self.parameters.values)

    @property
    def routing_table(self) -> dict:
        """dict: The flow-direction table, mapping `"row,col"` to the cells draining into it.

        `FlowNetwork` carries it as optional because the MAXBAS paths never route cell to cell.
        Reaching it through here keeps that honest while giving the Muskingum routing a plain
        dict to index.

        Raises:
            ValueError: The network was built without a direction table.
        """
        table = self.flow_network.FDT
        if table is None:
            raise ValueError(
                "cell-to-cell routing needs the flow-direction table, but the flow network "
                "was built without one; pass a flow-direction raster to "
                "FlowNetwork.from_rasters"
            )
        return table

    @classmethod
    def from_model(
        cls,
        model: CatchmentLike,
        *,
        needs_flow_direction: bool = True,
        with_river_geometry: bool = False,
        skip_hydraulic_cells: bool = False,
    ) -> DistributedRun:
        """Narrow a built catchment into a validated distributed run.

        The single seam every distributed execution path passes through, `Run` and
        `Calibration` alike. Everything checkable is checked here, so a caller cannot arrive at
        the engines with an unvalidated model and does not have to remember to ask.

        Args:
            model: A catchment with its inputs read.
            needs_flow_direction: Whether the flow-direction raster is required. Cell-to-cell
                routing needs it; MAXBAS sends every cell straight to the outlet and never
                reads it.
            with_river_geometry: Carry the river geometry through, for the flood path.
            skip_hydraulic_cells: Leave the river cells to a hydraulic model.

        Returns:
            DistributedRun: The validated inputs.

        Raises:
            ValueError: A required input is unset, or the inputs disagree.
        """
        flow_network = _require(
            model,
            "flow_network",
            "assign FlowNetwork.from_rasters(...) to model.flow_network",
        )
        if needs_flow_direction:
            if flow_network.flow_dir_arr is None:
                raise ValueError(
                    "this run routes cell to cell and needs a flow-direction raster, but the "
                    "flow network was built without one; pass it to FlowNetwork.from_rasters"
                )
            # `FlowNetwork.__post_init__` already checks this at construction, but
            # `__setattr__` does not re-check on replacement (unlike `MeteoInputs`), so a
            # raster swapped in afterwards can still disagree with the grid.
            if flow_network.flow_dir_arr.shape != (flow_network.rows, flow_network.cols):
                raise ValueError(GRID_MISMATCH_ERROR)
            if flow_network.FDT is None:
                raise ValueError(
                    "cell-to-cell routing needs the flow-direction table; the flow network "
                    "was built without one"
                )

        geometry = None
        if with_river_geometry or skip_hydraulic_cells:
            geometry = _require(model, "river_geometry", "call read_river_geometry first")

        return cls(
            period=model.period,
            meteo=_require(
                model, "meteo", "assign MeteoInputs.from_rasters(...) to model.meteo"
            ),
            flow_network=flow_network,
            parameters=_require(model, "parameters", "call read_parameters first"),
            model_setup=_require(model, "model_setup", "call read_lumped_model first"),
            river_geometry=geometry,
            skip_hydraulic_cells=skip_hydraulic_cells,
            flow_path_length=getattr(model, "flow_path_length_arr", None),
        )


@dataclass(frozen=True)
class LumpedRun:
    """Everything a lumped run needs, checked and non-optional.

    A lumped catchment has no grid, so it carries one column per driver rather than three cubes,
    and no flow network at all.

    Attributes:
        period: The span the run covers.
        data: `(time, 4)` array of precipitation, ET, temperature and the long-term average.
        parameters: The parameter vector plus the `(snow, maxbas)` pair fixing its width.
        model_setup: The conceptual model instance and the state it starts from.
    """

    period: SimulationPeriod
    data: np.ndarray
    parameters: ParameterSet
    model_setup: ConceptualModelSetup

    def __post_init__(self):
        """Check the driver record covers the period the model was built for.

        Raises:
            ValueError: The record is not four columns wide, or does not span the period.
        """
        if np.ndim(self.data) != 2 or np.shape(self.data)[1] != 4:
            raise ValueError(
                "the lumped drivers must be a (time, 4) array of precipitation, ET, "
                f"temperature and the long-term average, got shape {np.shape(self.data)}"
            )
        steps = np.shape(self.data)[0]
        if steps != len(self.period):
            raise ValueError(
                f"the lumped drivers hold {steps} steps but the model spans "
                f"{len(self.period)} ({self.period.start:%Y-%m-%d} to "
                f"{self.period.end:%Y-%m-%d}); the run is positional, so a mismatch silently "
                "pairs each step with the wrong date"
            )

    @classmethod
    def from_model(cls, model: CatchmentLike) -> LumpedRun:
        """Narrow a built catchment into a validated lumped run.

        Args:
            model: A catchment with its inputs read.

        Returns:
            LumpedRun: The validated inputs.

        Raises:
            ValueError: A required input is unset, or the record does not span the period.
        """
        return cls(
            period=model.period,
            data=_require(model, "data", "call read_lumped_inputs first"),
            parameters=_require(model, "parameters", "call read_parameters first"),
            model_setup=_require(model, "model_setup", "call read_lumped_model first"),
        )
