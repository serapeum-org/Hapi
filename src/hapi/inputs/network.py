"""The routing network: flow accumulation, flow direction, and the grid they define."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from loguru import logger
from pyramids.dataset import Dataset

from hapi.inputs.dem import DEM
from hapi.inputs.rasters import _warn_if_no_sentinel


def _to_int_codes(array: np.ndarray) -> np.typing.NDArray:
    """Return the finite cells of `array` truncated to 64-bit integers.

    Shared by the flow-accumulation and flow-direction reads, which both need the
    distinct *integer* values of a masked raster.

    Truncation happens before the caller de-duplicates: collapsing to integers first is
    what makes 1.2 and 1.8 a single value, matching the per-cell `set(int(...))` this
    replaced. De-duplicating first would leave both and yield a repeated `1`.

    Args:
        array: A 2-D array whose masked cells are `NaN`.

    Returns:
        np.ndarray: 1-D `int64` array of the finite cells, unsorted and not
            de-duplicated.

    Raises:
        ValueError: A cell is infinite, or is too large for `int64`. `astype` would
            otherwise saturate silently to `INT64_MIN`/`INT64_MAX` with only a
            `RuntimeWarning`.
    """
    finite = array[~np.isnan(array)]
    if not np.isfinite(finite).all():
        raise ValueError(
            "raster contains infinite values, which cannot be converted to integer "
            "cell codes; check the source raster's no-data handling."
        )
    info = np.iinfo(np.int64)
    if finite.size and (finite.min() < info.min or finite.max() > info.max):
        raise ValueError(
            f"raster values fall outside the int64 range [{info.min}, {info.max}]; "
            "converting them would silently saturate."
        )
    return finite.astype(np.int64)


#: The eight-directional ESRI codes a flow-direction raster may hold.
D8_CODES = frozenset({1, 2, 4, 8, 16, 32, 64, 128})


@dataclass(eq=False)
class FlowNetwork:
    """The catchment's routing network and the grid it defines.

    Built from the flow-accumulation and flow-direction rasters, which together fix both
    *where* the catchment is -- its grid, its domain cells, its outlet -- and *how* water
    moves through it. The two were separate readers on
    :class:`~hapi.model.catchment.Catchment`; holding them together keeps the grid with the
    array it is measured from.

    Only what the rasters carry is stored. Everything the flow-accumulation reader computed
    is derived here instead, so the grid can never disagree with the accumulation array
    it came from.

    Cells outside the domain are `NaN` in both arrays: masking is delegated to pyramids
    via `read_array(masked=True)`, which compares integer bands to the sentinel exactly
    and honours a band's GDAL mask.

    Attributes:
        flow_acc_arr: `(rows, cols)` flow accumulation, `NaN` outside the domain.
        flow_dir_arr: `(rows, cols)` D8 flow direction, `NaN` outside the domain.
        FDT: Flow-direction table -- `"row,col"` mapped to the cells draining into it.
        no_data_value: The accumulation raster's sentinel, as declared on the band.
        cell_size: Pixel width in map units.
        px_area: Pixel area in km2 -- width times height, so a non-square grid is not
            silently squared off. Assumes a metric CRS; a geographic one gives a
            meaningless area.

    Examples:
        >>> from hapi.inputs import FlowNetwork
        >>> network = FlowNetwork.from_rasters(  # doctest: +SKIP
        ...     "gis/acc4000.tif", "gis/fd4000.tif"
        ... )
        >>> network.rows, network.cols  # doctest: +SKIP
        (13, 14)

        - Or straight from arrays, which is what the properties below derive from:

          >>> import numpy as np
          >>> from hapi.inputs import FlowNetwork
          >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
          >>> network = FlowNetwork(
          ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
          ... )
          >>> network.shape, network.no_elem
          ((2, 2), 3)

    Warning:
        `FDT` is **not** derived from the masked `flow_dir_arr`. It comes from
        :meth:`hapi.inputs.dem.DEM.flow_direction_table`, which reads the raster a second time and
        applies its own `np.isclose(rtol=1e-5)` comparison, ignoring the band's GDAL mask.
        The two therefore disagree on any cell whose masking depends on the mask band or on
        the exact-vs-tolerant comparison: such a cell can be `NaN` in `flow_dir_arr` and
        still appear as a key in `FDT`. The masks already differed before masking was
        delegated to pyramids (`rel_tol=0.001` against `rtol=1e-5`); delegating widened the
        gap rather than creating it. Reconciling them means changing :mod:`hapi.inputs.dem`, which
        is slated to move to `digital-rivers`, so it is left as it is and documented here.

    """

    flow_acc_arr: np.ndarray
    no_data_value: float | int | None
    cell_size: float
    px_area: float
    flow_dir_arr: np.ndarray | None = None
    FDT: dict | None = None

    def __post_init__(self):
        """Check the two rasters describe the same grid.

        Raises:
            ValueError: The accumulation and direction arrays are not the same shape, so
                a cell index would mean a different place in each.
        """
        if (
            self.flow_dir_arr is not None
            and self.flow_acc_arr.shape != self.flow_dir_arr.shape
        ):
            raise ValueError(
                f"the flow accumulation raster is {self.flow_acc_arr.shape} but the flow "
                f"direction raster is {self.flow_dir_arr.shape}; both must share the "
                "catchment's grid"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """tuple[int, int]: The `(rows, cols)` grid both rasters share."""
        return self.flow_acc_arr.shape

    @property
    def rows(self) -> int:
        """int: Number of grid rows."""
        return self.shape[0]

    @property
    def cols(self) -> int:
        """int: Number of grid columns."""
        return self.shape[1]

    @property
    def no_elem(self) -> int:
        """int: Number of cells inside the domain, i.e. not masked.

        Sizes the parameter vectors a calibration produces, so it is derived from the
        masked array rather than recounted from the raster.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import FlowNetwork
            >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
            >>> network = FlowNetwork(
            ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
            ... )
            >>> network.no_elem
            3

        """
        return int(np.count_nonzero(~np.isnan(self.flow_acc_arr)))

    def __setattr__(self, name: str, value: object) -> None:
        """Drop the caches derived from the accumulation array when it is replaced.

        Args:
            name: Attribute being set.
            value: New value.
        """
        if name in ("flow_acc_arr", "flow_dir_arr"):
            self._check_replacement(name, value)
        if name == "flow_acc_arr":
            self.__dict__.pop("acc_val", None)
            self.__dict__.pop("cells_by_acc_val", None)
        object.__setattr__(self, name, value)

    def _check_replacement(self, name: str, value: object) -> None:
        """Reject a raster that would leave the two describing different grids.

        `__post_init__` alone cannot hold the class's promise that the two rasters share a
        grid: the fields are plain mutable attributes, so replacing one afterwards silently
        broke it, and a cell index then meant a different place in each. `MeteoInputs` has
        guarded its cubes this way since it was written; this is the same guard, which the run
        layer had been compensating for by re-checking the shape itself.

        Args:
            name: Which raster is being replaced.
            value: The replacement.

        Raises:
            ValueError: The replacement is not a 2D array of the shape the other one has.
        """
        other = "flow_dir_arr" if name == "flow_acc_arr" else "flow_acc_arr"
        current = getattr(self, other, None)
        if current is None or value is None:
            return
        expected = np.shape(current)
        if not isinstance(value, np.ndarray) or value.shape != expected:
            got = getattr(value, "shape", type(value).__name__)
            raise ValueError(
                f"{name} must stay {expected} to match {other}, got {got}; build a new "
                "FlowNetwork to change the catchment's grid"
            )

    @cached_property
    def acc_val(self) -> list[int]:
        """list[int]: The distinct accumulation values inside the domain, ascending.

        Cached: `route_muskingum` reads this once per `(accumulation level, row, column)`, so
        recomputing the `np.unique` on every read costs `(n_acc - 1) x rows x cols` scans of
        the whole grid -- unnoticeable on the 13x14 test catchment and hours on a real one.
        Replacing `flow_acc_arr` clears the cache.

        The maximum is expected to equal :attr:`no_elem`, or one less depending on whether
        the outlet is counted; :meth:`from_rasters` logs a mismatch at DEBUG rather than
        raising, since some upstream tools number cells from one.

        Examples:
            - Values are truncated before de-duplication, so 1.2 and 1.8 are one code:

              >>> import numpy as np
              >>> from hapi.inputs import FlowNetwork
              >>> acc = np.array([[1.2, 1.8], [3.0, np.nan]])
              >>> FlowNetwork(
              ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
              ... ).acc_val
              [1, 3]

        """
        values: list[int] = np.unique(_to_int_codes(self.flow_acc_arr)).tolist()
        return values

    @cached_property
    def cells_by_acc_val(self) -> dict[int, list[tuple[int, int]]]:
        """dict: In-domain cell indices grouped by their accumulation code, row-major.

        The routing has to visit cells upstream-first, and accumulation gives that order. Asking
        "which cells are at level j" used to be answered by walking the whole grid and testing
        every cell against j -- once per level. Since the number of distinct levels grows with
        the domain, that made the routing pass O(n_acc x rows x cols), effectively quadratic, to
        visit each cell once. On the 13x14 Coello grid it was 4,004 visits for 89 cells; at
        250x250 it projects to about 1.9 billion.

        Building the answer once makes the same pass O(no_elem). Cached for the same reason
        :attr:`acc_val` is, and dropped with it when `flow_acc_arr` is replaced.

        Keys are **truncated** to integers, the same codes :attr:`acc_val` holds, so every
        in-domain cell is reachable through one of them. That matters for a fractional raster:
        the grid scan this replaced compared the raw value against a truncated code, so `1.2`
        never matched the code `1` and the cell was never routed at all -- and because the
        routing sums `quz_routed` from upstream neighbours, that cell's whole contribution
        vanished from every cell below it, silently. Truncating both sides is what
        :attr:`acc_val` has always documented ("1.2 and 1.8 are one code"); only the comparison
        had drifted.

        Order within a level is row-major, matching the `x`-outer/`y`-inner scan it replaces, so
        the routing visits cells in exactly the order it did.

        Examples:
            - Integral accumulation, which is what a cell-count raster holds:

              >>> import numpy as np
              >>> from hapi.inputs import FlowNetwork
              >>> acc = np.array([[0.0, 1.0], [1.0, np.nan]])
              >>> network = FlowNetwork(
              ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
              ... )
              >>> network.cells_by_acc_val[0]
              [(0, 0)]
              >>> network.cells_by_acc_val[1]
              [(0, 1), (1, 0)]

            - Fractional values share the code they truncate to, so they route together
              rather than being dropped:

              >>> import numpy as np
              >>> from hapi.inputs import FlowNetwork
              >>> acc = np.array([[1.2, 1.8], [3.0, np.nan]])
              >>> network = FlowNetwork(
              ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
              ... )
              >>> network.acc_val
              [1, 3]
              >>> network.cells_by_acc_val[1]
              [(0, 0), (0, 1)]

        """
        grouped: dict[int, list[tuple[int, int]]] = {}
        rows, cols = np.nonzero(~np.isnan(self.flow_acc_arr))
        for x, y in zip(rows, cols):
            # Truncated, matching `acc_val`: see the note above on the comparison that drifted.
            key = int(self.flow_acc_arr[x, y])
            grouped.setdefault(key, []).append((int(x), int(y)))
        return grouped

    @property
    def outlet(self) -> tuple:
        """tuple: Index of the most-accumulated cell, as `np.where` returns it."""
        return np.nonzero(self.flow_acc_arr == np.nanmax(self.flow_acc_arr))

    @property
    def px_tot_area(self) -> float:
        """float: Total domain area in km2 -- :attr:`no_elem` times :attr:`px_area`."""
        return self.no_elem * self.px_area

    def matches(self, rows: int, cols: int) -> bool:
        """Report whether the network covers a given grid.

        Args:
            rows: Number of rows to compare against.
            cols: Number of columns.

        Returns:
            bool: True when the network's grid is exactly `(rows, cols)`.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import FlowNetwork
            >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
            >>> network = FlowNetwork(
            ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
            ... )
            >>> network.matches(2, 2), network.matches(3, 3)
            (True, False)

        """
        return self.shape == (rows, cols)

    @property
    def has_flow_direction(self) -> bool:
        """bool: Whether a flow-direction raster was loaded.

        The Muskingum path routes cell to cell and needs one; the triangular (MAXBAS) path
        sends every cell straight to the outlet and does not.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import FlowNetwork
            >>> acc = np.array([[0.0, 1.0], [2.0, np.nan]])
            >>> network = FlowNetwork(
            ...     acc, no_data_value=-9999.0, cell_size=4000.0, px_area=16.0
            ... )
            >>> network.has_flow_direction
            False

        """
        return self.flow_dir_arr is not None

    @classmethod
    def from_rasters(
        cls, flow_acc: str | Path, flow_dir: str | Path | None = None
    ) -> FlowNetwork:
        """Read the routing network from the accumulation and direction rasters.

        Args:
            flow_acc: Path to the flow-accumulation raster. Any format GDAL can open.
            flow_dir: Path to the flow-direction raster, in the eight-directional ESRI
                encoding. Optional: the Muskingum routing needs it, but the triangular
                (MAXBAS) path sends every cell straight to the outlet and never reads it.

        Returns:
            FlowNetwork: The two masked arrays, the direction table, and the cell geometry
                read off the accumulation raster's transform.

        Raises:
            FileNotFoundError: Either path does not exist.
            ValueError: The direction raster holds a code outside
                :data:`D8_CODES`, the two rasters disagree on the grid, or every
                accumulation cell is no-data.

        Warns:
            UserWarning: A raster declares no no-data value, so every cell is treated as
                inside the catchment.
        """
        # `with`: on Windows an open GDAL handle keeps a lock on the file, so a script
        # that reads a catchment and then moves or rewrites its inputs fails, and a loop
        # over basins accumulates handles. Everything read from the dataset is copied into
        # arrays here, so nothing needs the handle afterwards.
        with Dataset.read_file(flow_acc) as acc:
            _warn_if_no_sentinel(acc, "flow accumulation")
            acc_arr = np.ma.filled(
                acc.read_array(band=0, masked=True).astype(float), np.nan
            )
            transform = acc.transform

        dir_arr, table = None, None
        if flow_dir is not None:
            with DEM.read_file(flow_dir) as direction:
                _warn_if_no_sentinel(direction, "flow direction")
                dir_arr = np.ma.filled(
                    direction.read_array(band=0, masked=True).astype(float), np.nan
                )
                codes = set(np.unique(_to_int_codes(dir_arr)).tolist())
                if not codes <= set(D8_CODES):
                    raise ValueError(
                        "flow direction raster should contain values 1,2,4,8,16,32,64,128 "
                        f"only, found {sorted(codes - set(D8_CODES))}"
                    )
                table = direction.flow_direction_table()
        network = cls(
            flow_acc_arr=acc_arr,
            flow_dir_arr=dir_arr,
            FDT=table,
            no_data_value=acc.no_data_value[0],
            cell_size=abs(acc.cell_size),
            px_area=(abs(transform.pixel_width) / 1000.0)
            * (abs(transform.pixel_height) / 1000.0),
        )

        if not network.acc_val:
            raise ValueError(
                f"every cell of {flow_acc} is no-data, so the catchment has no domain: "
                "check the raster's no-data value and its mask band"
            )
        acc_val_mx = max(network.acc_val)
        if acc_val_mx not in (network.no_elem, network.no_elem - 1):
            logger.debug(
                "flow accumulation raster values are not correct max value should equal "
                "number of cells or number of cells -1 Max Value in the Flow Acc raster "
                f"is {acc_val_mx} while No of cells are {network.no_elem}"
            )
        logger.debug("Flow network is read successfully")
        return network
