"""The three meteorological drivers, as aligned `(rows, cols, time)` cubes.

Read from folders of dated rasters, from one NetCDF per variable, or from a single NetCDF
holding all three.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pyramids.netcdf import NetCDF

from hapi.core.config import (
    NETCDF_PATH_MESSAGE,
    MeteoConfig,
    missing_drivers_message,
)
from hapi.inputs.rasters import _as_datetime, read_rasters

#: The meteorological drivers the conceptual model consumes, in the order the readers report them.
METEO_VARIABLES = ("precipitation", "temperature", "evapotranspiration")


def _cube_from_netcdf(nc: NetCDF, variable: str) -> np.ndarray:
    """Read one NetCDF variable as a `(rows, cols, time)` cube.

    Args:
        nc: An open :class:`~pyramids.netcdf.NetCDF`.
        variable: Name of the variable to read.

    Returns:
        np.ndarray: The variable with time moved to the last axis, matching the layout the
            raster readers produce.

    Raises:
        KeyError: `variable` is not in the file.
    """
    if variable not in nc.variable_names:
        raise KeyError(
            f"variable {variable!r} is not in the NetCDF. Available: {nc.variable_names}."
        )
    values = np.asarray(nc.get_variable(variable).read_array())
    # NetCDF stores (time, y, x); the model indexes cells then time.
    return np.moveaxis(values, 0, -1)


@dataclass(eq=False)
class MeteoInputs:
    r"""The three meteorological drivers of the rainfall-runoff model, held as aligned cubes.

    Each field is a `(rows, cols, time)` array — cell first, time last — which is the layout
    :class:`~hapi.model.catchment.Catchment` and the conceptual models index. The three cubes must
    agree on all three axes; that is checked on construction, because a silent mismatch surfaces
    much later as a confusing index error inside the run loop.

    No-data cells are carried through **as stored**, not converted to NaN. The distributed model
    takes its domain from the flow-accumulation raster rather than from the meteorological
    no-data mask, so masking here would change what the run sees.

    Build one with whichever classmethod matches how the data is stored:

    * :meth:`from_rasters` -- three folders of date-stamped rasters (the historical layout).
    * :meth:`from_netcdf_files` -- one NetCDF per variable.
    * :meth:`from_netcdf` -- a single NetCDF holding all three as separate variables.

    Attributes:
        precipitation: `(rows, cols, time)` rainfall cube.
        temperature: `(rows, cols, time)` temperature cube.
        evapotranspiration: `(rows, cols, time)` potential-evapotranspiration cube.
        time: Optional calendar axis, one entry per timestep. Carried for reference and for
            cross-checking against the model's own date index; the run itself is positional.

    Examples:
        - From three folders of rasters:
            ```python
            >>> from hapi.inputs import MeteoInputs
            >>> data = MeteoInputs.from_rasters(  # doctest: +SKIP
            ...     "data/prec", "data/temp", "data/evap", file_name_data_fmt="%Y.%m.%d"
            ... )

            ```
        - From one NetCDF per variable:
            ```python
            >>> data = MeteoInputs.from_netcdf_files(  # doctest: +SKIP
            ...     "data/prec.nc", "data/temp.nc", "data/evap.nc"
            ... )

            ```
    """

    precipitation: np.ndarray
    temperature: np.ndarray
    evapotranspiration: np.ndarray
    time: pd.DatetimeIndex | None = field(default=None)
    #: Cache behind the :attr:`ll_temp` property; not part of the constructor signature.
    _ll_temp: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Check the three cubes are 3D and share a shape.

        Raises:
            ValueError: A cube is not 3-dimensional, the three shapes disagree, or `time` does
                not have one entry per timestep.
        """
        for name in METEO_VARIABLES:
            cube = getattr(self, name)
            if not isinstance(cube, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy array, got {type(cube).__name__}"
                )
            if cube.ndim != 3:
                raise ValueError(
                    f"{name} must be a 3D (rows, cols, time) array, got shape {cube.shape}"
                )

        shapes = {name: getattr(self, name).shape for name in METEO_VARIABLES}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"the three cubes must share one shape, got {shapes}")

        if self.time is not None and len(self.time) != self.time_steps:
            raise ValueError(
                f"time has {len(self.time)} entries but the cubes hold {self.time_steps} steps"
            )

    def __setattr__(self, name: str, value: object) -> None:
        """Set an attribute, keeping the three cubes in agreement.

        The class promises the cubes share a shape, and `__post_init__` alone cannot hold
        that promise: the fields are plain mutable attributes, so replacing one afterwards
        silently breaks it. `shape`, `rows` and `time_steps` all report precipitation's, so
        a replacement of the wrong size passes `validate_against` and the run then indexes
        past the end of whichever cube is short -- or, worse, reads the right index of the
        wrong grid. Re-check on assignment instead.

        Replacing `temperature` also drops the cached `ll_temp`, which is derived from it.

        Args:
            name: Attribute being set.
            value: New value.

        Raises:
            ValueError: `value` is a cube that does not match the other two.
        """
        if name in METEO_VARIABLES and getattr(self, name, None) is not None:
            self._check_replacement(name, value)
        if name == "temperature" and getattr(self, "_ll_temp", None) is not None:
            object.__setattr__(self, "_ll_temp", None)
        object.__setattr__(self, name, value)

    def _check_replacement(self, name: str, value: object) -> None:
        """Reject a cube that would leave the three disagreeing.

        Args:
            name: Which cube is being replaced.
            value: The replacement.

        Raises:
            ValueError: The replacement is not a 3D array of the shape the others share.
        """
        others = [
            getattr(self, other)
            for other in METEO_VARIABLES
            if other != name and getattr(self, other, None) is not None
        ]
        if not others:
            return
        expected = others[0].shape
        if not isinstance(value, np.ndarray) or value.shape != expected:
            got = getattr(value, "shape", type(value).__name__)
            raise ValueError(
                f"{name} must stay {expected} to match the other cubes, got {got}; "
                "build a new MeteoInputs to change the grid or the period"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """tuple[int, int, int]: The shared `(rows, cols, time)` shape."""
        return self.precipitation.shape

    @property
    def rows(self) -> int:
        """int: Number of grid rows."""
        return self.shape[0]

    @property
    def cols(self) -> int:
        """int: Number of grid columns."""
        return self.shape[1]

    @property
    def time_steps(self) -> int:
        """int: Number of timesteps the drivers cover."""
        return self.shape[2]

    @property
    def simulation_steps(self) -> int:
        """int: `time_steps` plus one, the length the run's state arrays need.

        The conceptual model carries an initial state before the first driver step, so the
        per-cell result arrays hold one slot more than there is data.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import MeteoInputs
            >>> cube = np.zeros((2, 3, 4), dtype="float32")
            >>> data = MeteoInputs(cube, cube, cube)
            >>> data.time_steps, data.simulation_steps
            (4, 5)

        """
        return self.time_steps + 1

    @property
    def ll_temp(self) -> np.ndarray:
        """np.ndarray: Long-term average temperature, `(rows, cols, time)`.

        Each cell's mean over the whole record, broadcast back across the time axis -- the
        reference the snow routine compares each step against. Derived on first use and cached,
        since the run reads it per cell and it never changes once the cubes are set.

        Assign to this to override the derived value; the replacement must match
        :attr:`shape`.

        Examples:
            - Each cell's own mean, repeated across time:

                >>> import numpy as np
                >>> from hapi.inputs import MeteoInputs
                >>> temp = np.arange(8, dtype="float32").reshape(1, 2, 4)
                >>> data = MeteoInputs(temp, temp, temp)
                >>> data.ll_temp[0, 0, :]
                array([1.5, 1.5, 1.5, 1.5], dtype=float32)
                >>> data.ll_temp[0, 1, :]
                array([5.5, 5.5, 5.5, 5.5], dtype=float32)

        """
        if self._ll_temp is None:
            avg = self.temperature.mean(axis=2)
            self._ll_temp = np.repeat(
                avg[:, :, np.newaxis], self.time_steps, axis=2
            ).astype(np.float32)
        return self._ll_temp

    @ll_temp.setter
    def ll_temp(self, value: np.ndarray) -> None:
        """Override the derived long-term average temperature.

        Args:
            value: A `(rows, cols, time)` array matching :attr:`shape`.

        Raises:
            ValueError: `value` does not match the cubes' shape.
        """
        value = np.asarray(value)
        if value.shape != self.shape:
            raise ValueError(
                f"ll_temp must match the cubes {self.shape}, got {value.shape}"
            )
        self._ll_temp = value

    def validate_against(
        self, rows: int, cols: int, date_index: pd.DatetimeIndex | None = None
    ) -> None:
        """Check the cubes cover the model's grid, and optionally its calendar.

        The three cubes already agree with each other -- that is settled at construction. This
        is the other half: that they agree with the grid the GIS inputs defined, and with the
        period the model was built for.

        Args:
            rows: Number of grid rows the model expects.
            cols: Number of grid columns.
            date_index: The model's own dates. When given, the drivers must supply one step
                per date. `None` skips the check, which is what a caller with no calendar of
                its own does.

        Raises:
            ValueError: The cubes do not cover that grid, or they do not span `date_index`.

        Examples:
            >>> import numpy as np
            >>> from hapi.inputs import MeteoInputs
            >>> cube = np.zeros((2, 3, 4), dtype="float32")
            >>> data = MeteoInputs(cube, cube, cube)
            >>> data.validate_against(2, 3)
            >>> data.validate_against(5, 5)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: the meteorological inputs are 2x3 but the model grid is 5x5...

        """
        if (self.rows, self.cols) != (rows, cols):
            raise ValueError(
                f"the meteorological inputs are {self.rows}x{self.cols} but the model grid is "
                f"{rows}x{cols}; every input must share the catchment's grid"
            )

        if date_index is None:
            return

        if self.time_steps != len(date_index):
            raise ValueError(
                f"the meteorological inputs hold {self.time_steps} steps but the model spans "
                f"{len(date_index)} ({date_index[0]:%Y-%m-%d} to {date_index[-1]:%Y-%m-%d}); "
                "the run is positional, so a mismatch silently pairs each step with the wrong "
                "date"
            )
        if self.time is not None and (
            self.time[0] != date_index[0] or self.time[-1] != date_index[-1]
        ):
            raise ValueError(
                f"the meteorological inputs cover {self.time[0]:%Y-%m-%d} to "
                f"{self.time[-1]:%Y-%m-%d} but the model spans {date_index[0]:%Y-%m-%d} to "
                f"{date_index[-1]:%Y-%m-%d}"
            )

    @classmethod
    def from_rasters(
        cls,
        precipitation: str | Path,
        temperature: str | Path,
        evapotranspiration: str | Path,
        *,
        per_variable: dict[str, dict[str, Any]] | None = None,
        glob: str = "*.tif",
        regex_string: str = r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        start: str | int | dt.datetime | None = None,
        end: str | int | dt.datetime | None = None,
        fmt: str = "%Y-%m-%d",
        gdal_env: dict[str, str] | None = None,
    ) -> MeteoInputs:
        r"""Read the three drivers from folders of date-stamped rasters.

        Args:
            precipitation: Folder of rainfall rasters.
            temperature: Folder of temperature rasters.
            evapotranspiration: Folder of evapotranspiration rasters.
            per_variable: Per-folder overrides, keyed by driver name, applied over the
                shared arguments for that folder only. Needed when the three folders come
                from different sources, which the documented download workflow produces:
                CHIRPS names its rainfall `..._2009.01.01.tif` while ERA5 names its
                temperature `..._20090101.tif`, and no single `regex_string` finds the date
                in both. Its inner keys are the reader arguments below.
            glob: :mod:`fnmatch` pattern selecting the rasters. Defaults to `"*.tif"`.
            regex_string: Where the date sits in each file name.
            date: Whether the matched value is a date. `False` orders by a plain index.
            file_name_data_fmt: `strptime` format of the date. Inferred from the names when
                omitted; pass it for a layout the digits cannot settle, such as a day-first
                `03.02.1990`.
            start: Inclusive lower bound, to read a window rather than the whole folder.
            end: Inclusive upper bound; see `start`.
            fmt: `strptime` format of `start` / `end`.
            gdal_env: GDAL configuration applied for the reads, e.g.
                `{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}` to skip the per-open
                directory listing on network storage.

        Returns:
            MeteoInputs: The three cubes plus a calendar -- the rainfall folder's when it carries
                one, otherwise the first source that does, and None when none do.

        Raises:
            FileNotFoundError: A folder does not exist or holds no matching raster.
            KeyError: `per_variable` names something that is not one of the three drivers.
            ValueError: The three folders do not yield the same shape.

        Examples:
            Three folders from one source share every argument:

            >>> MeteoInputs.from_rasters(  # doctest: +SKIP
            ...     prec_dir, temp_dir, evap_dir, start="2009-01-01", end="2009-12-31"
            ... )

            CHIRPS rainfall alongside ERA5 temperature and evapotranspiration:

            >>> MeteoInputs.from_rasters(  # doctest: +SKIP
            ...     chirps_dir,
            ...     era5_temp_dir,
            ...     era5_evap_dir,
            ...     per_variable={
            ...         "temperature": {"regex_string": r"\d{8}"},
            ...         "evapotranspiration": {"regex_string": r"\d{8}"},
            ...     },
            ... )
        """
        overrides = per_variable or {}
        unknown = set(overrides) - set(METEO_VARIABLES)
        if unknown:
            raise KeyError(
                f"per_variable names {sorted(unknown)}, which are not drivers; "
                f"expected any of {list(METEO_VARIABLES)}"
            )

        shared = dict(
            glob=glob,
            regex_string=regex_string,
            date=date,
            file_name_data_fmt=file_name_data_fmt,
            start=start,
            end=end,
            fmt=fmt,
            gdal_env=gdal_env,
        )
        unknown_keys = {k for o in overrides.values() for k in o} - set(shared)
        if unknown_keys:
            raise TypeError(
                f"per_variable holds {sorted(unknown_keys)}, which are not reader "
                f"arguments; expected any of {sorted(shared)}"
            )

        cubes, calendar = {}, None
        for name, path in zip(
            METEO_VARIABLES, (precipitation, temperature, evapotranspiration)
        ):
            collection = read_rasters(path, **{**shared, **overrides.get(name, {})})
            cubes[name] = np.moveaxis(np.asarray(collection.values), 0, -1)
            if calendar is None and collection.time is not None:
                calendar = pd.DatetimeIndex(list(collection.time))
        return cls(**cubes, time=calendar)

    @classmethod
    def from_netcdf_files(
        cls,
        precipitation: str | Path,
        temperature: str | Path,
        evapotranspiration: str | Path,
        variable: str | None = None,
        start: str | dt.datetime | None = None,
        end: str | dt.datetime | None = None,
        fmt: str = "%Y-%m-%d",
    ) -> MeteoInputs:
        """Read the three drivers from one NetCDF per variable.

        Args:
            precipitation: NetCDF holding the rainfall cube.
            temperature: NetCDF holding the temperature cube.
            evapotranspiration: NetCDF holding the evapotranspiration cube.
            variable: Name of the variable to take from each file. `None` (default) takes each
                file's only variable, which is what `DatasetCollection.to_netcdf` writes for a
                single-band collection.
            start: Inclusive lower bound on the period to keep. `None` reads the whole file.
            end: Inclusive upper bound; see `start`.
            fmt: `strptime` format of `start` / `end` when they are strings.

        Returns:
            MeteoInputs: The three cubes plus a calendar -- the rainfall file's when it carries
                one, otherwise the first source that does, and None when none do.

        Raises:
            KeyError: `variable` is not in one of the files.
            ValueError: A file holds several variables and `variable` was not given, or the
                three files do not yield the same shape.
        """
        cubes, calendar = {}, None
        for name, path in zip(
            METEO_VARIABLES, (precipitation, temperature, evapotranspiration)
        ):
            nc = NetCDF.read_file(path)
            if variable is None:
                if len(nc.variable_names) != 1:
                    raise ValueError(
                        f"{path} holds {len(nc.variable_names)} variables "
                        f"({nc.variable_names}); pass variable= to pick one, or use "
                        "from_netcdf() for a single file holding all three drivers."
                    )
                target = nc.variable_names[0]
            else:
                target = variable
            cubes[name] = _cube_from_netcdf(nc, target)
            if calendar is None:
                calendar = cls._calendar(nc)
        cubes, calendar = cls._window(cubes, calendar, start, end, fmt)
        return cls(**cubes, time=calendar)

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        precipitation: str,
        temperature: str,
        evapotranspiration: str,
        start: str | dt.datetime | None = None,
        end: str | dt.datetime | None = None,
        fmt: str = "%Y-%m-%d",
    ) -> MeteoInputs:
        """Read all three drivers from one NetCDF holding them as separate variables.

        Args:
            path: The NetCDF file.
            precipitation: Name of the rainfall variable inside it.
            temperature: Name of the temperature variable.
            evapotranspiration: Name of the evapotranspiration variable.
            start: Inclusive lower bound on the period to keep. `None` reads the whole file.
                A packed record often spans decades while a run covers one year, and the
                drivers pair with the model's dates by position, so the window is what makes
                a long file usable for a short run.
            end: Inclusive upper bound; see `start`.
            fmt: `strptime` format of `start` / `end` when they are strings.

        Returns:
            MeteoInputs: The three cubes plus the file's calendar, trimmed to the window.

        Raises:
            KeyError: One of the named variables is not in the file.
            ValueError: The three variables do not share a shape.
        """
        nc = NetCDF.read_file(path)
        cubes = {
            name: _cube_from_netcdf(nc, var)
            for name, var in zip(
                METEO_VARIABLES, (precipitation, temperature, evapotranspiration)
            )
        }
        cubes, calendar = cls._window(cubes, cls._calendar(nc), start, end, fmt)
        return cls(**cubes, time=calendar)

    @classmethod
    def from_config(
        cls,
        config: MeteoConfig,
        start: str | None = None,
        end: str | None = None,
        fmt: str = "%Y-%m-%d",
    ) -> MeteoInputs:
        """Build the drivers with whichever loader the configuration's `source` names.

        The dispatch behind a `meteo` block of a YAML run configuration: `"rasters"` reads three
        folders, `"netcdf_files"` one file per driver, and `"netcdf"` a single combined file
        whose variables the block names. `hapi.core.config.RunConfig` has already checked that the
        fields the chosen source needs are set, so this calls the loader directly.

        Each bound is parsed with the format it was written in -- `config.fmt` for a bound the
        block states, `fmt` for one inherited from the caller -- and handed on as a `datetime`.
        The two formats are independent fields, so parsing an inherited bound with the block's
        format would either fail loudly or, between two mutually parseable layouts such as
        `"%d-%m-%Y"` and `"%m-%d-%Y"`, silently window the drivers to the wrong period.

        Args:
            config: The `meteo` block of a distributed configuration.
            start: Window start used when `config.start` is unset, so the drivers can default to
                the period the model spans. `None` leaves the lower bound open.
            end: Window end used when `config.end` is unset. `None` leaves it open.
            fmt: `strptime` format of `start` / `end`, the inherited bounds. Bounds stated by
                the block are parsed with `config.fmt` instead.

        Returns:
            MeteoInputs: The three cubes plus the calendar, windowed to the requested period.

        Raises:
            ValueError: A field the chosen source needs is unset -- one of the three drivers, or
                `path` for `source="netcdf"`. `RunConfig` rejects such a configuration, so this
                only fires for a `MeteoConfig` built by hand.

        Examples:
            The paths below are fixtures in the Hapi repository, so these run from a checkout
            rather than an installed wheel; substitute your own file to try them elsewhere.

            - Load a combined NetCDF by naming the variable each driver sits in:
                ```python
                >>> from hapi.core.config import MeteoConfig
                >>> from hapi.inputs import MeteoInputs
                >>> meteo = MeteoInputs.from_config(
                ...     MeteoConfig(
                ...         source="netcdf",
                ...         path="tests/rrm/data/coello/meteo.nc",
                ...         precipitation="precipitation",
                ...         temperature="temperature",
                ...         evapotranspiration="evapotranspiration",
                ...     )
                ... )
                >>> meteo.shape
                (13, 14, 10)
                >>> meteo.time[0].strftime("%Y-%m-%d")
                '2009-01-01'

                ```
            - Narrow the same file to part of its record with the fallback window:
                ```python
                >>> from hapi.core.config import MeteoConfig
                >>> from hapi.inputs import MeteoInputs
                >>> meteo = MeteoInputs.from_config(
                ...     MeteoConfig(
                ...         source="netcdf",
                ...         path="tests/rrm/data/coello/meteo.nc",
                ...         precipitation="precipitation",
                ...         temperature="temperature",
                ...         evapotranspiration="evapotranspiration",
                ...     ),
                ...     start="2009-01-03",
                ...     end="2009-01-07",
                ... )
                >>> meteo.time_steps
                5
                >>> meteo.time[-1].strftime("%Y-%m-%d")
                '2009-01-07'

                ```

        See Also:
            from_rasters: The loader `source="rasters"` dispatches to.
            from_netcdf: The loader `source="netcdf"` dispatches to.
            from_netcdf_files: The loader `source="netcdf_files"` dispatches to.
        """
        # Resolve each bound with the format it was written in, then pass datetimes on so the
        # loader's own `fmt` cannot re-parse them.
        window_start = (
            _as_datetime(config.start, config.fmt)
            if config.start is not None
            else _as_datetime(start, fmt)
        )
        window_end = (
            _as_datetime(config.end, config.fmt)
            if config.end is not None
            else _as_datetime(end, fmt)
        )

        # The three are optional on the model because a lumped configuration sets none of them,
        # while every distributed source needs all three. `RunConfig` enforces that, so reaching
        # the raise means a `MeteoConfig` was built by hand -- and it raises the same sentence,
        # phrased once in `hapi.core.config`, because it is the same rule. Bound to locals so the
        # check both reports what is missing and narrows the type for the calls below.
        precipitation = config.precipitation
        temperature = config.temperature
        evapotranspiration = config.evapotranspiration
        if precipitation is None or temperature is None or evapotranspiration is None:
            missing = [
                name for name in METEO_VARIABLES if getattr(config, name) is None
            ]
            raise ValueError(missing_drivers_message(missing))

        if config.source == "rasters":
            extra: dict[str, Any] = {}
            if config.per_variable is not None:
                extra["per_variable"] = config.per_variable
            if config.gdal_env is not None:
                extra["gdal_env"] = config.gdal_env
            return cls.from_rasters(
                precipitation,
                temperature,
                evapotranspiration,
                glob=config.glob,
                regex_string=config.regex_string,
                file_name_data_fmt=config.file_name_data_fmt,
                start=window_start,
                end=window_end,
                fmt=config.fmt,
                **extra,
            )

        if config.source == "netcdf":
            if config.path is None:
                raise ValueError(NETCDF_PATH_MESSAGE)
            return cls.from_netcdf(
                config.path,
                precipitation=precipitation,
                temperature=temperature,
                evapotranspiration=evapotranspiration,
                start=window_start,
                end=window_end,
                fmt=config.fmt,
            )

        return cls.from_netcdf_files(
            precipitation,
            temperature,
            evapotranspiration,
            variable=config.variable,
            start=window_start,
            end=window_end,
            fmt=config.fmt,
        )

    @staticmethod
    def raster_folder_to_netcdf(
        path: str | Path,
        out_path: str | Path,
        *,
        glob: str = "*.tif",
        regex_string: str = r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        start: str | int | dt.datetime | None = None,
        end: str | int | dt.datetime | None = None,
        fmt: str = "%Y-%m-%d",
        gdal_env: dict[str, str] | None = None,
    ) -> Path:
        r"""Pack one driver's folder of dated rasters into a single NetCDF.

        A folder of per-date GeoTIFFs is what the download backends produce, and it is the
        slowest thing the model can be driven from: every run re-opens every file. Packing it
        once into a NetCDF makes later runs read one file, and makes the folder portable --
        the calendar travels inside the file instead of living in the file names.

        The rasters are ordered by :func:`read_rasters`, whose reader arguments are repeated
        here rather than forwarded as `**kwargs`, so a typo is caught at this call rather
        than one frame deeper.

        Args:
            path: Folder holding one variable's rasters.
            out_path: NetCDF file to write. Overwritten if it exists.
            glob: :mod:`fnmatch` pattern selecting the rasters. Defaults to `"*.tif"`.
            regex_string: Where the date sits in each file name.
            date: Whether the matched value is a date. `False` orders by a plain index
                instead, which carries no calendar and so cannot be written here.
            file_name_data_fmt: `strptime` format of the date. Inferred from the names when
                omitted; pass it for a layout the digits cannot settle, such as a day-first
                `03.02.1990`.
            start: Inclusive lower bound, to convert a window rather than the whole folder.
            end: Inclusive upper bound; see `start`.
            fmt: `strptime` format of `start` / `end`.
            gdal_env: GDAL configuration applied for the read, e.g.
                `{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}` to skip the per-open
                directory listing on network storage.

        Returns:
            Path: The file that was written.

        Raises:
            FileNotFoundError: The folder does not exist or matched no raster.
            ValueError: The rasters carry no usable calendar, so the NetCDF would have no
                time axis to write.

        Examples:
            Pack a folder, then drive a model from the result:

            >>> MeteoInputs.raster_folder_to_netcdf(temp_dir, "temp.nc")  # doctest: +SKIP
            >>> MeteoInputs.from_netcdf_files(  # doctest: +SKIP
            ...     "prec.nc", "temp.nc", "evap.nc"
            ... )
        """
        collection = read_rasters(
            path,
            glob=glob,
            regex_string=regex_string,
            date=date,
            file_name_data_fmt=file_name_data_fmt,
            start=start,
            end=end,
            fmt=fmt,
            gdal_env=gdal_env,
        )
        if collection.time is None:
            raise ValueError(
                f"the rasters in {path} carry no calendar, so the NetCDF would have no time "
                "axis; pass regex_string and file_name_data_fmt so the dates can be parsed"
            )

        stamps = list(collection.time)
        if stamps != sorted(stamps):
            raise ValueError(
                f"the rasters in {path} did not come back in chronological order, so the "
                "NetCDF would pair each step with the wrong date"
            )

        out = Path(out_path)
        out.unlink(missing_ok=True)
        collection.to_netcdf(out)
        logger.debug(
            f"{collection.time_length} rasters from {path} written to {out} "
            f"({stamps[0]:%Y-%m-%d} to {stamps[-1]:%Y-%m-%d})"
        )
        return out

    @staticmethod
    def combine_netcdf_files(
        precipitation: str | Path,
        temperature: str | Path,
        evapotranspiration: str | Path,
        out_path: str | Path,
    ) -> Path:
        """Merge one NetCDF per driver into a single file holding all three.

        The counterpart to :meth:`from_netcdf`: three single-variable files go in, one file
        comes out whose variables are named `precipitation`, `temperature` and
        `evapotranspiration`, so a reader can ask for them by name rather than guessing at
        whatever `to_netcdf` called the band.

        The first file seeds the container and the other two are copied in with
        `NetCDF.add_variable`; nothing touches disk until the write, so the sources are left
        as they are.

        Args:
            precipitation: NetCDF holding the rainfall cube.
            temperature: NetCDF holding the temperature cube.
            evapotranspiration: NetCDF holding the evapotranspiration cube.
            out_path: File to write. Overwritten if it exists.

        Returns:
            Path: The file that was written.

        Raises:
            ValueError: One of the sources holds more than one variable, so which cube it
                contributes would be a guess.

        Examples:
            >>> MeteoInputs.combine_netcdf_files(  # doctest: +SKIP
            ...     "prec.nc", "temp.nc", "evap.nc", "meteo.nc"
            ... )
            >>> MeteoInputs.from_netcdf(  # doctest: +SKIP
            ...     "meteo.nc",
            ...     precipitation="precipitation",
            ...     temperature="temperature",
            ...     evapotranspiration="evapotranspiration",
            ... )
        """
        sources = dict(
            zip(METEO_VARIABLES, (precipitation, temperature, evapotranspiration))
        )
        for name, source in sources.items():
            holder = NetCDF.read_file(source)
            if len(holder.variable_names) != 1:
                raise ValueError(
                    f"{source} holds {len(holder.variable_names)} variables "
                    f"({holder.variable_names}); {name} must come from a file with exactly "
                    "one, as raster_folder_to_netcdf writes"
                )

        (seed_name, seed_path), *rest = sources.items()
        combined = NetCDF.read_file(seed_path)
        combined.rename_variable(combined.variable_names[0], seed_name)

        for name, source in rest:
            holder = NetCDF.read_file(source)
            combined.add_variable(holder)
            combined.rename_variable(holder.variable_names[0], name)

        out = Path(out_path)
        out.unlink(missing_ok=True)
        combined.to_file(out)
        logger.debug(f"three drivers combined into {out}")
        return out

    @staticmethod
    def _window(
        cubes: dict[str, np.ndarray],
        calendar: pd.DatetimeIndex | None,
        start: str | dt.datetime | None,
        end: str | dt.datetime | None,
        fmt: str,
    ) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex | None]:
        """Trim the cubes and the calendar to an inclusive date range.

        The raster loader takes `start` / `end` because a folder is read file by file. A
        NetCDF is read whole, so the window is applied after the fact -- but a caller running
        one year out of a forty-year file still needs it, and the drivers pair with the
        model's dates by position, so an untrimmed cube is rejected rather than merely large.

        Args:
            cubes: The three driver cubes, keyed by name.
            calendar: The time axis, or None when the file carries no dates.
            start: Inclusive lower bound, or None for "from the beginning".
            end: Inclusive upper bound, or None for "to the end".
            fmt: `strptime` format for `start` / `end` when they are strings.

        Returns:
            tuple: The trimmed cubes and calendar, unchanged when no bound was given.

        Raises:
            ValueError: A bound was given but the file carries no calendar to apply it to,
                or the range selects no step.
        """
        if start is None and end is None:
            return cubes, calendar
        if calendar is None:
            raise ValueError(
                "start/end need a calendar, and this file carries none; read it whole, or "
                "rebuild it from rasters whose names hold the dates"
            )

        low = _as_datetime(start, fmt) if start is not None else None
        high = _as_datetime(end, fmt) if end is not None else None
        keep = np.ones(len(calendar), dtype=bool)
        if low is not None:
            keep &= calendar >= low
        if high is not None:
            keep &= calendar <= high
        if not keep.any():
            raise ValueError(
                f"no step falls in [{start}, {end}]; the file covers "
                f"{calendar[0]:%Y-%m-%d} to {calendar[-1]:%Y-%m-%d}"
            )
        return {n: c[:, :, keep] for n, c in cubes.items()}, calendar[keep]

    @staticmethod
    def _calendar(nc: NetCDF) -> pd.DatetimeIndex | None:
        """Return a NetCDF's decoded time axis, or None when it carries no calendar.

        `NetCDF.time_stamp` decodes the axis only for a file holding a single data variable; on
        one holding all three drivers it returns None even though the `time` array is there and
        correct. So fall back to the raw values, which `to_netcdf` writes as nanoseconds since
        the epoch.

        Args:
            nc: An open :class:`~pyramids.netcdf.NetCDF`.

        Returns:
            pd.DatetimeIndex | None: The decoded stamps, or None when the file carries no
                calendar -- `to_netcdf` writes a positional index for an undated collection,
                and those values are left alone rather than misread as 1970.
        """
        try:
            stamps = nc.time_stamp
        except (AttributeError, KeyError, ValueError):
            stamps = None
        if stamps:
            return pd.DatetimeIndex(list(stamps))

        try:
            raw = nc.get_time_values()
        except (AttributeError, KeyError, ValueError):
            return None
        if raw is None:
            # No time dimension. Checking explicitly rather than letting `np.asarray(None)`
            # produce a 0-d object array that happens to fail the dtype test below.
            return None
        values = np.asarray(raw)
        # A positional index runs 0..n-1; a nanosecond epoch stamp is astronomically larger, so
        # the magnitude tells the two apart without depending on an attribute GDAL may not expose.
        if values.size and values.dtype.kind in "iu" and values.min() > 10**12:
            return pd.DatetimeIndex(pd.to_datetime(values))
        return None
