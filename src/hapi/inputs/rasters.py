"""Reading a folder of rasters, in the order the model needs them.

`read_rasters` is the one adapter over `DatasetCollection.from_files` that every other
reader in this package goes through, so the rule for deciding a folder's order -- by date
parsed from the file name, by a trailing index, or not at all -- is stated once.
"""

from __future__ import annotations

import datetime as dt
import re
import warnings
from pathlib import Path

from pyramids.dataset import DatasetCollection as Datacube


def _as_datetime(value: str | int | dt.datetime | None, fmt: str) -> dt.datetime | None:
    """Parse a `start` / `end` bound into a datetime for the date-ordered read.

    Args:
        value: The bound as given by the caller, or None for "unbounded". A `datetime` is
            returned as it is; `int` is admitted because the numeric ordering takes indices,
            and in the date branch a string is parsed with `fmt`.
        fmt: `strptime` format of the bound.

    Returns:
        dt.datetime | None: The parsed bound, or None when `value` is None.

    Raises:
        ValueError: `value` does not match `fmt`.
    """
    if value is None or isinstance(value, dt.datetime):
        return value
    return dt.datetime.strptime(str(value), fmt)


def _infer_date_format(sample: str) -> str | None:
    """Derive a `strptime` format from a date already matched out of a file name.

    `from_files` needs an explicit format before it will sort by date, but the caller has
    already said where the date sits via `regex_string`. Rather than leave the read
    unordered when no format is given, rebuild one from the shape of what the regex matched:
    the digit runs give the fields, the characters between them are kept verbatim.

    Only the unambiguous layouts are inferred. `1990.02.03` is `%Y.%m.%d` because a
    four-digit leading run can only be a year; `03.02.1990` is refused because day-first and
    month-first cannot be told apart from the digits alone.

    Args:
        sample: The substring `regex_string` matched, e.g. `"2009.01.01"` or `"20090101"`.

    Returns:
        str | None: The format, or None when the layout is ambiguous or unrecognised.

    Examples:
        >>> _infer_date_format("2009.01.01")
        '%Y.%m.%d'
        >>> _infer_date_format("2009_01_01")
        '%Y_%m_%d'
        >>> _infer_date_format("20090101")
        '%Y%m%d'
        >>> _infer_date_format("01.01.2009") is None
        True
    """
    parts = re.findall(r"\d+|\D+", sample)
    widths = tuple(len(p) for p in parts if p.isdigit())

    if widths == (8,):
        return "%Y%m%d"
    if widths != (4, 2, 2):
        # (2, 2, 4) and friends cannot be resolved: nothing in the digits says whether the
        # leading pair is the day or the month, and guessing wrong reorders the whole cube.
        return None
    if any("%" in p for p in parts if not p.isdigit()):
        return None

    directives = iter(("%Y", "%m", "%d"))
    return "".join(next(directives) if p.isdigit() else p for p in parts)


def _infer_date_format_from_folder(
    path: str | Path,
    glob: str,
    regex_string: str,
    gdal_env: dict[str, str] | None = None,
) -> str | None:
    """Sample a folder's file names and infer the date format they carry.

    Args:
        path: Folder holding the rasters.
        glob: `fnmatch` pattern selecting them.
        regex_string: Where the date sits in each name.
        gdal_env: GDAL configuration options applied while resolving the folder.

    Returns:
        str | None: The inferred `strptime` format, or None when no name matched the regex
            or the layout is ambiguous -- in which case the read falls back to unordered and
            a warning says so.
    """
    for file in Datacube.from_files(path, glob=glob, gdal_env=gdal_env).files:
        match = re.search(regex_string, Path(file).name)
        if match is None:
            continue
        inferred = _infer_date_format(match.group())
        if inferred is not None:
            return inferred
        warnings.warn(
            f"could not tell the date layout of {match.group()!r} in {Path(file).name!r} "
            f"apart (day-first and month-first look alike), so {path} is read in file-name "
            "order rather than by date; pass file_name_data_fmt to say which it is",
            stacklevel=3,
        )
        return None

    warnings.warn(
        f"regex {regex_string!r} matched no file name in {path}, so it is read in file-name "
        "order rather than by date; pass regex_string and file_name_data_fmt to order it",
        stacklevel=3,
    )
    return None


def read_rasters(
    path: str | Path,
    *,
    glob: str = "*.tif",
    regex_string: str = r"\d{4}.\d{2}.\d{2}",
    date: bool = True,
    file_name_data_fmt: str | None = None,
    start: str | int | dt.datetime | None = None,
    end: str | int | dt.datetime | None = None,
    fmt: str = "%Y-%m-%d",
    gdal_env: dict[str, str] | None = None,
) -> Datacube:
    r"""Read a folder of rasters into a `DatasetCollection` in the right order.

    A thin adapter over :meth:`DatasetCollection.from_files` -- pyramids does every bit of the
    resolving and reading; this only decides the order the files are handed over in, and
    translates Hapi's string/int `start` / `end` into what `from_files` accepts.

    Three orderings are supported, matching Hapi's public reader arguments:

    * **By date** (`date=True`) -- delegated wholesale to
      `from_files(date_format=..., date_regex=...)`, which sorts and builds the time axis.
      When no `file_name_data_fmt` is given it is inferred from the first name `regex_string`
      matches, so the default ordering is chronological rather than lexicographic.
    * **By number** (`date=False`) -- for names carrying a plain index, e.g.
      `01_Par_RFCF.tif` or `1000_Temp_..._1981_9_27.tif`. `from_files` sorts only by date, and
      its default order is lexicographic, which puts `10_` before `2_` whenever the index is
      not zero-padded. So the files are resolved through `from_files`, sorted on the integer in
      each name, and handed back to `from_files` as an explicit sequence -- which it keeps in
      the given order.
    * **Unordered** -- only when `date=True` and the layout cannot be inferred (an ambiguous
      day-first/month-first date, or a regex that matches no name). Both warn.

    Args:
        path: Folder holding the rasters.
        glob: :mod:`fnmatch` pattern selecting them. Defaults to `"*.tif"`.
        regex_string: Where the date (or the index, when `date=False`) sits in each name.
        date: Whether the matched value is a date. `False` selects the numeric ordering.
        file_name_data_fmt: `strptime` format of the date in the names. Inferred from the
            names themselves when omitted; pass it for a layout that cannot be told apart
            from the digits alone, such as a day-first `03.02.1990`.
        start: Inclusive lower bound -- a date string parsed with `fmt`, or an integer index
            when `date=False`.
        end: Inclusive upper bound; see `start`.
        fmt: `strptime` format of `start` / `end` when they are date strings.
        gdal_env: GDAL configuration options applied for the read, e.g.
            `{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}`. GDAL lists the directory on
            every open to look for sidecars, which on network storage is a remote listing
            per raster -- 369 ms against 18 ms in one measurement over 14,823 files. Left
            unset by default because disabling it also stops GDAL finding `.aux.xml`,
            world files and `.ovr`, which some of this repository's fixtures rely on.

    Returns:
        DatasetCollection: The collection, ordered as described above.

    Raises:
        FileNotFoundError: The folder does not exist, matched no file, or `start` / `end`
            excluded every file.
        ValueError: `regex_string` matched no number in a file name (numeric ordering only).
    """
    if date and file_name_data_fmt is None:
        # The caller said where the date is; that is enough to sort on. Reading the folder
        # unordered here used to hand back a lexicographic cube -- `10_precip_2009.01.11`
        # in slot 1 -- with no error and no time axis, so neither the length check nor the
        # calendar check could see it, and the run silently paired each day's rainfall with
        # the wrong date.
        file_name_data_fmt = _infer_date_format_from_folder(
            path, glob, regex_string, gdal_env
        )

    if date and file_name_data_fmt is not None:
        return Datacube.from_files(
            path,
            glob=glob,
            date_format=file_name_data_fmt,
            date_regex=regex_string,
            start=_as_datetime(start, fmt),
            end=_as_datetime(end, fmt),
            gdal_env=gdal_env,
        )

    if not date:
        # The numeric ordering bounds by the index in the file name, so a datetime has no
        # meaning here -- `int()` would fail on it several frames down.
        if isinstance(start, dt.datetime) or isinstance(end, dt.datetime):
            raise TypeError(
                "a datetime bound needs date=True; with date=False the rasters are ordered "
                "by the number in their name, so start/end are indices"
            )
        return _read_by_index(path, glob, regex_string, start, end, gdal_env)

    return Datacube.from_files(path, glob=glob, gdal_env=gdal_env)


def _read_by_index(
    path: str | Path,
    glob: str,
    regex_string: str,
    start: str | int | None,
    end: str | int | None,
    gdal_env: dict[str, str] | None = None,
) -> Datacube:
    """Read a folder whose names carry a plain index, ordered numerically.

    `from_files` sorts only by date and otherwise keeps lexicographic order, which puts
    `10_` before `2_` whenever the index is not zero-padded -- scrambling parameter rasters
    into the wrong HBV slots. So resolve the files, sort on the integer in each name, and
    hand them back as an explicit sequence, which `from_files` preserves.

    Args:
        path: Folder holding the rasters.
        glob: `fnmatch` pattern selecting them.
        regex_string: Where the index sits in each name.
        start: Inclusive lower bound on the index, or None.
        end: Inclusive upper bound on the index, or None.
        gdal_env: GDAL configuration options applied for the read.

    Returns:
        Datacube: The collection in ascending index order.

    Raises:
        ValueError: `regex_string` matched no number in one of the names.
        FileNotFoundError: `start` / `end` excluded every file.
    """
    keyed = []
    for file in Datacube.from_files(path, glob=glob, gdal_env=gdal_env).files:
        match = re.search(regex_string, Path(file).name)
        if match is None:
            raise ValueError(
                f"regex {regex_string!r} matched no number in {Path(file).name!r}"
            )
        keyed.append((int(match.group()), file))
    keyed.sort()

    if start is not None or end is not None:
        low = int(start) if start is not None else None
        high = int(end) if end is not None else None
        keyed = [
            (number, file)
            for number, file in keyed
            if (low is None or number >= low) and (high is None or number <= high)
        ]
        if not keyed:
            raise FileNotFoundError(
                f"no file in {path} carries an index within [{start}, {end}]"
            )

    return Datacube.from_files([file for _, file in keyed], gdal_env=gdal_env)


def _warn_if_no_sentinel(dataset, label: str) -> None:
    """Warn when a raster declares no no-data value, so the whole grid is the domain.

    Before masking was delegated to pyramids, a raster with no marker raised
    `TypeError` from `math.isclose(value, None)` — accidental, but loud. pyramids
    masks nothing instead, which is the correct reading of such a raster but silently
    makes every cell part of the catchment. Warn rather than raise: a raster legitimately
    having no marker is valid input.

    Args:
        dataset: The opened pyramids `Dataset`.
        label: Human-readable name of the input, used in the message.
    """
    if dataset.no_data_value[0] is None:
        warnings.warn(
            f"the {label} raster declares no no-data value, so every cell is treated as "
            "inside the catchment. If it has a sentinel, set it on the band; otherwise "
            "check that a whole-grid domain is intended.",
            UserWarning,
            stacklevel=3,
        )
