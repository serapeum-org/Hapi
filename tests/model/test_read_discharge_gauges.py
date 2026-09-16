"""Tests for the distributed branch of `Catchment.read_discharge_gauges`.

In distributed mode every gauge's discharge is its own `<id>.csv` inside one folder, and the
reader has two ways to open each file: from the header (the default) or skipping `readfrom`
leading rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hapi.model import catchment as catchment_module
from hapi.model.catchment import Catchment

GAUGE_IDS = (1, 2)


def write_gauges(folder: Path, leading_lines: int) -> Path:
    """Write a two-gauge table and one discharge CSV per gauge into `folder`.

    Args:
        folder: Where to write.
        leading_lines: Lines of boilerplate above each CSV's `date,<id>` header -- what
            `readfrom` exists to skip.

    Returns:
        Path: `folder`, holding `gauges.csv`, `1.csv` and `2.csv`.
    """
    pd.DataFrame({"id": GAUGE_IDS, "name": ["upper", "lower"]}).to_csv(
        folder / "gauges.csv", index=False
    )
    boilerplate = "exported by the gauging agency\n" * leading_lines
    for gauge in GAUGE_IDS:
        rows = "\n".join(f"2009-01-0{day},{gauge * 10 + day}" for day in (1, 2, 3))
        (folder / f"{gauge}.csv").write_text(
            f"{boilerplate}date,{gauge}\n{rows}\n", encoding="utf-8"
        )
    return folder


def _model(gauge_folder: Path) -> Catchment:
    """A distributed catchment over the three days the CSVs cover, with the table read."""
    model = Catchment(
        "gauges", "2009-01-01", "2009-01-03", spatial_resolution="Distributed"
    )
    model.read_gauge_table(gauge_folder / "gauges.csv")
    return model


class TestReadOneDischargeFilePerGauge:
    """Tests for how each gauge's file is located and read."""

    @pytest.mark.parametrize(
        ("readfrom", "leading_lines"),
        [("", 0), (1, 1)],
        ids=["from-header", "skip-rows"],
    )
    def test_each_gauge_file_is_joined_onto_the_folder(
        self, tmp_path: Path, readfrom, leading_lines: int, mocker
    ):
        """Test that both branches hand pandas `<folder>/<id>.csv` joined as a path.

        Args:
            tmp_path: pytest's per-test temporary directory.
            readfrom: `""` for the default branch, `1` for the row-skipping one.
            leading_lines: Boilerplate lines above each header, matching `readfrom`.
            mocker: Spies on `pandas.read_csv` to record what it is given.

        Test scenario:
            The default branch interpolated `f"{path}/{name}.csv"`, which on Windows splices
            a POSIX separator into a backslash path (`C:\\...\\tmp/1.csv`) while the other
            branch joined with `/`. Both must produce the path `Path(folder) / "<id>.csv"`
            would. On POSIX the two spellings coincide, so this can only fail on Windows.
        """
        gauge_folder = write_gauges(tmp_path, leading_lines)
        model = _model(gauge_folder)
        read_csv = mocker.spy(catchment_module.pd, "read_csv")

        model.read_discharge_gauges(
            gauge_folder, column="id", fmt="%Y-%m-%d", readfrom=readfrom
        )

        seen = [str(call.args[0]) for call in read_csv.call_args_list]
        expected = [str(gauge_folder / f"{gauge}.csv") for gauge in GAUGE_IDS]
        assert seen == expected, f"expected pandas to open {expected}, got {seen}"

    def test_readfrom_skips_leading_rows_before_the_header(self, tmp_path: Path):
        """Test that `readfrom` skips the lines above the header and reads the series.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            Each file opens with one line of agency boilerplate above `date,<id>`. With
            `readfrom=1` that line is skipped, so the header is found and every gauge gets its
            three days of discharge. No test exercised this branch before.
        """
        gauge_folder = write_gauges(tmp_path, leading_lines=1)
        model = _model(gauge_folder)

        model.read_discharge_gauges(
            gauge_folder, column="id", fmt="%Y-%m-%d", readfrom=1
        )

        assert list(model.QGauges.columns) == list(GAUGE_IDS), (
            f"expected one column per gauge id, got {list(model.QGauges.columns)}"
        )
        assert model.QGauges.loc["2009-01-02", 2] == 22, (
            f"gauge 2 on the second day should read 22, got {model.QGauges.loc['2009-01-02', 2]}"
        )
