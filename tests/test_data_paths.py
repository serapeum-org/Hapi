"""Every data path a test names must exist.

The fixtures read the Coello and Jiboa datasets through repo-root-relative string literals
(`"tests/datasets/coello/meteo.nc"`), so moving the data directory breaks nothing at import time
-- a stale literal only surfaces when its test runs and raises `FileNotFoundError`, one test at a
time. This checks every such literal statically, so a move that misses one fails in one place with
the file and line.

Only string constants whose whole value is a `tests/...` path are checked; a path mentioned inside
prose is not. Paths that are named on purpose but never committed -- run-time output directories
and the uncommitted Jiboa raster folders -- are excluded, since a fresh checkout does not have them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"

#: A string constant that is nothing but a repo-relative path under `tests/`.
PATH_LITERAL = re.compile(r"tests/[A-Za-z0-9_.\-/]+")

#: Paths that are named on purpose but are not in the repository: directories a test writes into
#: rather than reads from, and the Jiboa raster folders, which are deliberately not committed (see
#: the NOTE in `tests/datasets/jiboa/convert_and_combine_meteo_inputs_to_netcdf.py`). An empty
#: directory is not tracked by git, so a local checkout can have these while CI never does.
NOT_IN_REPOSITORY = (
    "tests/datasets/test_results",
    "tests/datasets/parameters/download_files",
    "tests/datasets/jiboa/meteo_inputs",
)


def path_literals() -> list[tuple[str, str]]:
    """Return `(location, literal)` for every whole-string `tests/...` path in the test tree."""
    found: list[tuple[str, str]] = []
    for module in sorted(TESTS.rglob("*.py")):
        if "__pycache__" in module.parts:
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            value = node.value.rstrip("/")
            if not PATH_LITERAL.fullmatch(value):
                continue
            if value.startswith(NOT_IN_REPOSITORY):
                continue
            location = f"{module.relative_to(REPO).as_posix()}:{node.lineno}"
            found.append((location, value))
    return found


LITERALS = path_literals()


def test_the_scan_finds_the_dataset_literals():
    """Test that the scan is measuring something.

    Test scenario:
        The fixtures name the Coello dataset in more than a dozen places. If a change to the
        pattern or the directory layout left the scan empty, the parametrized test below would
        collect zero cases and pass vacuously.
    """
    datasets = [literal for _, literal in LITERALS if literal.startswith("tests/datasets/")]

    assert len(datasets) >= 10, (
        f"expected the fixtures' dataset paths to be found, got {len(datasets)}: {datasets}"
    )


@pytest.mark.parametrize(
    ("location", "literal"), LITERALS, ids=[location for location, _ in LITERALS]
)
def test_the_named_path_exists(location: str, literal: str):
    """Test that a path literal in the test tree points at something on disk.

    Args:
        location: `file:line` of the literal.
        literal: The repo-root-relative path it names.

    Test scenario:
        Moving `tests/rrm/data` to `tests/datasets` left nine files still naming the old
        directory in the commit, which 85 tests only discovered one `FileNotFoundError` at a
        time. Checked statically here, the same miss fails once per literal, at its line.
    """
    assert (REPO / literal).exists(), f"{location} names {literal!r}, which does not exist"
