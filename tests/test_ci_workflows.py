"""The packaging checks in `pure-wheel-test.yml` must name the package as it actually is.

That workflow only runs on pushes to `main`, so a stale module path in it cannot fail a pull
request -- it turns `main` red on the merge that introduced it. These tests read the workflow and
hold its paths against the source tree on every run, which is the only place a rename can be caught
before it lands.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "pure-wheel-test.yml"
PACKAGE = REPO / "src" / "hapi"


def step(name: str) -> str:
    """Return the `run` script of the workflow step called `name`."""
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    for job in jobs.values():
        for candidate in job["steps"]:
            if candidate.get("name") == name:
                return candidate["run"]
    raise LookupError(f"no step named {name!r} in {WORKFLOW.name}")


WHEEL_STEP = "Verify the wheel contains the package modules"
IMPORT_STEP = "Verify package import and console scripts"


class TestPureWheelWorkflow:
    """Tests for the module paths `pure-wheel-test.yml` checks the built wheel against."""

    def test_every_grepped_path_exists_in_the_source_tree(self):
        """Test that the wheel check greps only for files the package really has.

        Test scenario:
            A `grep` for a module that no longer exists exits 1 and fails `build-wheel`. The
            restructure left three such paths (`hapi/catchment.py`, `hapi/rrm/distrrm.py`,
            `hapi/parameters/parameters.py`), and nothing on the pull request noticed.
        """
        targets = re.findall(r'grep "(hapi/[^"]+)"', step(WHEEL_STEP))
        missing = [
            target for target in targets if not (PACKAGE.parent / target).is_file()
        ]

        assert targets, f"the {WHEEL_STEP!r} step greps for nothing"
        assert not missing, (
            f"{WORKFLOW.name} greps the wheel for files that do not exist: {missing}"
        )

    def test_every_subpackage_is_checked(self):
        """Test that each subpackage's `__init__.py` is one of the wheel check's targets.

        Test scenario:
            The step exists to catch a subpackage silently dropping out of the distribution. A
            subpackage added to `src/hapi` but not to the step is exactly the case it would miss.
        """
        targets = set(re.findall(r'grep "(hapi/[^"]+)"', step(WHEEL_STEP)))
        subpackages = sorted(p.parent.name for p in PACKAGE.glob("*/__init__.py"))
        unchecked = [
            name for name in subpackages if f"hapi/{name}/__init__.py" not in targets
        ]

        assert subpackages, "found no subpackages under src/hapi"
        assert not unchecked, (
            f"{WORKFLOW.name} does not check these subpackages: {unchecked}"
        )

    def test_the_import_smoke_test_imports_real_names(self):
        """Test that every `from hapi... import X` in the smoke step resolves.

        Test scenario:
            The step runs a one-line `python -c` against the installed wheel. A moved module turns
            it into a `ModuleNotFoundError` after merge; resolving each import here moves that
            failure to the pull request.
        """
        imports = re.findall(r"from (hapi[\w.]*) import (\w+)", step(IMPORT_STEP))
        unresolved = []
        for module, name in imports:
            try:
                resolved = hasattr(importlib.import_module(module), name)
            except ImportError:
                resolved = False
            if not resolved:
                unresolved.append(f"{module}.{name}")

        assert imports, f"the {IMPORT_STEP!r} step imports nothing from hapi"
        assert not unresolved, f"the smoke import names that do not exist: {unresolved}"
