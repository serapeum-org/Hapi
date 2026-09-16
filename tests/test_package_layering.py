"""The subpackages form a layering, and this is what keeps them that way.

`src/hapi` is grouped into ranks: a module may import its own subpackage, or any lower rank,
and nothing else. The rule is not decoration -- the reason `hapi.simulation.protocols` exists
at all is so `hapi.engine` can state what a run needs without importing `hapi.model`, and
that decoupling is invisible to a reader and trivial to undo by accident.

Only module-scope imports are checked. An `if TYPE_CHECKING` import creates no runtime edge
and cannot cause a cycle, so `engine` naming `Lake` for an annotation is allowed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "hapi"

#: Which rank each subpackage sits at. A module may import its own rank only within its own
#: subpackage; across subpackages the target must sit strictly lower.
RANKS: dict[str, int] = {
    "core": 0,
    "data": 0,
    "inputs": 1,
    "conceptual": 1,
    "simulation": 2,
    "model": 3,
    "engine": 3,
    "calibration": 4,
}


def subpackage(module: str) -> str | None:
    """The subpackage a dotted `hapi.*` module belongs to, or None for the package root."""
    parts = module.split(".")
    return parts[0] if parts and parts[0] in RANKS else None


def runtime_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every module-scope `hapi.*` import in `path`, as `(line, dotted module)`.

    Imports inside a function and imports guarded by `if TYPE_CHECKING` are skipped: neither
    creates an edge at import time, which is what the layering is about.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    deferred: set[int] = set()
    for node in ast.walk(tree):
        guarded = isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)
        in_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if guarded or in_function:
            for child in ast.walk(node):
                deferred.add(id(child))

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in deferred:
            continue
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hapi."):
            found.append((node.lineno, node.module[len("hapi.") :]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("hapi."):
                    found.append((node.lineno, alias.name[len("hapi.") :]))
    return found


def edges() -> list[tuple[str, int, str, str, str]]:
    """Every cross-subpackage runtime edge, as `(file, line, source pkg, target pkg, target)`."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        module = ".".join(path.relative_to(SRC).with_suffix("").parts)
        source = subpackage(module)
        if source is None:
            continue
        for line, target_module in runtime_imports(path):
            target = subpackage(target_module)
            if target is None or target == source:
                continue
            rel = path.relative_to(SRC.parents[1]).as_posix()
            out.append((rel, line, source, target, target_module))
    return out


class TestPackageLayering:
    """Tests for the import direction between `hapi`'s subpackages."""

    def test_every_cross_package_import_points_down(self):
        """Test that no subpackage imports from its own rank or above.

        Test scenario:
            The ranks in `RANKS` are derived from the real import graph, so a violation means
            either a genuinely new dependency or a module filed in the wrong place. Both are
            worth a decision rather than a merge.
        """
        violations = [
            f"{rel}:{line}  {source}(rank {RANKS[source]}) -> {target}(rank {RANKS[target]})"
            f"  [{target_module}]"
            for rel, line, source, target, target_module in edges()
            if RANKS[target] >= RANKS[source]
        ]

        assert not violations, (
            "cross-package imports must point down a rank:\n" + "\n".join(violations)
        )

    def test_the_engine_does_not_import_the_builder(self):
        """Test that `hapi.engine` never imports `hapi.model` at runtime.

        Test scenario:
            This is the specific decoupling `hapi.simulation.protocols` was introduced for:
            the run layer describes what it needs structurally, and `Catchment` satisfies it
            without the engine ever naming the builder. `model` and `engine` share a rank, so
            the test above already forbids it -- this one fails with the reason attached,
            because that is the edge most likely to come back.
        """
        offenders = [
            f"{rel}:{line} imports {target_module}"
            for rel, line, source, target, target_module in edges()
            if source == "engine" and target == "model"
        ]

        assert not offenders, (
            "hapi.engine must not import hapi.model at runtime; state the requirement in "
            "hapi.simulation.protocols instead:\n" + "\n".join(offenders)
        )

    @pytest.mark.parametrize("name", sorted(RANKS))
    def test_each_declared_subpackage_exists(self, name: str):
        """Test that every subpackage named in `RANKS` is a real package.

        Args:
            name: The subpackage under test.

        Test scenario:
            Keeps the table honest: a renamed or removed subpackage silently stops being
            checked otherwise, and the layering test would pass by measuring nothing.
        """
        assert (SRC / name / "__init__.py").is_file(), (
            f"hapi.{name} is in the rank table but has no package at src/hapi/{name}"
        )

    def test_the_package_root_holds_no_modules(self):
        """Test that `src/hapi` itself contains only `__init__.py`.

        Test scenario:
            The point of the restructure. A module added at the root belongs to no rank, so
            the layering test would skip it entirely -- it has to be filed somewhere.
        """
        loose = sorted(p.name for p in SRC.glob("*.py") if p.name != "__init__.py")

        assert not loose, f"modules at the package root belong in a subpackage: {loose}"
