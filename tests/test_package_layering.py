"""The subpackages form a layering, and this is what keeps them that way.

`src/hapi` is grouped into ranks: a module may import its own subpackage, or any lower rank,
and nothing else. The rule is not decoration -- the reason `hapi.simulation.protocols` exists
at all is so `hapi.engine` can state what a run needs without importing `hapi.model`, and
that decoupling is invisible to a reader and trivial to undo by accident.

Only imports that run when the module is imported are checked. One inside a function, or in
the body of an `if TYPE_CHECKING:` block, creates no runtime edge and cannot cause a cycle,
so `engine` naming `Lake` for an annotation is allowed. Everything else counts -- absolute
and relative imports alike, `from hapi import model`, and the `else` of a `TYPE_CHECKING`
guard, which does run.
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


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Whether an `if` test is exactly `TYPE_CHECKING` or `typing.TYPE_CHECKING`."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def runtime_imports(
    source: str, module: str, is_package: bool = False
) -> list[tuple[int, str]]:
    """Every `hapi` import that runs when `source` is imported, as `(line, dotted module)`.

    Args:
        source: Python source of the module.
        module: Its dotted name relative to `hapi`, e.g. `engine.wrapper`, which relative
            imports are resolved against.
        is_package: Whether the source is a package `__init__`, whose relative imports are
            resolved against the package itself rather than its parent.

    Returns:
        list[tuple[int, str]]: Each import's line and target, relative to `hapi`.
    """
    tree = ast.parse(source)
    deferred: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            deferred.update(id(child) for child in ast.walk(node))
        elif isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            # Only the guarded body is skipped: the `else` branch runs at import time.
            for statement in node.body:
                deferred.update(id(child) for child in ast.walk(statement))

    package = module.split(".") if is_package else module.split(".")[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in deferred:
            continue
        if isinstance(node, ast.Import):
            found.extend(
                (node.lineno, alias.name[len("hapi.") :])
                for alias in node.names
                if alias.name.startswith("hapi.")
            )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package) - (node.level - 1)
            if keep < 0:
                continue  # climbs above `hapi`, so it is not one of its modules
            base = package[:keep]
        elif node.module == "hapi":
            base = []
        elif (node.module or "").startswith("hapi."):
            found.append((node.lineno, node.module[len("hapi.") :]))
            continue
        else:
            continue
        if node.level and node.module:
            found.append((node.lineno, ".".join(base + node.module.split("."))))
        else:
            # `from hapi import model` / `from .. import model`: each name is a submodule.
            found.extend(
                (node.lineno, ".".join([*base, alias.name])) for alias in node.names
            )
    return found


def edges() -> list[tuple[str, int, str, str, str]]:
    """Every cross-subpackage runtime edge, as `(file, line, source pkg, target pkg, target)`."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        module = ".".join(path.relative_to(SRC).with_suffix("").parts)
        source = subpackage(module)
        if source is None:
            continue
        is_package = path.name == "__init__.py"
        if is_package:
            module = ".".join(path.relative_to(SRC).parent.parts)
        source_text = path.read_text(encoding="utf-8")
        for line, target_module in runtime_imports(source_text, module, is_package):
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

    def test_every_subpackage_directory_has_a_rank(self):
        """Test that every directory of modules under `src/hapi` is a key of `RANKS`.

        Test scenario:
            A subpackage missing from the table makes `subpackage()` return None for every
            module in it, so its edges are skipped in both directions and the layering test
            passes by measuring nothing. A directory holding modules without an `__init__.py`
            is still importable as a namespace package, so it counts too.
        """
        directories = sorted(
            child.name
            for child in SRC.iterdir()
            if child.is_dir()
            and child.name != "__pycache__"
            and any(p for p in child.rglob("*.py") if "__pycache__" not in p.parts)
        )
        unranked = [name for name in directories if name not in RANKS]

        assert directories, "found no subpackage directories under src/hapi"
        assert not unranked, (
            f"these subpackages have no rank, so nothing checks them: {unranked}"
        )


#: `(case, module the source lives in, is it a package __init__, source, expected targets)`.
#: Each source is one way to write an import. The expected targets are what a correct checker
#: must report as a runtime edge -- an empty tuple means the import creates no edge.
IMPORT_FORMS = [
    (
        "absolute from",
        "engine.wrapper",
        False,
        "from hapi.model import Catchment",
        ("model",),
    ),
    (
        "plain import",
        "engine.run",
        False,
        "import hapi.model.catchment",
        ("model.catchment",),
    ),
    (
        "relative two levels",
        "engine.wrapper",
        False,
        "from ..model import Catchment",
        ("model",),
    ),
    (
        "relative into a module",
        "engine.wrapper",
        False,
        "from ..model.lake import Lake",
        ("model.lake",),
    ),
    (
        "relative sibling",
        "engine.wrapper",
        False,
        "from .routing import Routing",
        ("engine.routing",),
    ),
    (
        "relative from a package init",
        "engine",
        True,
        "from ..model import Catchment",
        ("model",),
    ),
    ("relative bare names", "engine.run", False, "from .. import model", ("model",)),
    (
        "from the root package",
        "engine.run",
        False,
        "from hapi import model, simulation",
        ("model", "simulation"),
    ),
    (
        "negated guard",
        "engine.run",
        False,
        "from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    from hapi.model import Catchment",
        ("model",),
    ),
    (
        "else of a guard",
        "engine.run",
        False,
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    from hapi.model import Catchment",
        ("model",),
    ),
    (
        "guard only",
        "engine.run",
        False,
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from hapi.model import Catchment",
        (),
    ),
    (
        "typing.TYPE_CHECKING guard",
        "engine.run",
        False,
        "import typing\nif typing.TYPE_CHECKING:\n    from hapi.model import Catchment",
        (),
    ),
    (
        "inside a function",
        "engine.run",
        False,
        "def f():\n    from hapi.model import Catchment",
        (),
    ),
    (
        "third-party",
        "engine.run",
        False,
        "import numpy\nfrom pyramids.dataset import Dataset",
        (),
    ),
]


@pytest.mark.parametrize(
    ("module", "is_package", "source", "expected"),
    [case[1:] for case in IMPORT_FORMS],
    ids=[case[0] for case in IMPORT_FORMS],
)
def test_runtime_imports_sees_every_way_to_write_an_import(
    module: str, is_package: bool, source: str, expected: tuple[str, ...]
):
    """Test the checker against one synthetic source per way of writing an import.

    Args:
        module: Dotted name, relative to `hapi`, of the module the source lives in.
        is_package: Whether that module is a package `__init__`.
        source: The Python source to scan.
        expected: The targets a correct checker reports, in order.

    Test scenario:
        A form the checker cannot see is a one-line way around the whole layering rule: a
        relative import, `from hapi import model`, or an import under `if not TYPE_CHECKING`
        or in the `else` of a guard all run at import time. The guarded, in-function and
        third-party cases are the controls: they must stay invisible.
    """
    found = tuple(target for _, target in runtime_imports(source, module, is_package))

    assert found == expected, f"{module}: expected edges {expected}, got {found}"
