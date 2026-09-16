"""A subpackage's re-exports must not make its modules expensive to import.

`from hapi.inputs import MeteoInputs` works because `hapi/inputs/__init__.py` re-exports the name.
When that `__init__` imported every submodule eagerly, importing *any* module under the package
paid for all of them: `import hapi.calibration.distribution`, which needs only `DEM`, loaded 31
`hapi` modules plus matplotlib and Oasis, because Python runs the package `__init__` first and that
`__init__` imported `search`, which imports the builder and the engine.

The re-exports are resolved on first attribute access instead. Each check runs in a fresh
interpreter, since import side effects are only visible from a clean `sys.modules`.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

#: Subpackages whose `__init__` re-exports names from their modules.
REEXPORTING = ("inputs", "conceptual", "calibration", "model")


def modules_loaded_by(statement: str) -> list[str]:
    """Run `statement` in a fresh interpreter and return the `hapi` modules it loaded.

    The child gets this process's `sys.path`, so it resolves `hapi` exactly as the test run does
    -- the source tree locally, the installed wheel in the pure-wheel job.
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    probe = (
        f"import sys; {statement}; "
        "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('hapi'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, f"{statement!r} failed:\n{result.stderr[-600:]}"
    return result.stdout.split()


@pytest.mark.parametrize("package", REEXPORTING)
def test_importing_the_package_loads_none_of_its_modules(package: str):
    """Test that `import hapi.<package>` runs no submodule of that package.

    Args:
        package: The re-exporting subpackage under test.

    Test scenario:
        Python runs a package's `__init__` before any module inside it, so an eager re-export in
        the `__init__` is paid by every import under the package. Lazy, it costs nothing until a
        re-exported name is actually used.
    """
    loaded = modules_loaded_by(f"import hapi.{package}")
    submodules = [m for m in loaded if m.startswith(f"hapi.{package}.")]

    assert not submodules, f"import hapi.{package} ran {submodules}"


@pytest.mark.parametrize(
    ("statement", "must_not_load"),
    [
        (
            "import hapi.calibration.distribution",
            ("hapi.calibration.search", "hapi.model.catchment", "hapi.engine.wrapper"),
        ),
        ("import hapi.model.lake", ("hapi.model.catchment",)),
        (
            "import hapi.inputs.dem",
            ("hapi.inputs.meteo", "hapi.inputs.network", "hapi.inputs.preparation"),
        ),
    ],
    ids=["calibration.distribution", "model.lake", "inputs.dem"],
)
def test_a_module_does_not_drag_in_its_siblings(
    statement: str, must_not_load: tuple[str, ...]
):
    """Test that importing one module leaves unrelated siblings unloaded.

    Args:
        statement: The import under test.
        must_not_load: Modules it has no reason to load.

    Test scenario:
        `Parameters` needs only `DEM`, `Lake` needs no `Catchment`, and `DEM` needs none of the
        driver readers. Each of these was loaded anyway through the package `__init__`.
    """
    loaded = modules_loaded_by(statement)
    dragged = [m for m in must_not_load if m in loaded]

    assert not dragged, f"{statement} also loaded {dragged}"


@pytest.mark.parametrize("package", REEXPORTING)
def test_every_listed_name_resolves_to_its_defining_object(package: str):
    """Test that each name in `__all__` is the very object its submodule defines.

    Args:
        package: The re-exporting subpackage under test.

    Test scenario:
        A lazy re-export is a table from name to module. A typo in the table, or a name moved to
        another module, would otherwise only surface when a user first touched the name.
    """
    module = importlib.import_module(f"hapi.{package}")
    exports = module._EXPORTS

    assert sorted(module.__all__) == sorted(exports), (
        f"hapi.{package}.__all__ and its export table disagree: {module.__all__} vs {sorted(exports)}"
    )
    for name, source in exports.items():
        defined = getattr(importlib.import_module(source), name)
        assert getattr(module, name) is defined, (
            f"hapi.{package}.{name} is not {source}.{name}"
        )


@pytest.mark.parametrize("package", REEXPORTING)
def test_an_unknown_name_is_an_attribute_error(package: str):
    """Test that a name the package does not export raises `AttributeError`.

    Args:
        package: The re-exporting subpackage under test.

    Test scenario:
        `hasattr` and `from package import submodule` both rely on `AttributeError` for a missing
        name; a `KeyError` leaking out of the lookup table would break both.
    """
    module = importlib.import_module(f"hapi.{package}")

    with pytest.raises(AttributeError, match="no attribute 'not_a_real_name'"):
        getattr(module, "not_a_real_name")


@pytest.mark.parametrize("package", REEXPORTING)
def test_dir_lists_the_names_before_they_are_touched(package: str):
    """Test that `dir()` shows every re-exported name, loaded or not.

    Args:
        package: The re-exporting subpackage under test.

    Test scenario:
        Tab completion and `dir()` read the module's namespace, where a lazy name does not exist
        until first use. `__dir__` has to add them back.
    """
    listed = modules_loaded_by(
        f"import hapi.{package} as p; missing = set(p.__all__) - set(dir(p)); "
        "assert not missing, missing"
    )

    assert f"hapi.{package}" in listed, f"the probe did not import hapi.{package}"


#: The public top-level names of the single module each of these packages replaced, as they were
#: on `main` before the restructure. The package's commit said its imports were unaffected, so
#: every one of them has to stay importable from the package itself.
OLD_MODULE_NAMES = {
    "calibration": (
        "COLUMNS_MISMATCH_ERROR",
        "Calibration",
        "OBJECTIVE_FN_ARGS_ERROR",
        "ObjectiveFunctionArityError",
        "ROWS_MISMATCH_ERROR",
    ),
    "conceptual": (
        "ConceptualModelSetup",
        "PARAMETER_COUNTS",
        "ParameterBounds",
        "ParameterSet",
        "parameter_count",
        "validate_initial_cond",
        "validate_parameter_count",
        "validate_q_init",
    ),
    "inputs": (
        "D8_CODES",
        "FlowNetwork",
        "Inputs",
        "METEO_VARIABLES",
        "MeteoInputs",
        "PARAMETERS_LIST",
        "RIVER_GEOMETRY_RASTERS",
        "RiverGeometry",
        "read_rasters",
    ),
}


@pytest.mark.parametrize("package", sorted(OLD_MODULE_NAMES))
def test_a_package_keeps_every_name_its_old_module_had(package: str):
    """Test that a package named after the module it replaced still exports all of its names.

    Args:
        package: A subpackage that took over a single module's import path.

    Test scenario:
        `hapi.calibration` became a package whose `__init__` re-exported only `Calibration` and
        `ObjectiveFunctionArityError`, so `from hapi.calibration import OBJECTIVE_FN_ARGS_ERROR`
        -- valid against the old module -- raised `ImportError`, while the commit said imports of
        `hapi.calibration` were unaffected.
    """
    module = importlib.import_module(f"hapi.{package}")
    missing = [name for name in OLD_MODULE_NAMES[package] if name not in module.__all__]

    assert not missing, f"hapi.{package} no longer exports {missing}"
