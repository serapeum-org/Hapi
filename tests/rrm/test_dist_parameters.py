import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pyramids.dataset import Dataset

from hapi.rrm.parameters import Parameters as DP

NO_DATA = -9999
"""No-data sentinel stamped on the synthetic rasters in :class:`TestParametersMasking`."""

NEAR_SENTINEL = -9990
"""Legitimate value within 0.1% of :data:`NO_DATA`.

``math.isclose(-9990, -9999, rel_tol=0.001)`` is ``True``, so the loop that
``Parameters.__init__`` used before delegating to ``read_array(masked=True)`` masked this
value. Exact integer comparison keeps it.
"""


def test_create_distparameters_instance(
    coello_acc_path: str,
    coello_acc_raster: Dataset,
    coello_no_parameters: int,
    coello_rows: int,
    coello_cols: int,
):
    klb = 0.5
    kub = 1
    no_lumped_par = 1
    lumped_par_pos = [7]

    SpatialVarFun = DP(
        coello_acc_raster,
        coello_no_parameters,
        no_lumped_par=no_lumped_par,
        lumped_par_pos=lumped_par_pos,
        function=2,
        k_lower_bound=klb,
        k_upper_bound=kub,
    )
    assert SpatialVarFun.no_lumped_par == no_lumped_par
    assert SpatialVarFun.lumped_par_pos == lumped_par_pos
    assert isinstance(SpatialVarFun.raster, Dataset)
    assert SpatialVarFun.rows == coello_rows
    assert SpatialVarFun.cols == coello_cols
    assert isinstance(SpatialVarFun.raster_array, np.ndarray)
    assert SpatialVarFun.no_parameters == 11
    assert SpatialVarFun.Par3d.shape == (coello_rows, coello_cols, coello_no_parameters)
    assert SpatialVarFun.totnumberpar == 980
    assert SpatialVarFun.Par2d.shape == (11, 89)


def test_par3d(
    coello_acc_path: str,
    coello_acc_raster: Dataset,
    coello_no_parameters: int,
    coello_parameters: np.ndarray,
    coello_parameters_dist: np.ndarray,
):
    klb = 0.5
    kub = 1
    no_lumped_par = 1
    lumped_par_pos = [7]

    SpatialVarFun = DP(
        coello_acc_raster,
        coello_no_parameters,
        no_lumped_par=no_lumped_par,
        lumped_par_pos=lumped_par_pos,
        function=2,
        k_lower_bound=klb,
        k_upper_bound=kub,
    )
    SpatialVarFun.Function(coello_parameters)
    arr = SpatialVarFun.Par3d
    assert np.array_equal(arr, coello_parameters_dist, equal_nan=True)


class TestParametersMasking:
    """Tests for the no-data masking ``Parameters.__init__`` delegates to pyramids."""

    @staticmethod
    def _raster(array: np.ndarray, no_data_value: int = NO_DATA) -> Dataset:
        """Build an in-memory Dataset from ``array``.

        Args:
            array: Pixel values for the single band.
            no_data_value: Sentinel stamped on the band.

        Returns:
            Dataset: An in-memory raster on a 4000 m UTM 18N grid.
        """
        return Dataset.create_from_array(
            array,
            top_left_corner=(0.0, array.shape[0] * 4000.0),
            cell_size=4000.0,
            epsg=32618,
            no_data_value=no_data_value,
        )

    def test_masks_no_data_cells_to_nan(self):
        """Test that sentinel cells become NaN in ``raster_array``.

        Test scenario:
            A 2x2 raster whose bottom-right cell holds the sentinel. Only that cell is NaN
            and the remaining three values survive unchanged.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32")), 12
        )

        assert np.isnan(parameters.raster_array[1, 1]), (
            f"the no-data cell should be NaN, got {parameters.raster_array[1, 1]}"
        )
        assert parameters.raster_array[0, 0] == 1.0, (
            f"the value 1 should survive unchanged, got {parameters.raster_array[0, 0]}"
        )

    def test_preserves_value_near_sentinel(self):
        """Test that a real value within 0.1% of the sentinel is no longer destroyed.

        Test scenario:
            The regression guarding the behaviour change on this branch. ``-9990`` beside a
            ``-9999`` marker survives, so every cell counts toward the domain and toward the
            parameter vector length.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NEAR_SENTINEL]], dtype="int32")), 12
        )

        assert not np.isnan(parameters.raster_array).any(), (
            f"no cell holds the sentinel, so nothing should be masked, got {parameters.raster_array}"
        )
        assert parameters.raster_array[1, 1] == float(NEAR_SENTINEL), (
            f"expected {NEAR_SENTINEL} to survive masking, got {parameters.raster_array[1, 1]}"
        )
        assert parameters.no_elem == 4, (
            f"all 4 cells are real data, got no_elem={parameters.no_elem}"
        )

    def test_no_elem_counts_domain_cells(self):
        """Test that ``no_elem`` counts only cells inside the domain.

        Test scenario:
            A 2x3 raster with two sentinel cells leaves four real cells, which sizes the
            per-cell parameter array.
        """
        parameters = DP(
            self._raster(np.array([[1, 2, 3], [4, NO_DATA, NO_DATA]], dtype="int32")),
            12,
        )

        assert parameters.no_elem == 4, (
            f"expected 4 domain cells out of 6, got {parameters.no_elem}"
        )
        assert parameters.Par2d.shape == (12, 4), (
            f"the per-cell parameter array should be 12x4, got {parameters.Par2d.shape}"
        )

    def test_cell_indices_exclude_no_data(self):
        """Test that ``celli``/``cellj`` list only domain cells, in row-major order.

        Test scenario:
            A 2x2 raster with one sentinel cell yields the three remaining coordinates.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32")), 12
        )

        assert list(zip(parameters.celli, parameters.cellj)) == [
            (0, 0),
            (0, 1),
            (1, 0),
        ], (
            "expected the three domain cells in row-major order, got "
            f"{list(zip(parameters.celli, parameters.cellj))}"
        )

    def test_raster_array_is_float(self):
        """Test that an integer source raster is promoted to float.

        Test scenario:
            An ``int32`` source must come back floating, since an integer array cannot hold
            the NaN used to mark cells outside the domain.
        """
        parameters = DP(
            self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32")), 12
        )

        assert parameters.raster_array.dtype.kind == "f", (
            f"expected a floating dtype so NaN can be stored, got {parameters.raster_array.dtype}"
        )

    def test_rejects_non_dataset_raster(self):
        """Test that a raw numpy array is rejected with ``TypeError``.

        Test scenario:
            ``Parameters`` requires a pyramids ``Dataset``; passing the array directly must
            fail with a message naming the expected type.
        """
        with pytest.raises(TypeError, match="pyramids Dataset"):
            DP(np.zeros((2, 2)), 12)

    def test_hru_mode_counts_distinct_classes(self):
        """Test that HRU mode sizes the domain by distinct class values, not by cell count.

        Test scenario:
            With ``hru=True`` the raster holds land-use classes rather than one value per
            cell, so ``no_elem`` is the number of distinct non-masked classes. Two cells
            share class ``1``, one holds class ``2``, and the sentinel cell is excluded,
            giving two classes over three domain cells.
        """
        parameters = DP(
            self._raster(np.array([[1, 1], [2, NO_DATA]], dtype="int32")), 12, hru=True
        )

        assert sorted(parameters.values) == [1, 2], (
            f"expected the distinct classes [1, 2], got {sorted(parameters.values)}"
        )
        assert parameters.no_elem == 2, (
            f"expected 2 distinct classes rather than 3 domain cells, got {parameters.no_elem}"
        )

    def test_hru_mode_excludes_masked_cells_from_classes(self):
        """Test that a masked cell never becomes an HRU class.

        Test scenario:
            The sentinel must not appear among the class values; only the two real classes
            are collected.
        """
        parameters = DP(
            self._raster(np.array([[5, 5], [7, NO_DATA]], dtype="int32")), 12, hru=True
        )

        assert NO_DATA not in parameters.values, (
            f"the no-data sentinel leaked into the HRU classes: {parameters.values}"
        )
        assert sorted(parameters.values) == [5, 7], (
            f"expected the distinct classes [5, 7], got {sorted(parameters.values)}"
        )

    def test_rejects_non_list_lumped_par_pos(self):
        """Test that a non-list ``lumped_par_pos`` is rejected with ``ValueError``.

        Test scenario:
            Declaring one lumped parameter but passing its position as a bare integer must
            fail with a message pointing at the expected list form.
        """
        raster = self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32"))

        with pytest.raises(ValueError, match="has to be entered as a list"):
            DP(raster, 12, no_lumped_par=1, lumped_par_pos=7)

    @pytest.mark.parametrize(
        "function, expected",
        [
            (1, "par3d_lumped"),
            (2, "par3d"),
            (3, "par2d_lumped_k1_lake"),
            (4, "hydrologic_response_units"),
        ],
    )
    def test_function_selects_distribution_strategy(self, function, expected):
        """Test that ``function`` binds the matching parameter-distribution strategy.

        Args:
            function: The strategy selector passed to the constructor.
            expected: Name of the method ``Function`` should be bound to.

        Test scenario:
            The four documented selectors each bind a different distribution strategy.
            Only the binding is checked — the strategies themselves are exercised by the
            ``par3d`` / ``par3d_lumped`` tests above, and ``par2d_lumped_k1_lake`` needs a
            lake parameter count it is not given here.
        """
        raster = self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32"))

        parameters = DP(raster, 12, function=function)

        assert parameters.Function.__name__ == expected, (
            f"function={function} should bind {expected}, got {parameters.Function.__name__}"
        )

    def test_hru_mode_overrides_the_selected_function(self):
        """Test that ``hru=True`` wins over an explicit ``function`` selector.

        Test scenario:
            The constructor documents that HRU mode overrides whatever strategy the
            caller selected. Asking for ``function=2`` (``par3d``) while passing
            ``hru=True`` must still bind the HRU strategy.
        """
        raster = self._raster(np.array([[1, 1], [2, NO_DATA]], dtype="int32"))

        parameters = DP(raster, 12, function=2, hru=True)

        assert parameters.Function.__name__ == "hydrologic_response_units", (
            f"hru=True should override function=2, got {parameters.Function.__name__}"
        )

    @pytest.mark.parametrize(
        "bad", [0, 5, 99, -1, "2", None, True, False, 1.0, 2.0, [1], {1}]
    )
    def test_unknown_function_is_rejected(self, bad):
        """Test that an unrecognised ``function`` selector fails at construction.

        Args:
            bad: A selector outside the documented set, including wrong types.

        Test scenario:
            The dispatch used to be an if/elif chain with no ``else``, so an unknown
            selector constructed happily and never bound ``Function`` at all. The mistake
            then surfaced far away, mid-calibration, as a bare ``AttributeError``. It is
            now rejected where it is supplied, and the message names the valid values.

            The parametrisation covers the type traps too: ``True`` is an ``int``
            subclass and would otherwise select strategy 1; ``1.0``/``2.0`` are caller
            errors rather than spellings of 1 and 2; and an unhashable ``[1]``/``{1}``
            used to raise ``TypeError`` from the membership test instead of the
            documented ``ValueError``.
        """
        raster = self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32"))

        with pytest.raises(ValueError, match=r"function must be one of \[1, 2, 3, 4\]"):
            DP(raster, 12, function=bad)

    def test_rejection_message_names_the_strategies(self):
        """Test that the error explains what each selector means.

        Test scenario:
            A bare list of valid integers would not tell a caller which one they wanted,
            so the message maps each to its strategy.
        """
        raster = self._raster(np.array([[1, 2], [3, NO_DATA]], dtype="int32"))

        with pytest.raises(ValueError) as exc_info:
            DP(raster, 12, function=99)

        message = str(exc_info.value)
        for name in (
            "par3d_lumped",
            "par3d",
            "par2d_lumped_k1_lake",
            "hydrologic_response_units",
        ):
            assert name in message, f"the error should name {name}, got: {message}"
        assert "99" in message, (
            f"the error should echo the rejected value, got: {message}"
        )

    def test_hru_override_still_validates_the_selector(self):
        """Test that ``hru=True`` does not mask an invalid selector.

        Test scenario:
            HRU mode overrides whatever strategy was chosen, so it would be easy to skip
            validation when it is on. A typo must still be reported rather than silently
            absorbed by the override.
        """
        raster = self._raster(np.array([[1, 1], [2, NO_DATA]], dtype="int32"))

        with pytest.raises(ValueError, match="function must be one of"):
            DP(raster, 12, function=99, hru=True)


SUBPROCESS_PROBE = """
import sys
import numpy as np
from pyramids.dataset import Dataset
from hapi.rrm.parameters import Parameters

raster = Dataset.create_from_array(
    np.array([[1, 2], [3, -9999]], dtype="int32"),
    top_left_corner=(0.0, 8000.0), cell_size=4000.0, epsg=32618, no_data_value=-9999,
)
try:
    Parameters(raster, 12).save_parameters("definitely-missing-dir/")
except FileNotFoundError:
    sys.exit(0)
sys.exit(1)
"""
"""Probe run in a child interpreter: exits 0 only if the missing-directory guard fires."""


def _raster() -> Dataset:
    """Build a small in-memory raster for the save_parameters tests.

    Returns:
        Dataset: A 2x2 raster on a 4000 m UTM 18N grid with one no-data cell.
    """
    return Dataset.create_from_array(
        np.array([[1, 2], [3, NO_DATA]], dtype="int32"),
        top_left_corner=(0.0, 8000.0),
        cell_size=4000.0,
        epsg=32618,
        no_data_value=NO_DATA,
    )


class TestSaveParametersValidation:
    """Tests for the input validation on ``Parameters.save_parameters``."""

    def test_missing_output_directory_raises(self, tmp_path):
        """Test that a non-existent output directory is rejected up front.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            ``save_parameters`` builds each output name by concatenating onto ``path``, so
            it cannot delegate to pyramids the way the readers do — the failure would
            otherwise surface per-file, midway through writing.
        """
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DP(_raster(), 12).save_parameters(str(tmp_path / "absent") + "/")

    def test_non_string_path_raises(self, tmp_path):
        """Test that a non-``str`` output path is rejected with ``TypeError``.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            Unlike the readers, this method really does need a ``str``: it concatenates
            ``path + name``, which ``Path`` does not support. The check is kept rather
            than delegated — but raised, not asserted.
        """
        with pytest.raises(TypeError, match="string"):
            DP(_raster(), 12).save_parameters(tmp_path)

    def test_validation_survives_optimised_mode(self):
        """Test that the guard still fires under ``python -O``.

        Test scenario:
            This is what made issue 3 more than a tidy-up. ``python -O`` strips every
            ``assert``, so the previous ``assert os.path.exists(path)`` did not exist in
            an optimised run and the method proceeded to write. Unlike the readers — where
            pyramids raises a moment later regardless — nothing downstream replaces this
            one, so it had to become a real ``raise``. Run in a child interpreter because
            ``-O`` is fixed at start-up.
        """
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-O", "-c", SUBPROCESS_PROBE],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        )

        # Distinguish "the guard did not fire" from "the child could not import hapi";
        # both exit non-zero, and conflating them reports a misleading cause.
        assert (
            "ImportError" not in result.stderr
            and "ModuleNotFoundError" not in result.stderr
        ), (
            f"the probe could not import hapi, so it never reached the guard: {result.stderr[-400:]}"
        )
        assert result.returncode == 0, (
            "save_parameters must still reject a missing output directory under "
            f"`python -O`; exit={result.returncode}, stderr={result.stderr[-400:]}"
        )

    @pytest.mark.parametrize("snow, expected", [(0, 12), (1, 15)])
    def test_writes_one_raster_per_parameter(self, tmp_path, snow, expected):
        """Test that a successful save writes one raster per distributed parameter.

        Args:
            tmp_path: pytest's per-test temporary directory.
            snow: Whether the snow subroutine is active.
            expected: Number of parameter rasters the branch should write.

        Test scenario:
            The two snow branches carry different parameter-name lists (12 without snow,
            15 with), and the writer emits one dated GeoTIFF per layer of ``Par3d``. The
            method had no tests before, so this covers the happy path alongside the
            validation guards above. ``path`` must end with a separator: output names are
            built by concatenation.
        """
        distributor = DP(_raster(), expected)
        distributor.Snow = snow
        distributor.Par3d = np.zeros((2, 2, expected), dtype="float32")

        distributor.save_parameters(str(tmp_path) + "/")

        written = sorted(tmp_path.glob("*.tif"))
        assert len(written) == expected, (
            f"snow={snow} should write {expected} rasters, got "
            f"{[f.name for f in written]}"
        )
        assert all(f.stat().st_size > 0 for f in written), (
            "every written raster should be non-empty, got "
            f"{[f.stat().st_size for f in written]}"
        )
