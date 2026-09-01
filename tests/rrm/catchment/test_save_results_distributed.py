"""Tests for the distributed branch of ``Catchment.save_results``.

The distributed branch scaffolds an in-memory ``DatasetCollection`` off the flow-accumulation
raster and writes one GeoTIFF per timestep. It was previously exercised only by the
non-collected script ``tests/run/distributed_mode_run.py``, so the pyramids call it makes was
unverified — these tests cover it directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run


@pytest.fixture(scope="module")
def coello_run(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_dist_parameters_maxbas: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> Catchment:
    """Distributed Coello catchment with a completed FW1 run."""
    coello = Catchment(
        "coello",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    kwargs = dict(
        start=coello_start_date,
        end=coello_end_date,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    coello.meteo = MeteoInputs.from_rasters(
        coello_prec_path, coello_temp_path, coello_evap_path, **kwargs
    )
    coello.flow_network = FlowNetwork.from_rasters(coello_acc_path)
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    Run.run_maxbas(coello)
    return coello


def test_save_results_distributed_writes_one_raster_per_step(
    coello_run: Catchment, coello_acc_path: str, tmp_path
):
    """Test that the distributed branch writes one readable GeoTIFF per timestep.

    Args:
        coello_run: Distributed Coello catchment with a completed run.
        coello_acc_path: Path to the flow-accumulation raster used as the template.
        tmp_path: Destination directory.

    Test scenario:
        `save_results` scaffolds a `DatasetCollection` from the flow-accumulation
        raster via pyramids' `from_dataset` named constructor and writes the
        selected result to one raster per date. Pins that the files land, are
        readable, and carry the template's grid.
    """
    out = tmp_path / "dist"
    out.mkdir()
    # result=4 is the snow-pack state variable. The discharge options are available after
    # a FW1 run too since `_set_maxbas_output_fields` landed; this covers the state-variable
    # branch, which reads a different array.
    coello_run.save_results(
        flow_acc_path=coello_acc_path,
        result=4,
        start="2009-01-01",
        end="2009-01-05",
        path=f"{out}/",
    )

    written = sorted(out.glob("*.tif"))
    assert len(written) == 5, f"expected one raster per date, got {len(written)}"

    template = Dataset.read_file(coello_acc_path)
    first = Dataset.read_file(str(written[0]))
    assert (first.rows, first.columns) == (template.rows, template.columns), (
        "the written raster must inherit the flow-accumulation grid"
    )


def test_save_results_distributed_values_match_the_model_array(
    coello_run: Catchment, coello_acc_path: str, tmp_path
):
    """Test that the written rasters carry the model's discharge values.

    Args:
        coello_run: Distributed Coello catchment with a completed run.
        coello_acc_path: Path to the flow-accumulation raster used as the template.
        tmp_path: Destination directory.

    Test scenario:
        The scaffold is filled by assigning `cube.values` a time-first array, so a
        wrong axis order or an off-by-one in the date slice would still write files
        but with the wrong content. Reads the first timestep back and compares it to
        the state variable's matching slice, which catches that.
    """
    out = tmp_path / "dist"
    out.mkdir()
    coello_run.save_results(
        flow_acc_path=coello_acc_path,
        result=4,
        start="2009-01-01",
        end="2009-01-03",
        path=f"{out}/",
    )

    written = sorted(out.glob("*.tif"))
    start_i = np.where(coello_run.date_index == np.datetime64("2009-01-01"))[0][0]
    expected = coello_run.results.state_variables[:, :, start_i, 0]
    actual = Dataset.read_file(str(written[0])).read_array(band=0)

    np.testing.assert_allclose(
        actual, expected, rtol=1e-5, err_msg="the first raster must hold the first step"
    )


def test_save_results_joins_a_directory_written_without_a_separator(
    coello_run: Catchment, coello_acc_path: str, tmp_path
):
    """Test that a directory given without a trailing separator still writes inside it.

    Args:
        coello_run: Distributed Coello catchment with a completed run.
        coello_acc_path: Path to the flow-accumulation raster used as the template.
        tmp_path: Parent of the destination directory.

    Test scenario:
        The names used to be built by concatenation, so `some/dir` produced
        `some/dirResult_2009-01-01.tif` -- a sibling of the directory rather than a file in
        it. The other tests in this file all pass a trailing separator, so none of them
        would notice.
    """
    out = tmp_path / "no-separator"
    out.mkdir()

    coello_run.save_results(
        flow_acc_path=coello_acc_path,
        result=4,
        start="2009-01-01",
        end="2009-01-02",
        path=str(out),
    )

    assert len(sorted(out.glob("*.tif"))) == 2, (
        f"the rasters must land inside the directory, found {sorted(tmp_path.iterdir())}"
    )


def test_save_results_creates_the_directory_it_is_given(
    coello_run: Catchment, coello_acc_path: str, tmp_path
):
    """Test that a destination directory that does not exist yet is created.

    Args:
        coello_run: Distributed Coello catchment with a completed run.
        coello_acc_path: Path to the flow-accumulation raster used as the template.
        tmp_path: Parent of the destination directory.

    Test scenario:
        `outputs.results_dir` in a run configuration names where results go, and nothing
        guarantees it exists before the first run. Covers a nested path, so a single
        `mkdir` would not be enough.
    """
    out = tmp_path / "nested" / "results"

    coello_run.save_results(
        flow_acc_path=coello_acc_path,
        result=4,
        start="2009-01-01",
        end="2009-01-02",
        path=str(out),
    )

    assert len(sorted(out.glob("*.tif"))) == 2, (
        f"the directory must be created and written into, got {out.exists()}"
    )


def test_save_results_refuses_a_path_that_is_not_a_string(coello_run: Catchment):
    """Test that a non-string `path` is refused by name rather than by concatenation.

    Args:
        coello_run: Distributed Coello catchment with a completed run.

    Test scenario:
        `outputs.results_dir` is optional in a run configuration, so a caller forwarding it
        straight through can hold None. That used to surface as a `TypeError` from a string
        concatenation, naming neither the argument nor what it should be.
    """
    with pytest.raises(TypeError, match="path must be a string") as exc:
        coello_run.save_results(flow_acc_path="unused", result=1, path=None)

    assert "NoneType" in str(exc.value), (
        f"the error should name what it got: {exc.value}"
    )
