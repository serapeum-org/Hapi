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
    Run.runFW1(coello)
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
    expected = coello_run.state_variables[:, :, start_i, 0]
    actual = Dataset.read_file(str(written[0])).read_array(band=0)

    np.testing.assert_allclose(
        actual, expected, rtol=1e-5, err_msg="the first raster must hold the first step"
    )
