"""Every path Hapi accepts may be a `pathlib.Path`, not only a `str`.

pyramids types each of its readers and writers `str | Path`, so nothing in Hapi has a reason
to demand a string -- yet several signatures did, and one (`Parameters.save_parameters`)
raised `TypeError` on a `Path` outright because it built its output names by concatenation.
Call sites were paying for that with `str(...)` wrappers.

These tests hand a `Path` to every reader and writer that takes one and run the model
through to the files on disk. The str spelling stays covered by the rest of the suite, which
feeds these same readers from `str` fixtures; what is pinned here is that neither spelling is
privileged.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame

from hapi.catchment import Catchment
from hapi.conceptual.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.engine.run import Run
from hapi.inputs import FlowNetwork, MeteoInputs


@pytest.fixture(scope="module")
def distributed_from_paths(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_muskingum: str,
    coello_cat_area: int,
    coello_initial_cond: list,
    coello_gauges_table: str,
    coello_gauges_path: str,
) -> Catchment:
    """Assemble and run distributed Coello with every path given as a `Path`.

    The fixtures hand out `str`; each is wrapped in `Path` here and passed unconverted, so
    the assembly itself is the assertion -- a reader that still demanded a string would
    raise during setup.

    Returns:
        Catchment: Model with a completed Muskingum run.
    """
    model = Catchment(
        "coello-from-path-objects",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    model.meteo = MeteoInputs.from_rasters(
        Path(coello_prec_path),
        Path(coello_temp_path),
        Path(coello_evap_path),
        file_name_data_fmt="%Y.%m.%d",
    )
    model.flow_network = FlowNetwork.from_rasters(
        Path(coello_acc_path), Path(coello_fd_path)
    )
    model.read_parameters(Path(coello_dist_parameters_muskingum), False)
    model.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    model.read_gauge_table(Path(coello_gauges_table), Path(coello_acc_path))
    model.read_discharge_gauges(Path(coello_gauges_path), column="id", fmt="%Y-%m-%d")
    Run.run_distributed(model)
    return model


class TestDistributedReadersTakeAPath:
    """The distributed assembly, driven entirely by `Path` arguments."""

    def test_the_run_completes_and_fills_the_grid(
        self, distributed_from_paths: Catchment
    ):
        """Test that a Path-driven assembly produces the same shaped results.

        Test scenario:
            `MeteoInputs.from_rasters`, `FlowNetwork.from_rasters`, `read_parameters`,
            `read_gauge_table` and `read_discharge_gauges` each received a `Path`. If any of
            them still required a `str` the fixture would have raised, so reaching a routed
            `q_total` at grid size is the whole claim.
        """
        model = distributed_from_paths
        rows, cols = model.flow_network.rows, model.flow_network.cols
        steps = model.meteo.simulation_steps

        assert model.results.q_total is not None, "the run must fill q_total"
        assert model.results.q_total.shape == (rows, cols, steps), (
            f"q_total should be {(rows, cols, steps)}, got {model.results.q_total.shape}"
        )
        assert np.isfinite(
            model.results.q_total[~np.isnan(model.flow_network.flow_acc_arr)]
        ).all(), "the routed discharge must be finite inside the catchment"

    def test_the_gauge_table_and_observations_arrived(
        self, distributed_from_paths: Catchment
    ):
        """Test that the two gauge readers filled their frames from Path arguments.

        Test scenario:
            `read_gauge_table` is the one reader that used to branch on
            `path.endswith(".geojson")` -- an attribute a `Path` does not have. It is read
            off `Path(path).suffix` now, so a CSV table given as a `Path` still lands.
        """
        model = distributed_from_paths

        assert isinstance(model.GaugesTable, DataFrame), (
            f"the gauge table should be a frame, got {type(model.GaugesTable)}"
        )
        assert "cell_row" in model.GaugesTable.columns, (
            "the accumulation raster given as a Path should have located each gauge; "
            f"columns are {model.GaugesTable.columns.tolist()}"
        )
        assert not model.QGauges.empty, (
            "the observed discharge frame should not be empty"
        )

    def test_flow_path_length_takes_a_path(
        self, distributed_from_paths: Catchment, coello_acc_path: str
    ):
        """Test that `read_flow_path_length` accepts a `Path`.

        Args:
            distributed_from_paths: A model whose grid the raster has to match.
            coello_acc_path: Stands in as a same-grid raster to read.

        Test scenario:
            The reader forwards straight to pyramids, and its docstring already promised
            `str | Path` while its annotation said `str`. Any raster on the model's own grid
            proves the argument is accepted.
        """
        model = distributed_from_paths
        model.read_flow_path_length(Path(coello_acc_path))

        assert model.flow_path_length_arr is not None, (
            "a Path argument should have populated flow_path_length_arr"
        )

    def test_results_are_written_from_path_arguments(
        self, distributed_from_paths: Catchment, coello_acc_path: str, tmp_path
    ):
        """Test that `SimulationResults.save` writes rasters from `Path` arguments.

        Args:
            distributed_from_paths: A model with a completed run.
            coello_acc_path: The georeferencing template.
            tmp_path: pytest's per-test temporary directory, passed unconverted.

        Test scenario:
            `save` used to reject anything that was not a `str` before it looked at
            anything else, so `tmp_path` had to be wrapped at every call site. Both the
            destination and the template are `Path` here, and the destination is a
            directory that does not exist yet, which exercises the `mkdir` branch too.
        """
        destination = tmp_path / "rasters"
        model = distributed_from_paths

        model.results.save(
            path=destination,
            result=1,
            flow_acc_path=Path(coello_acc_path),
            start=dt.datetime(2009, 1, 1),
            end=dt.datetime(2009, 1, 3),
        )

        written = sorted(destination.glob("*.tif"))
        assert len(written) == 3, (
            f"three steps should write three rasters, got {[f.name for f in written]}"
        )


class TestLumpedReadersTakeAPath:
    """The lumped assembly reads three files, none of which needs converting."""

    def test_a_lumped_run_assembles_and_runs_from_paths(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        lumped_parameters_path: str,
        lumped_gauges_path: str,
        coello_cat_area: int,
        coello_initial_cond: list,
        coello_gauges_date_fmt: str,
    ):
        """Test that the lumped readers accept `Path` arguments end to end.

        Test scenario:
            `read_lumped_inputs`, `read_parameters` and `read_discharge_gauges` all hand
            their argument to pandas or numpy, which take either spelling -- only Hapi's own
            annotations stood in the way. Running to a hydrograph covers all three at once.
        """
        model = Catchment("lumped-from-path-objects", *coello_rrm_date)
        model.read_lumped_inputs(Path(lumped_meteo_data_path))
        model.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        model.read_parameters(Path(lumped_parameters_path), False)
        model.read_discharge_gauges(
            Path(lumped_gauges_path), fmt=coello_gauges_date_fmt
        )

        results = Run.run_lumped(model)

        assert results.q_total is not None, "the lumped run must produce a hydrograph"
        assert len(results.q_total) == len(model.period.date_index), (
            f"the hydrograph should span {len(model.period.date_index)} steps, "
            f"got {len(results.q_total)}"
        )

    def test_a_lumped_run_writes_its_csv_to_a_path(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        lumped_parameters_path: str,
        coello_cat_area: int,
        coello_initial_cond: list,
        tmp_path,
    ):
        """Test that the CSV branch of `save` takes a `Path` naming the file.

        Args:
            coello_rrm_date: The run's start and end.
            lumped_meteo_data_path: The lumped drivers.
            lumped_parameters_path: The lumped parameter file.
            coello_cat_area: Catchment area.
            coello_initial_cond: Initial conditions.
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            Here `path` is the output file rather than a directory, which is the other half
            of what `save` dispatches on. pandas writes to a `Path` happily; it was Hapi's
            up-front `isinstance(path, str)` that refused it.
        """
        model = Catchment("lumped-csv-from-path", *coello_rrm_date)
        model.read_lumped_inputs(Path(lumped_meteo_data_path))
        model.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
        model.read_parameters(Path(lumped_parameters_path), False)
        results = Run.run_lumped(model)

        destination = tmp_path / "discharge.csv"
        results.save(path=destination, result=1)

        assert destination.exists(), f"{destination} should have been written"
        assert destination.read_text().startswith("date,Qsim"), (
            f"the CSV should carry the discharge header, got "
            f"{destination.read_text()[:40]!r}"
        )
