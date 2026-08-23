"""End-to-end Coello runs whose only meteorological input is the single combined NetCDF.

`test_meteo_inputs` covers the loaders and pins that a NetCDF-driven Muskingum run reproduces
the raster-driven one. This module covers the rest of the case study: from one
`MeteoInputs.from_netcdf` call through routing, gauge extraction, metrics and the saved
rasters, on both routing paths.

The point is the whole chain rather than any one link. `meteo.nc` packs three folders of
GeoTIFFs into one file with the calendar inside it, so a run driven from it touches no
meteorological raster at all -- and everything downstream, including the files written to
disk, has to come out exactly as it does from the folders.
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame
from pyramids.dataset import Dataset

from hapi.catchment import Catchment
from hapi.inputs import METEO_VARIABLES, FlowNetwork, MeteoInputs
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run

COMBINED_NC = "tests/rrm/data/coello/meteo.nc"

METRIC_NAMES = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]


@pytest.fixture(scope="module")
def meteo_from_one_file() -> MeteoInputs:
    """Load all three drivers from the single combined NetCDF.

    Returns:
        MeteoInputs: The three cubes and the calendar, from `meteo.nc` alone.
    """
    return MeteoInputs.from_netcdf(
        COMBINED_NC,
        precipitation="precipitation",
        temperature="temperature",
        evapotranspiration="evapotranspiration",
    )


def _build(
    name: str,
    meteo: MeteoInputs,
    parameters: str,
    setup: dict,
    *,
    maxbas: bool,
    with_flow_direction: bool,
) -> Catchment:
    """Assemble a distributed Coello model on the given drivers.

    Args:
        name: Model name.
        meteo: The drivers to run on.
        parameters: Folder of distributed parameter rasters.
        setup: The non-meteorological inputs, from the `setup` fixture.
        maxbas: Whether the parameter set is the triangular-routing one.
        with_flow_direction: Whether to load the direction raster. MAXBAS never reads it.

    Returns:
        Catchment: Model ready to run, with gauges loaded.
    """
    model = Catchment(
        name,
        setup["start"],
        setup["end"],
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    model.meteo = meteo
    model.flow_network = FlowNetwork.from_rasters(
        setup["acc"], setup["fd"] if with_flow_direction else None
    )
    model.read_parameters(parameters, False, maxbas=maxbas)
    model.read_lumped_model(HBVLumped, setup["area"], setup["initial"])
    model.read_gauge_table(setup["gauges_table"], setup["acc"])
    model.read_discharge_gauges(setup["gauges"], column="id", fmt="%Y-%m-%d")
    return model


@pytest.fixture(scope="module")
def setup(
    coello_start_date: str,
    coello_end_date: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_muskingum: str,
    coello_cat_area: int,
    coello_initial_cond: list,
    coello_gauges_table: str,
    coello_gauges_path: str,
) -> dict:
    """The non-meteorological inputs the Coello case study needs.

    Returns:
        dict: Paths and constants for the GIS, parameter and gauge inputs.
    """
    return dict(
        start=coello_start_date,
        end=coello_end_date,
        acc=coello_acc_path,
        fd=coello_fd_path,
        parameters=coello_dist_parameters_muskingum,
        area=coello_cat_area,
        initial=coello_initial_cond,
        gauges_table=coello_gauges_table,
        gauges=coello_gauges_path,
    )


@pytest.fixture(scope="module")
def meteo_from_folders(
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_start_date: str,
    coello_end_date: str,
) -> MeteoInputs:
    """The same drivers read from the three raster folders, as the control.

    Returns:
        MeteoInputs: The cubes `meteo.nc` was packed from.
    """
    return MeteoInputs.from_rasters(
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        start=coello_start_date,
        end=coello_end_date,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        file_name_data_fmt="%Y.%m.%d",
    )


@pytest.fixture(scope="module")
def muskingum_run(meteo_from_one_file: MeteoInputs, setup: dict) -> Catchment:
    """A completed Muskingum run driven from `meteo.nc`.

    Returns:
        Catchment: Model with the routed fields populated.
    """
    model = _build(
        "coello-nc-muskingum",
        meteo_from_one_file,
        setup["parameters"],
        setup,
        maxbas=False,
        with_flow_direction=True,
    )
    Run.RunHapi(model)
    return model


@pytest.fixture(scope="module")
def maxbas_run(
    meteo_from_one_file: MeteoInputs,
    setup: dict,
    coello_dist_parameters_maxbas: str,
) -> Catchment:
    """A completed triangular (MAXBAS) run driven from the same file.

    Returns:
        Catchment: Model with the routed fields populated.
    """
    model = _build(
        "coello-nc-maxbas",
        meteo_from_one_file,
        coello_dist_parameters_maxbas,
        setup,
        maxbas=True,
        with_flow_direction=False,
    )
    Run.runFW1(model)
    return model


@pytest.mark.e2e
class TestMuskingumPipeline:
    """The full Muskingum case study, from one NetCDF to the saved rasters."""

    def test_the_drivers_come_from_the_file_and_cover_the_model(
        self, muskingum_run: Catchment
    ):
        """Test that the run is driven by the file's own cubes and calendar.

        Test scenario:
            The first link: one `from_netcdf` call replaces three folder reads, and the
            calendar travels inside the file rather than in the file names. Everything below
            depends on this being the model's actual input, so pin the grid, the step count
            and the period against the model's own date index.
        """
        model = muskingum_run

        assert model.meteo.shape == (13, 14, 10), (
            f"expected a 13x14 grid over 10 steps, got {model.meteo.shape}"
        )
        assert model.meteo.time_steps == len(model.date_index), (
            f"the drivers hold {model.meteo.time_steps} steps but the model spans "
            f"{len(model.date_index)}"
        )
        assert model.meteo.time is not None, "the calendar must come out of the file"
        assert model.meteo.time[0] == model.date_index[0], (
            f"the drivers start at {model.meteo.time[0]}, the model at {model.date_index[0]}"
        )
        assert model.meteo.time[-1] == model.date_index[-1], (
            f"the drivers end at {model.meteo.time[-1]}, the model at {model.date_index[-1]}"
        )

    def test_routing_fills_the_distributed_fields(self, muskingum_run: Catchment):
        """Test that the run populates the per-cell output fields at grid size.

        Test scenario:
            `Qtot`, `quz_routed` and `qlz_translated` back every downstream reader --
            `extract_discharge`, `save_results`, the animations. All three must come back at
            `(rows, cols, simulation_steps)` and finite inside the catchment.
        """
        model = muskingum_run
        rows, cols = model.flow_network.rows, model.flow_network.cols
        steps = model.meteo.simulation_steps
        inside = ~np.isnan(model.flow_network.flow_acc_arr)

        for name in ("Qtot", "quz_routed", "qlz_translated"):
            field = getattr(model, name)
            assert field is not None, f"{name} must be set by the run"
            assert field.shape == (rows, cols, steps), (
                f"{name} should be {(rows, cols, steps)}, got {field.shape}"
            )
            assert np.isfinite(field[inside]).all(), (
                f"{name} must be finite inside the catchment"
            )

    def test_gauge_extraction_and_metrics(self, muskingum_run: Catchment):
        """Test that every gauge yields a hydrograph and all seven metrics.

        Test scenario:
            The end of the chain a modeller actually reads. `extract_discharge` walks the
            gauge table, pulls each gauge's cell out of `Qtot`, and scores it against the
            observations -- so this is where a driver that never reached the model, or
            reached it shifted in time, would finally show up as a non-finite score.
        """
        model = muskingum_run
        model.extract_discharge(calculate_metrics=True)

        n_gauges = len(model.GaugesTable)
        assert isinstance(model.metrics, DataFrame), (
            f"metrics should be a DataFrame, got {type(model.metrics)}"
        )
        assert list(model.metrics.index) == METRIC_NAMES, (
            f"metrics rows mismatch: {list(model.metrics.index)}"
        )
        assert model.metrics.shape[1] == n_gauges, (
            f"expected one column per gauge ({n_gauges}), got {model.metrics.shape[1]}"
        )
        assert np.isfinite(model.metrics.to_numpy(dtype=float)).all(), (
            "every metric must be finite"
        )
        assert model.Qsim.shape == (len(model.date_index), n_gauges), (
            f"Qsim shape mismatch: {model.Qsim.shape}"
        )
        assert np.isfinite(model.Qsim.to_numpy(dtype=float)).all(), (
            "the simulated hydrographs must be finite"
        )

    def test_saved_rasters_carry_the_routed_discharge(
        self, muskingum_run: Catchment, coello_acc_path: str, tmp_path
    ):
        """Test that the results reach disk as readable rasters holding `Qtot`.

        Test scenario:
            The last link, and the one nothing else exercises for the NetCDF-driven path:
            `save_results` writes one raster per step, georeferenced from the flow
            accumulation grid. Reading the first one back and comparing it to `Qtot`'s first
            slice proves the file holds the run's own numbers rather than an empty grid.
        """
        model = muskingum_run
        out = tmp_path / "muskingum"
        out.mkdir()

        model.save_results(flow_acc_path=coello_acc_path, result=1, path=f"{out}/")

        written = sorted(out.glob("*.tif"))
        assert written, "save_results must write at least one raster"
        assert len(written) == len(model.date_index), (
            f"expected one raster per step ({len(model.date_index)}), got {len(written)}"
        )

        first = Dataset.read_file(str(written[0])).read_array()
        expected = model.Qtot[:, :, 0]
        inside = ~np.isnan(model.flow_network.flow_acc_arr)
        np.testing.assert_allclose(
            np.asarray(first)[inside],
            expected[inside],
            rtol=1e-5,
            err_msg="the saved raster must hold the routed discharge of its step",
        )


@pytest.mark.e2e
class TestMaxbasPipeline:
    """The same case study through the triangular (MAXBAS) routing path."""

    def test_runs_without_a_flow_direction_raster(self, maxbas_run: Catchment):
        """Test that the triangular path completes on accumulation alone.

        Test scenario:
            MAXBAS sends every cell straight to the outlet, so it never reads a direction
            grid -- `FlowNetwork` makes that raster optional for exactly this path. Running
            it from the same NetCDF with no direction loaded proves the drivers and the
            network are independent of each other.
        """
        model = maxbas_run

        assert model.flow_network.has_flow_direction is False, (
            "the fixture must load accumulation only, or this proves nothing"
        )
        assert model.Qtot is not None, "the triangular run must fill Qtot"
        assert model._maxbas_routed is True, (
            "the triangular path must mark the model, so extract_discharge refuses the "
            "outlet-cell shortcut"
        )

    def test_basin_wide_discharge_and_metrics(self, maxbas_run: Catchment):
        """Test that the MAXBAS run scores against the gauges via the basin-wide sum.

        Test scenario:
            Triangular routing makes a cell of `Qtot` a contribution rather than a discharge,
            so the per-gauge shortcut is refused and `frame_work_1=True` selects the
            basin-wide sum instead. That is the only way to score this path, and it has to
            produce the same seven finite metrics.
        """
        model = maxbas_run

        with pytest.raises(ValueError, match="MAXBAS"):
            model.extract_discharge(calculate_metrics=False)

        model.extract_discharge(calculate_metrics=True, frame_work_1=True)

        assert isinstance(model.metrics, DataFrame), (
            f"metrics should be a DataFrame, got {type(model.metrics)}"
        )
        assert list(model.metrics.index) == METRIC_NAMES, (
            f"metrics rows mismatch: {list(model.metrics.index)}"
        )
        assert np.isfinite(model.metrics.to_numpy(dtype=float)).all(), (
            "every metric must be finite"
        )


@pytest.mark.e2e
class TestBothPathsAgreeWithTheRasterRun:
    """The equivalence claim that makes the packed file usable rather than merely readable."""

    def test_the_netcdf_run_matches_the_raster_run_cell_for_cell(
        self, muskingum_run: Catchment, meteo_from_folders: MeteoInputs, setup: dict
    ):
        """Test that packing the folders into one file does not move the hydrograph.

        Test scenario:
            The whole justification for `meteo.nc`: a run driven from it must be
            indistinguishable from one driven from the folders it was packed from. Compares
            the routed field and every gauge hydrograph, and checks the drivers themselves
            match so a failure says which layer moved.
        """
        raster_model = _build(
            "coello-rasters-e2e",
            meteo_from_folders,
            setup["parameters"],
            setup,
            maxbas=False,
            with_flow_direction=True,
        )
        Run.RunHapi(raster_model)

        for name in METEO_VARIABLES:
            np.testing.assert_array_equal(
                getattr(muskingum_run.meteo, name),
                getattr(meteo_from_folders, name),
                err_msg=f"{name} differs before the run even starts",
            )
        np.testing.assert_allclose(
            muskingum_run.Qtot,
            raster_model.Qtot,
            rtol=1e-9,
            err_msg="the routed discharge field differs between the two sources",
        )

        raster_model.extract_discharge(calculate_metrics=False)
        muskingum_run.extract_discharge(calculate_metrics=False)
        np.testing.assert_allclose(
            muskingum_run.Qsim.to_numpy(dtype=float),
            raster_model.Qsim.to_numpy(dtype=float),
            rtol=1e-9,
            err_msg="the per-gauge hydrographs differ between the two sources",
        )
