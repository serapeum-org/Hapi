"""Tests for the two triangular-routing variants and the lumped routing branches.

`DistributedRRM.DistMaxbas2` is a public entry point that nothing inside the package calls -- it
rescales each cell's MAXBAS by its flow path length -- so it is reached only from a test or a
downstream user. `Wrapper.Lumped` picks its routing call by whether `maxbas` is set, a branch
the existing lumped tests never took.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.routing import Routing
from hapi.rrm.distrrm import DistributedRRM
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run
from hapi.wrapper import Wrapper


@pytest.fixture(scope="module")
def coello_before_routing(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_maxbas: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> Catchment:
    """Build a distributed catchment with the per-cell model run but no routing applied.

    Returns:
        Catchment: Model whose `quz`/`qlz` are populated and whose `flow_path_length_arr`
            increases away from the top-left corner.
    """
    coello = Catchment(
        "coello-maxbas",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    coello.meteo = MeteoInputs.from_rasters(
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        start=coello_start_date,
        end=coello_end_date,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    coello.flow_network = FlowNetwork.from_rasters(coello_acc_path, coello_fd_path)
    coello.read_parameters(coello_dist_parameters_maxbas, False, maxbas=True)
    coello.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    acc = coello.flow_network.flow_acc_arr
    rows, cols = coello.flow_network.rows, coello.flow_network.cols
    distance = np.add.outer(np.arange(rows), np.arange(cols)).astype("float64") * 1000.0
    coello.flow_path_length_arr = np.where(np.isnan(acc), np.nan, distance)
    DistributedRRM.run_lumped_model(coello)
    return coello


@pytest.fixture(scope="module")
def maxbas_parameters_path(lumped_parameters_path: str, tmp_path_factory) -> str:
    """Provide an eleven-value lumped parameter file, the MAXBAS configuration.

    The bundled lumped set is the twelve-value Muskingum one; the MAXBAS configuration
    replaces the two Muskingum parameters with a single MAXBAS width.

    Returns:
        str: Path to the written parameter file.
    """
    values = [
        line for line in Path(lumped_parameters_path).read_text().splitlines() if line
    ]
    path = tmp_path_factory.mktemp("maxbas") / "parameters.txt"
    path.write_text(os.linesep.join(values[:10] + ["maxbas,3.0"]) + os.linesep)
    return str(path)


class TestDistMaxbas2:
    """Tests for `DistributedRRM.DistMaxbas2`."""

    def test_conserves_volume_while_redistributing_it_in_time(
        self, coello_before_routing: Catchment
    ):
        """Test that the routing moves discharge between steps without creating or losing it.

        Test scenario:
            A triangular unit hydrograph is a normalised convolution kernel, so each cell's
            total over the record is invariant while its timing spreads. Asserting only an
            upper bound would pass for routing that destroyed 90% of the volume, so this is
            an equality.
        """
        model = coello_before_routing
        before = model.quz.copy()
        inside = ~np.isnan(model.flow_path_length_arr)

        DistributedRRM.DistMaxbas2(model)

        assert not np.array_equal(model.quz[inside], before[inside]), (
            "the routing must alter the upper-zone discharge of the masked cells"
        )
        assert np.isfinite(model.quz[inside]).all(), (
            "routed discharge must stay finite inside the catchment"
        )
        np.testing.assert_allclose(
            np.nansum(model.quz[inside]),
            np.nansum(before[inside]),
            rtol=1e-6,
            err_msg="triangular routing must conserve volume, not merely bound it",
        )

    def test_attenuates_the_peak_of_a_cell_far_from_the_outlet(
        self, coello_before_routing: Catchment
    ):
        """Test that the flow-path length actually scales the attenuation.

        Test scenario:
            This variant exists to give distant cells more attenuation than near ones -- that
            is the whole difference from `DistMaxbas1`. Comparing the peak reduction of the
            nearest and furthest in-domain cells is what distinguishes it from a uniform
            kernel; a routing that ignored the flow path would attenuate both equally.
        """
        model = coello_before_routing
        fpl = model.flow_path_length_arr
        inside = ~np.isnan(fpl)
        nearest = np.unravel_index(
            np.nanargmin(np.where(inside, fpl, np.nan)), fpl.shape
        )
        furthest = np.unravel_index(
            np.nanargmax(np.where(inside, fpl, np.nan)), fpl.shape
        )
        before = model.quz.copy()

        DistributedRRM.DistMaxbas2(model)

        near_drop = before[nearest].max() - model.quz[nearest].max()
        far_drop = before[furthest].max() - model.quz[furthest].max()
        assert far_drop > near_drop, (
            f"the cell {far_drop:.4g} from the outlet must be attenuated more than the near "
            f"one ({near_drop:.4g}); equal attenuation means the flow path was ignored"
        )

    def test_leaves_cells_outside_the_mask_untouched(
        self, coello_before_routing: Catchment
    ):
        """Test that cells with no flow-path length are skipped rather than routed.

        Test scenario:
            `run_lumped_model` leaves out-of-domain cells at zero, and zeros survive any
            convolution -- so comparing them before and after proves nothing. Seeding them
            with a recognisable series first is what makes the mask observable: if the guard
            were dropped, the convolution would reshape it.
        """
        model = coello_before_routing
        outside = np.isnan(model.flow_path_length_arr)
        sentinel = np.linspace(1.0, 10.0, model.quz.shape[2])
        model.quz[outside] = sentinel
        before = model.quz.copy()

        DistributedRRM.DistMaxbas2(model)

        np.testing.assert_array_equal(
            model.quz[outside],
            before[outside],
            err_msg="cells with no flow-path length must not be routed",
        )


def _lumped_model(
    dates: list,
    meteo_path: str,
    parameters_path: str,
    area: float,
    initial_cond: list,
) -> Catchment:
    """Build a lumped catchment ready for `Run.runLumped`.

    Args:
        dates: `[start, end]` simulation dates.
        meteo_path: Lumped meteorological CSV.
        parameters_path: Lumped parameter file.
        area: Catchment area coefficient.
        initial_cond: Initial HBV state.

    Returns:
        Catchment: Model with inputs, lumped model and parameters read.
    """
    model = Catchment("rrm", dates[0], dates[1])
    model.read_lumped_inputs(meteo_path)
    model.read_lumped_model(HBVLumped, area, initial_cond)
    model.read_parameters(parameters_path, False, maxbas=True)
    return model


class TestLumpedRouting:
    """Tests for the routing branches of `Wrapper.Lumped` reached through `Run.runLumped`."""

    def test_maxbas_routing_convolves_qsim_with_the_last_parameter(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        maxbas_parameters_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
    ):
        """Test that the MAXBAS branch routes on the trailing parameter and nothing else.

        Test scenario:
            `Wrapper.Lumped` has two routing calls: the MAXBAS one passes a single parameter,
            the Muskingum one passes three. Which runs is decided by the `maxbas` flag
            `read_parameters` stored. Asserting only that `Qsim` is finite would pass for an
            unrouted series, so compare against the same run left unrouted, convolved
            independently with the parameter the branch is supposed to use.
        """
        # Straight to the wrapper: `Run.runLumped` wraps `Qsim` in a date-indexed frame and
        # the unrouted series is one step longer than the index, so only the routed form
        # survives that call.
        unrouted = _lumped_model(
            coello_rrm_date,
            lumped_meteo_data_path,
            maxbas_parameters_path,
            coello_AreaCoeff,
            coello_InitialCond,
        )
        Wrapper.Lumped(unrouted, Routing=0)
        routed = _lumped_model(
            coello_rrm_date,
            lumped_meteo_data_path,
            maxbas_parameters_path,
            coello_AreaCoeff,
            coello_InitialCond,
        )

        Run.runLumped(routed, Route=1, routing_fn=Routing.triangular_routing_1)

        maxbas = routed.parameters[-1]
        expected = Routing.triangular_routing_1(
            np.array(np.asarray(unrouted.Qsim)[:-1]), maxbas
        )
        # `runLumped` wraps the routed series in a date-indexed frame; compare the values.
        actual = np.asarray(routed.Qsim, dtype=float).ravel()
        np.testing.assert_allclose(
            actual,
            np.asarray(expected, dtype=float).ravel(),
            rtol=1e-6,
            err_msg=(
                "the routed hydrograph must be the unrouted one convolved with the trailing "
                "parameter; a mismatch means the wrong branch or the wrong parameter"
            ),
        )
        assert not np.allclose(
            actual, np.asarray(unrouted.Qsim, dtype=float).ravel()[: len(actual)]
        ), "routing must actually change the hydrograph"

    def test_routing_without_a_function_is_rejected(
        self,
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        lumped_parameters_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
    ):
        """Test that asking for routing without supplying a routing function raises.

        Test scenario:
            `Route != 0` selects a routing call that would be made with `None`. Catching it
            at the entry point turns an obscure failure deep inside the wrapper into a
            message naming the argument that is missing.
        """
        model = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
        model.read_lumped_inputs(lumped_meteo_data_path)
        model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
        model.read_parameters(lumped_parameters_path, False, maxbas=False)

        with pytest.raises(ValueError, match="routing_fn"):
            Run.runLumped(model, Route=1)
