"""Base class for conceptual rainfall-runoff models.

The ``hapi.rrm.base_model`` module defines the abstract base class
:class:`BaseConceptualModel` from which all conceptual hydrological
models in the Hapi framework inherit. The class prescribes a common
interface of subroutines that every model must implement:

- **precipitation** -- partition incoming precipitation into rainfall
  and snowfall.
- **snow** -- simulate snow accumulation, melt, and refreezing.
- **soil** -- compute soil moisture changes, evapotranspiration, and
  recharge.
- **response** -- transform upper and lower zone storage into
  discharge.
- **routing** -- apply a transfer function to route discharge through
  time.
- **simulate** -- run the full model over a precipitation time series.

Concrete implementations include
:class:`~hapi.rrm.hbv.HBV`,
:class:`~hapi.rrm.hbv_bergestrom92.HBVBergestrom92`, and
:class:`~hapi.rrm.hbv_lake.HBVLake`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseConceptualModel(ABC):
    """Abstract base class for conceptual rainfall-runoff models.

    ``BaseConceptualModel`` defines the interface that every conceptual
    hydrological model in the Hapi framework must satisfy. Subclasses
    provide concrete implementations of the precipitation, snow, soil,
    response, routing, and simulate subroutines.

    The typical modelling pipeline is:

    1. Call :meth:`precipitation` to split total precipitation into
       rainfall and snowfall based on temperature thresholds.
    2. Call :meth:`snow` to update the snow pack and compute
       infiltration.
    3. Call :meth:`soil` to update soil moisture and compute recharge
       to the upper zone.
    4. Call :meth:`response` to convert zone storages into discharge
       components.
    5. Call :meth:`routing` to apply a transfer function that smooths
       the discharge hydrograph.
    6. Call :meth:`simulate` to execute the full pipeline over the
       entire time series.

    Examples:
        Subclass ``BaseConceptualModel`` to create a custom model:

        >>> from hapi.rrm.hbv import HBV
        >>> model = HBV()
        >>> rf, sf = model.precipitation(
        ...     temp=10.0, ltt=0.0, utt=2.0, prec=15.0,
        ...     rfcf=1.0, sfcf=1.0,
        ... )
        >>> print(f"rainfall={rf}, snowfall={sf}")
        rainfall=15.0, snowfall=0.0
    """

    @staticmethod
    def precipitation(
        temp: float,
        ltt: float,
        utt: float,
        rfcf: float,
        sfcf: float,
        pcorr: float = 1.0,
    ) -> tuple[float, float]:
        """Split precipitation into rainfall and snowfall.

        Partitions total precipitation into a rainfall component and a
        snowfall component based on temperature relative to a lower
        threshold (``ltt``) and an upper threshold (``utt``).
        Correction factors ``rfcf`` and ``sfcf`` are applied to
        rainfall and snowfall respectively.

        Args:
            temp (float): Measured air temperature [C].
            ltt (float): Lower temperature threshold [C]. Below this
                value all precipitation is snowfall.
            utt (float): Upper temperature threshold [C]. Above this
                value all precipitation is rainfall.
            rfcf (float): Rainfall correction factor [-].
            sfcf (float): Snowfall correction factor [-].
            pcorr (float): General precipitation correction factor
                [-]. Defaults to 1.0.

        Returns:
            tuple[float, float]: A tuple of ``(rainfall, snowfall)``
            in mm.

        Examples:
            >>> from hapi.rrm.hbv import HBV
            >>> rf, sf = HBV.precipitation(
            ...     temp=10.0, ltt=0.0, utt=2.0, prec=15.0,
            ...     rfcf=1.0, sfcf=1.0,
            ... )
            >>> print(f"rainfall={rf}, snowfall={sf}")
            rainfall=15.0, snowfall=0.0
        """
        ...

    @staticmethod
    def snow(
        temp, ttm, cfmax, cfr, cwh, rf, sf, wc_old, sp_old
    ) -> tuple[float, float, float]:
        """Simulate snow accumulation, melt, and refreezing.

        Updates the snow pack state by computing melt (when
        temperature exceeds the melting threshold ``ttm``) or
        refreezing (when temperature is below ``ttm``). Water that
        exceeds the holding capacity of the snow pack drains into the
        soil as infiltration.

        Args:
            temp (float): Air temperature [C].
            ttm (float): Temperature threshold for melting [C].
            cfmax (float): Day degree factor [mm C^-1 h^-1].
            cfr (float): Refreezing factor [-].
            cwh (float): Capacity for water holding in snow pack as
                a fraction of the snow water equivalent [-].
            rf (float): Rainfall [mm].
            sf (float): Snowfall [mm].
            wc_old (float): Water content in previous state [mm].
            sp_old (float): Snow pack in previous state [mm].

        Returns:
            tuple[float, float, float]: A tuple of
            ``(infiltration, wc_new, sp_new)`` where
            ``infiltration`` is the water draining into the soil
            [mm], ``wc_new`` is the updated liquid water content
            [mm], and ``sp_new`` is the updated snow pack [mm].

        Examples:
            >>> from hapi.rrm.hbv import HBV
            >>> inf, wc_new, sp_new = HBV.snow(
            ...     cfmax=0.1, temp=5.0, ttm=0.0, cfr=0.05,
            ...     cwh=0.1, rf=2.0, sf=0.0,
            ...     wc_old=0.0, sp_old=10.0,
            ... )
            >>> print(f"infiltration={inf:.2f}")
            infiltration=1.55
        """
        ...

    @staticmethod
    def soil(
        fc,
        beta,
        etf,
        temp,
        tm,
        e_corr,
        lp,
        tfac,
        c_flux,
        infiltration,
        ep,
        sm_old,
        uz_old,
    ) -> tuple[float, float]:
        """Compute soil moisture balance and recharge to the upper zone.

        Calculates the updated soil moisture after accounting for
        infiltration, recharge to the upper groundwater zone, actual
        evapotranspiration, and capillary rise from the upper zone.

        Args:
            fc (float): Field capacity [mm].
            beta (float): Shape coefficient for effective
                precipitation separation [-].
            etf (float): Total potential evapotranspiration factor
                [-].
            temp (float): Air temperature [C].
            tm (float): Long-term average temperature [C].
            e_corr (float): Evapotranspiration correction factor [-].
            lp (float): Wilting point as a fraction of field capacity
                [-].
            tfac (float): Time factor (e.g., 24 for daily, 1 for
                hourly) [-].
            c_flux (float): Maximum capillary flux in the root zone
                [mm].
            infiltration (float): Actual infiltration from
                precipitation and snowmelt [mm].
            ep (float): Potential evapotranspiration [mm].
            sm_old (float): Previous soil moisture value [mm].
            uz_old (float): Previous upper zone storage value [mm].

        Returns:
            tuple[float, float]: A tuple of ``(sm_new, uz_int_1)``
            where ``sm_new`` is the new soil moisture [mm] and
            ``uz_int_1`` is the new direct runoff into the upper
            zone [mm].

        Examples:
            >>> from hapi.rrm.hbv import HBV
            >>> sm_new, uz_int_1 = HBV.soil(
            ...     fc=200.0, beta=2.0, etf=0.1, temp=20.0,
            ...     tm=18.0, e_corr=1.0, lp=0.3, c_flux=0.01,
            ...     inf=5.0, ep=3.0, sm_old=100.0, uz_old=10.0,
            ... )
            >>> print(f"sm_new={sm_new:.2f}, uz_int_1={uz_int_1:.2f}")
            sm_new=101.13, uz_int_1=11.25
        """
        ...

    @staticmethod
    def response(
        tfac, perc, alpha, k, k1, area, lz_old, uz_int_1
    ) -> tuple[float, float, float]:
        """Convert upper and lower zone storage into stream discharge.

        Transforms the current values of the upper and lower storage
        zones into discharge components using recession coefficients.
        Also controls the recharge of the lower zone tank (baseflow)
        via percolation from the upper zone.

        Args:
            tfac (float): Time conversion factor (e.g., 24 for daily,
                1 for hourly) [-].
            perc (float): Percolation rate from the upper zone to the
                lower zone [mm/h].
            alpha (float): Response box parameter controlling the
                non-linearity of the upper zone outflow [-].
            k (float): Upper zone recession coefficient [h^-1].
            k1 (float): Lower zone recession coefficient [h^-1].
            area (float): Catchment area [km^2].
            lz_old (float): Previous lower zone storage [mm].
            uz_int_1 (float): Previous upper zone storage before
                percolation [mm].

        Returns:
            tuple[float, float, float]: A tuple of
            ``(q_new, uz_new, lz_new)`` where ``q_new`` is the total
            discharge [m^3/s], ``uz_new`` is the updated upper zone
            storage [mm], and ``lz_new`` is the updated lower zone
            storage [mm].

        Examples:
            >>> from hapi.rrm.hbv import HBV
            >>> q_0, q_1, uz_new, lz_new = HBV.response(
            ...     perc=0.5, alpha=0.5, k=0.01, k1=0.001,
            ...     lz_old=20.0, uz_int_1=15.0,
            ... )
            >>> print(f"q_upper={q_0:.4f}, q_lower={q_1:.4f}")
            q_upper=0.0381, q_lower=0.0205
        """
        ...

    def routing(self, q: np.ndarray, maxbas: int = 1) -> np.ndarray:
        """Apply a triangular transfer function to route discharge.

        Smooths the discharge hydrograph by distributing the signal
        over ``maxbas`` time steps using a triangular weighting
        function.

        Args:
            q (numpy.ndarray): Discharge time series [mm].
            maxbas (int): Number of time steps for the transfer
                function base. Must be >= 1. Defaults to 1.

        Returns:
            numpy.ndarray: Routed discharge time series with the same
            length as ``q``.

        Raises:
            AssertionError: If ``maxbas`` is less than 1.

        Examples:
            >>> import numpy as np
            >>> from hapi.rrm.hbv import HBV
            >>> model = HBV()
            >>> q = np.array([0.0, 0.0, 5.0, 3.0, 1.0, 0.0])
            >>> q_routed = model.routing(q, maxbas=3)
            >>> print(q_routed.round(4))
            [0.   0.   2.5  4.   2.5  0.5]
        """
        ...

    @abstractmethod
    def simulate(
        self, prec, temp, et, ll_temp, par, init_st=None, q_init=None, snow=0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model simulation over a full precipitation time series.

        Executes all model subroutines sequentially for each time step
        in the input arrays, producing discharge and state variable
        time series.

        Subclasses must implement this method with the specific model
        logic (parameter parsing, subroutine calls, state updates).

        Args:
            prec (numpy.ndarray): Average precipitation time series
                of length ``n`` [mm/h].
            temp (numpy.ndarray): Average temperature time series of
                length ``n`` [C].
            et (numpy.ndarray): Potential evapotranspiration time
                series of length ``n`` [mm/h].
            ll_temp (numpy.ndarray): Long-term average temperature
                time series of length ``n`` [C].
            par (numpy.ndarray): Parameter vector. The length depends
                on the specific model implementation and whether
                snow processes are active.
            init_st (list[float]): Initial model states as
                ``[sp, sm, uz, lz, wc]`` [mm]. If None, the model
                uses default initial values.
            q_init (float): Initial discharge value. If None, the
                model computes it from the initial states and
                parameters.
            snow (int): Set to 1 to run the snow subroutine, 0 to
                skip it. Defaults to 0.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            A tuple of ``(q_uz, q_lz, states)`` where:

            - ``q_uz``: Upper zone discharge array of length
              ``n+1``.
            - ``q_lz``: Lower zone discharge array of length
              ``n+1``.
            - ``states``: Model states array of shape ``(n+1, 5)``.

        Examples:
            >>> import numpy as np
            >>> from hapi.rrm.hbv import HBV
            >>> model = HBV()
            >>> par = np.array([
            ...     1.0, 200.0, 2.0, 0.1, 0.3, 0.01,
            ...     0.005, 0.0005, 0.5, 0.5,
            ... ])
            >>> n = 5
            >>> prec = np.array([10.0, 8.0, 12.0, 6.0, 4.0])
            >>> temp_arr = np.full(n, 25.0)
            >>> et = np.full(n, 3.0)
            >>> init_st = [0.0, 100.0, 10.0, 15.0, 0.0]
            >>> q_uz, q_lz, states = model.simulate(
            ...     prec, temp_arr, et, par,
            ...     init_st=init_st, snow=0,
            ... )
            >>> print(f"q_uz length={len(q_uz)}")
            q_uz length=6
        """
        ...
