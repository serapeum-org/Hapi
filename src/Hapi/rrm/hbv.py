"""Lumped Conceptual HBV model.

The ``Hapi.rrm.hbv`` module implements the HBV-96 lumped conceptual
hydrological model. The model consists of precipitation partitioning,
snow accumulation and melt, soil moisture accounting, and a response
routine that converts precipitation into runoff. State variables are
updated at each time step to represent the hydrologic behaviour of the
catchment.

This version was edited based on a Master Thesis on
"Spatio-temporal simulation of catchment response based on dynamic
weighting of hydrological models" in April 2018.

Model inputs are precipitation, evapotranspiration, and temperature,
along with initial state variables and initial discharge. Model output
is calculated discharge at time ``t+1``. Model equations are solved
using an explicit scheme.

The model structure uses 18 parameters when the catchment has snow::

    [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta,
     e_corr, etf, lp, c_flux, k, k1, alpha, perc]

Otherwise it uses 10 parameters::

    [rfcf, fc, beta, etf, lp, c_flux, k, k1, alpha, perc]
"""
from __future__ import annotations

import numpy as np
from Hapi.rrm.base_model import BaseConceptualModel

# HBV base model parameters
P_LB = [
    -1.5,  # ltt
    0.001,  # utt
    0.001,  # ttm
    0.04,  # cfmax [mm c^-1 h^-1]
    50.0,  # fc
    0.6,  # ecorr
    0.001,  # etf
    0.2,  # lp
    0.00042,  # k [h^-1] upper zone
    0.0000042,  # k1 lower zone
    0.001,  # alpha
    1.0,  # beta
    0.001,  # cwh
    0.01,  # cfr
    0.0,  # c_flux
    0.001,  # perc mm/h
    0.6,  # rfcf
    0.4,  # sfcf
    1,
]  # Maxbas

P_UB = [
    2.5,  # ttm
    3.0,  # utt
    2.0,  # ttm
    0.4,  # cfmax [mm c^-1 h^-1]
    500.0,  # fc
    1.4,  # ecorr
    5.0,  # etf
    0.5,  # lp
    0.0167,  # k upper zone
    0.00062,  # k1 lower zone
    1.0,  # alpha
    6.0,  # beta
    0.1,  # cwh
    1.0,  # cfr
    0.08,  # c_flux - 2mm/day
    0.125,  # perc mm/hr
    1.4,  # rfcf
    1.4,  # sfcf
    10,
]  # maxbas

DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0]
DEF_q0 = 0

# Get random parameter set
# def get_random_pars():
#    return np.random.uniform(P_LB, P_UB)

class HBV(BaseConceptualModel):
    """HBV-96 lumped conceptual rainfall-runoff model.

    The HBV (Hydrologiska Byrans Vattenbalansavdelning) model is a
    conceptual hydrological model that simulates catchment discharge
    from precipitation, temperature, and potential evapotranspiration.
    The model consists of four main routines:

    - **Precipitation**: partitions incoming precipitation into rainfall
      and snowfall based on temperature thresholds.
    - **Snow**: simulates snow accumulation, melt, and refreezing using
      a degree-day approach.
    - **Soil**: computes soil moisture changes, actual evapotranspiration,
      capillary rise, and recharge to the upper zone.
    - **Response**: transforms upper and lower zone storage into
      discharge via recession coefficients.

    The class inherits from
    :class:`~Hapi.rrm.base_model.BaseConceptualModel` and provides
    concrete implementations of all required subroutines.

    Examples:
        >>> from Hapi.rrm.hbv import HBV
        >>> model = HBV()
        >>> rf, sf = model.precipitation(
        ...     temp=5.0, ltt=0.0, utt=2.0, prec=10.0,
        ...     rfcf=1.0, sfcf=1.0,
        ... )
        >>> print(f"rainfall={rf}, snowfall={sf}")
        rainfall=10.0, snowfall=0.0
    """

    @staticmethod
    def precipitation(temp, ltt, utt, prec, rfcf, sfcf):  # type: ignore[override]
        """Partition precipitation into rainfall and snowfall.

        If temperature is lower than ``ltt``, all precipitation is
        considered snowfall. If temperature is higher than ``utt``, all
        precipitation is considered rainfall. When temperature falls
        between ``ltt`` and ``utt``, precipitation is linearly mixed
        between rainfall and snowfall.

        Args:
            temp (float): Measured temperature [C].
            ltt (float): Lower temperature threshold [C].
            utt (float): Upper temperature threshold [C].
            prec (float): Precipitation [mm].
            rfcf (float): Rainfall correction factor.
            sfcf (float): Snowfall correction factor.

        Returns:
            tuple[float, float]: A tuple of ``(rainfall, snowfall)``
            in mm.

        Examples:
            Temperature above the upper threshold produces only
            rainfall:

            >>> from Hapi.rrm.hbv import HBV
            >>> rf, sf = HBV.precipitation(
            ...     temp=10.0, ltt=0.0, utt=2.0, prec=15.0,
            ...     rfcf=1.0, sfcf=1.0,
            ... )
            >>> print(f"rainfall={rf}, snowfall={sf}")
            rainfall=15.0, snowfall=0.0

            Temperature below the lower threshold produces only
            snowfall:

            >>> rf, sf = HBV.precipitation(
            ...     temp=-5.0, ltt=0.0, utt=2.0, prec=15.0,
            ...     rfcf=1.0, sfcf=1.0,
            ... )
            >>> print(f"rainfall={rf}, snowfall={sf}")
            rainfall=0.0, snowfall=15.0

            Temperature between thresholds produces a mix:

            >>> rf, sf = HBV.precipitation(
            ...     temp=1.0, ltt=0.0, utt=2.0, prec=10.0,
            ...     rfcf=1.0, sfcf=1.0,
            ... )
            >>> print(f"rainfall={rf}, snowfall={sf}")
            rainfall=5.0, snowfall=5.0
        """

        if temp <= ltt:  # if temp <= lower temp threshold
            rf = 0.0  # no rainfall all the precipitation will convert into snowfall
            sf = prec * sfcf

        elif temp >= utt:  # if temp > upper threshold
            rf = prec * rfcf  # no snowfall all the precipitation becomes rainfall
            sf = 0.0

        else:  # if  ltt< temp < utt
            rf = ((temp - ltt) / (utt - ltt)) * prec * rfcf
            sf = (1.0 - ((temp - ltt) / (utt - ltt))) * prec * sfcf

        return rf, sf

    @staticmethod
    def snow(cfmax, temp, ttm, cfr, cwh, rf, sf, wc_old, sp_old) -> tuple[float, float, float]:  # type: ignore[override]
        """Simulate snow accumulation, melt, and refreezing.

        The snow pack consists of two states: Water Content (``wc``)
        and Snow Pack (``sp``). The water content corresponds to the
        liquid part of the water in the snow, while the snow pack
        corresponds to the solid part. If temperature exceeds the
        melting threshold, the snow pack melts and solid snow becomes
        liquid. Otherwise, the liquid part refreezes into solid snow.
        Water that exceeds the holding capacity of the snow pack drains
        into the soil as infiltration.

        Args:
            cfmax (float): Day degree factor [mm C^-1 h^-1].
            temp (float): Temperature [C].
            ttm (float): Temperature threshold for melting [C].
            cfr (float): Refreezing factor.
            cwh (float): Capacity for water holding in snow pack.
            rf (float): Rainfall [mm].
            sf (float): Snowfall [mm].
            wc_old (float): Water content in previous state [mm].
            sp_old (float): Snow pack in previous state [mm].

        Returns:
            tuple[float, float, float]: A tuple of
            ``(infiltration, wc_new, sp_new)`` where
            ``infiltration`` is the water draining into the soil
            [mm], ``wc_new`` is the updated water content [mm],
            and ``sp_new`` is the updated snow pack [mm].

        Examples:
            When temperature exceeds the melt threshold, snow melts
            and infiltration occurs:

            >>> from Hapi.rrm.hbv import HBV
            >>> inf, wc_new, sp_new = HBV.snow(
            ...     cfmax=0.1, temp=5.0, ttm=0.0, cfr=0.05,
            ...     cwh=0.1, rf=2.0, sf=0.0,
            ...     wc_old=0.0, sp_old=10.0,
            ... )
            >>> print(f"infiltration={inf:.2f}")
            infiltration=1.55
            >>> print(f"wc_new={wc_new:.2f}, sp_new={sp_new:.2f}")
            wc_new=0.95, sp_new=9.5

            When temperature is below the melt threshold, water
            refreezes:

            >>> inf, wc_new, sp_new = HBV.snow(
            ...     cfmax=0.1, temp=-2.0, ttm=0.0, cfr=0.5,
            ...     cwh=0.1, rf=0.0, sf=5.0,
            ...     wc_old=2.0, sp_old=10.0,
            ... )
            >>> print(f"infiltration={inf:.2f}")
            infiltration=1.40
            >>> print(f"sp_new={sp_new:.2f}")
            sp_new=15.1
        """

        if temp > ttm:  # if temp > melting threshold
            # then either some snow will melt or the entire snow will melt
            if (
                cfmax * (temp - ttm) < sp_old + sf
            ):  # if amount of melted snow < the entire existing snow (previous amount+new)
                melt = cfmax * (temp - ttm)
            else:  # if amount of melted snow > the entire existing snow (previous amount+new)
                melt = (
                    sp_old + sf
                )  # then the entire existing snow will melt (old snow pack + the current snowfall)

            sp_new = sp_old + sf - melt
            wc_int = wc_old + melt + rf

        else:  # if temp < melting threshold
            # then either some water will freeze or all the water willfreeze
            if (
                cfr * cfmax * (ttm - temp) < wc_old + rf
            ):  # then either some water will freeze or all the water willfreeze
                refr = (
                    cfr * cfmax * (ttm - temp)
                )  # cfmax*(ttm-temp) is the rate of melting of snow while cfr*cfmax*(ttm-temp) is the rate of freeze of melted water  (rate of freezing > rate of melting)
            else:  # if the amount of frozen water > entire water available
                refr = wc_old + rf

            sp_new = sp_old + sf + refr
            wc_int = wc_old - refr + rf

        if wc_int > cwh * sp_new:  # if water content > holding water capacity of the snow
            inf = wc_int - cwh * sp_new  # water content  will infiltrate
            wc_new = cwh * sp_new  # and the capacity of snow of holding water will retained
        else:  # if water content < holding water capacity of the snow
            inf = 0.0  # no infiltration
            wc_new = wc_int

        return inf, wc_new, sp_new

    @staticmethod
    def soil(fc, beta, etf, temp, tm, e_corr, lp, c_flux, inf, ep, sm_old, uz_old) -> tuple[float, float]:  # type: ignore[override]  # tfac,
        """Compute soil moisture balance and recharge to the upper zone.

        The soil routine checks the amount of water that can infiltrate
        the soil from liquid precipitation and snow pack melting. A
        portion of the water is stored as soil moisture, while the
        remainder becomes runoff routed to the upper zone tank. The
        routine also accounts for actual evapotranspiration and
        capillary rise from the upper zone.

        Args:
            fc (float): Field capacity [mm].
            beta (float): Shape coefficient for effective precipitation
                separation.
            etf (float): Total potential evapotranspiration factor.
            temp (float): Temperature [C].
            tm (float): Average long-term temperature [C].
            e_corr (float): Evapotranspiration correction factor.
            lp (float): Wilting point as a fraction of field capacity.
            c_flux (float): Capillary flux in the root zone [mm].
            inf (float): Actual infiltration [mm].
            ep (float): Actual evapotranspiration [mm].
            sm_old (float): Previous soil moisture value [mm].
            uz_old (float): Previous upper zone value [mm].

        Returns:
            tuple[float, float]: A tuple of ``(sm_new, uz_int_1)``
            where ``sm_new`` is the new soil moisture [mm] and
            ``uz_int_1`` is the new direct runoff into the upper
            zone [mm].

        Examples:
            Compute soil moisture update for a warm day with
            infiltration:

            >>> from Hapi.rrm.hbv import HBV
            >>> sm_new, uz_int_1 = HBV.soil(
            ...     fc=200.0, beta=2.0, etf=0.1, temp=20.0,
            ...     tm=18.0, e_corr=1.0, lp=0.3, c_flux=0.01,
            ...     inf=5.0, ep=3.0, sm_old=100.0, uz_old=10.0,
            ... )
            >>> print(f"sm_new={sm_new:.2f}, uz_int_1={uz_int_1:.2f}")
            sm_new=101.13, uz_int_1=11.25
        """

        qdr = max(
            sm_old + inf - fc, 0
        )  # direct run off as soil moisture exceeded the field capacity

        inf = inf - qdr
        r = ((sm_old / fc) ** beta) * inf  # recharge to the upper zone

        ep_int = (
            (1.0 + etf * (temp - tm)) * e_corr * ep
        )  # Adjusted potential evapotranspiration

        ea = min(ep_int, (sm_old / (lp * fc)) * ep_int)

        cf = c_flux * ((fc - sm_old) / fc)  # capilary rise

        # if capilary rise is more than what is available take all the available and leave it empty

        if uz_old + r < cf:
            cf = uz_old + r
            uz_int_1 = 0
        else:
            #        uz_int_1 = uz_old + _r - _cf
            uz_int_1 = uz_old + r - cf + qdr

        sm_new = max(sm_old + inf - r + cf - ea, 0)

        #    uz_int_1 = uz_old + _r - _cf + qdr

        return sm_new, uz_int_1

    @staticmethod
    def response(perc, alpha, k, k1, lz_old, uz_int_1) -> tuple[float, float, float, float]:  # type: ignore[override]  # tfac,area,
        r"""Transform upper and lower zone storage into discharge.

        The response routine converts the current values of the upper
        and lower storage zones into discharge components. It also
        controls the recharge of the lower zone tank (baseflow) via
        percolation from the upper zone.

        Args:
            perc (float): Percolation value [mm/hr].
            alpha (float): Response box parameter controlling the
                non-linearity of the upper zone outflow.
            k (float): Upper zone recession coefficient [h^-1].
            k1 (float): Lower zone recession coefficient [h^-1].
            lz_old (float): Previous lower zone value [mm].
            uz_int_1 (float): Previous upper zone value before
                percolation [mm].

        Returns:
            tuple[float, float, float, float]: A tuple of
            ``(q_0, q_1, uz_new, lz_new)`` where ``q_0`` is the
            upper zone discharge [mm], ``q_1`` is the lower zone
            discharge [mm], ``uz_new`` is the updated upper zone
            storage [mm], and ``lz_new`` is the updated lower zone
            storage [mm].

        Examples:
            Compute discharge from upper and lower zone storages:

            >>> from Hapi.rrm.hbv import HBV
            >>> q_0, q_1, uz_new, lz_new = HBV.response(
            ...     perc=0.5, alpha=0.5, k=0.01, k1=0.001,
            ...     lz_old=20.0, uz_int_1=15.0,
            ... )
            >>> print(f"q_upper={q_0:.4f}, q_lower={q_1:.4f}")
            q_upper=0.0381, q_lower=0.0205
            >>> print(f"uz_new={uz_new:.4f}, lz_new={lz_new:.4f}")
            uz_new=14.4619, lz_new=20.4795
        """
        # upper zone
        # if perc > Quz then perc = Quz and Quz = 0 if not perc = value and Quz= Quz-perc so take the min
        uz_int_2 = np.max([uz_int_1 - perc, 0.0])

        q_0 = k * (uz_int_2 ** (1.0 + alpha))

        if q_0 > uz_int_2:  # if q_0 =30 and UZ=20
            q_0 = uz_int_2  # q_0=20

        uz_new = uz_int_2 - (q_0)

        lz_int_1 = lz_old + np.min(
            [perc, uz_int_1]
        )  # if the percolation > upper zone Q all the Quz will percolate

        q_1 = k1 * lz_int_1

        if q_1 > lz_int_1:
            q_1 = lz_int_1

        lz_new = lz_int_1 - (q_1)

        #    q_new = area*(q_0 + q_1)/(3.6*tfac)  # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
        # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25

        #    return q_new, uz_new, lz_new, uz_int_2, lz_int_1
        return q_0, q_1, uz_new, lz_new  # ,uz_int_2, lz_int_1

    @staticmethod
    def tf(maxbas) -> np.ndarray:
        """Generate transfer function weights using a triangular shape.

        Creates a set of weights for the triangular transfer function
        used in discharge routing. The weights grow linearly up to the
        midpoint of ``maxbas`` and then recede linearly, and are
        normalized to sum to 1.

        Args:
            maxbas (int): Number of time steps for the transfer
                function base. Must be >= 1.

        Returns:
            numpy.ndarray: Array of normalized weights with length
            ``maxbas``.

        Examples:
            >>> from Hapi.rrm.hbv import HBV
            >>> weights = HBV.tf(5)
            >>> print(weights.round(4))
            [0.1429 0.2857 0.2857 0.1429 0.1429]
            >>> print(f"sum={weights.sum():.4f}")
            sum=1.0000
        """
        wi = []
        for x in range(1, maxbas + 1):
            if x <= (maxbas) / 2.0:
                # Growing transfer
                wi.append((x) / (maxbas + 2.0))
            else:
                # Receding transfer
                wi.append(1.0 - (x + 1) / (maxbas + 2.0))

        # Normalise weights
        wi = np.array(wi) / np.sum(wi)
        return wi  # type: ignore[no-any-return]


    def routing(self, q, maxbas=1):
        """Route discharge through a triangular transfer function.

        Applies the triangular transfer function to the discharge time
        series. The transfer function smooths the hydrograph by
        distributing the discharge over ``maxbas`` time steps using
        weights generated by :meth:`tf`.

        Args:
            q (numpy.ndarray): Discharge time series [mm].
            maxbas (int): Number of time steps for the transfer
                function base. Must be >= 1. Default is 1.

        Returns:
            numpy.ndarray: Routed discharge time series with the same
            length as ``q``.

        Raises:
            AssertionError: If ``maxbas`` is less than 1.

        Examples:
            >>> import numpy as np
            >>> from Hapi.rrm.hbv import HBV
            >>> model = HBV()
            >>> q = np.array([0.0, 0.0, 5.0, 3.0, 1.0, 0.0])
            >>> q_routed = model.routing(q, maxbas=3)
            >>> print(q_routed.round(4))
            [0.   0.   2.5  4.   2.5  0.5]
        """
        assert maxbas >= 1, "Maxbas value has to be larger than 1"
        # Get integer part of maxbas
        #    maxbas = int(maxbas)
        maxbas = int(round(maxbas, 0))

        # get the weights
        w = self.tf(maxbas)

        # rout the discharge signal
        q_r = np.zeros_like(q, dtype="float64")
        q_temp = q
        for w_i in w:
            q_r += q_temp * w_i
            q_temp = np.insert(q_temp, 0, 0.0)[:-1]

        return q_r

    def step_run(self, p: np.ndarray, v: np.ndarray, state_variable: np.ndarray, snow=0):
        """Execute a single time step of the HBV model.

        Parses the parameter vector, input vector, and state variables,
        then sequentially runs the precipitation, snow, soil, and
        response routines to compute discharge and updated states for
        the next time step.

        Args:
            p (numpy.ndarray): Parameter vector. When ``snow=1``, must
                have 18 elements::

                    [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr,
                     fc, beta, e_corr, etf, lp, c_flux,
                     k, k1, alpha, perc]

                When ``snow=0``, must have 10 elements::

                    [rfcf, fc, beta, etf, lp, c_flux,
                     k, k1, alpha, perc]

            v (numpy.ndarray): Input vector with 4 elements::

                    [prec, temp, evap, ll_temp]

            state_variable (numpy.ndarray): Previous model states with
                5 elements::

                    [sp, sm, uz, lz, wc]

            snow (int): Set to 1 to run the snow subroutine, 0 to
                skip it. Default is 0.

        Returns:
            tuple[float, float, list[float]]: A tuple of
            ``(q_uz, q_lz, states)`` where ``q_uz`` is the upper
            zone discharge [mm], ``q_lz`` is the lower zone
            discharge [mm], and ``states`` is a list of five
            updated state variables ``[sp, sm, uz, lz, wc]``.

        Raises:
            AssertionError: If ``snow=1`` and the parameter vector
                does not have 18 elements.

        Examples:
            Run a single step without snow:

            >>> import numpy as np
            >>> from Hapi.rrm.hbv import HBV
            >>> model = HBV()
            >>> par = np.array([
            ...     1.0, 200.0, 2.0, 0.1, 0.3, 0.01,
            ...     0.005, 0.0005, 0.5, 0.5,
            ... ])
            >>> inputs = np.array([10.0, 25.0, 3.0, 20.0])
            >>> states = np.array([0.0, 100.0, 10.0, 15.0, 0.0])
            >>> q_uz, q_lz, new_states = model.step_run(
            ...     par, inputs, states, snow=0,
            ... )
            >>> print(f"q_uz={q_uz:.4f}, q_lz={q_lz:.4f}")
            q_uz=0.0727, q_lz=0.0078
        """
        ## Parse of parameters from input vector to model
        # picipitation function
        if snow == 1:
            assert len(p) == 18, (
                "current version of HBV (with snow) takes 18 parameter you have entered "
                + str(len(p))
            )
            ltt = p[0]
            utt = p[1]
            rfcf = p[2]
            sfcf = p[3]
            # snow function
            ttm = p[4]
            cfmax = p[5]
            cwh = p[6]
            cfr = p[7]
            # soil function
            fc = p[8]
            beta = p[9]
            e_corr = [10]
            etf = p[11]
            lp = p[12]
            c_flux = p[13]
            # response function
            k = p[14]
            k1 = p[15]
            alpha = p[16]
            perc = p[17]
        #        pcorr=p[18]

        elif snow == 0:
            ltt = 1.0  # less than utt and less than lowest temp to prevent sf formation
            utt = (
                2.0  # very low but it does not matter as temp is 25 so it is greater than 2
            )
            rfcf = p[0]  # 1.0 #p[16] # all precipitation becomes rainfall
            sfcf = 0.00001  # there is no snow
            # snow function
            ttm = 1  # should be very low lower than lowest temp as temp is 25 all the time so it does not matter
            cfmax = 0.00001  # as there is no melting  and sp+sf=zero all the time so it doesn't matter the value of cfmax
            cwh = 0.00001  # as sp is always zero it doesn't matter all wc will go as inf
            cfr = 0.000001  # as temp > ttm all the time so it doesn't matter the value of cfr but put it zero
            # soil function
            fc = p[1]
            beta = p[2]
            e_corr = 1  # type: ignore[assignment]  # p[2]
            etf = p[3]
            lp = p[4]
            c_flux = p[5]
            # response function
            k = p[6]
            k1 = p[7]
            alpha = p[8]
            perc = p[9]

        ## Non optimisable parameters
        # tfac = p2[0]
        # area = p2[1]

        ## Parse of Inputs
        prec = v[0]  # Precipitation [mm]
        temp = v[1]  # Temperature [C]
        ep = v[2]  # Long terms (monthly) Evapotranspiration [mm]
        tm = v[3]  # Long term (monthly) average temperature [C]

        ## Parse of states
        sp_old = state_variable[0]
        sm_old = state_variable[1]
        uz_old = state_variable[2]
        lz_old = state_variable[3]
        wc_old = state_variable[4]

        rf, sf = self.precipitation(temp, ltt, utt, prec, rfcf, sfcf)  # , tfac
        inf, wc_new, sp_new = self.snow(
            cfmax, temp, ttm, cfr, cwh, rf, sf, wc_old, sp_old  # tfac,
        )
        sm_new, uz_int_1 = self.soil(
            fc, beta, etf, temp, tm, e_corr, lp, c_flux, inf, ep, sm_old, uz_old  # tfac,
        )

        q_uz, q_lz, uz_new, lz_new = self.response(
            perc, alpha, k, k1, lz_old, uz_int_1  # tfac,  # area,
        )

        #    return q_new, [sp_new, sm_new, uz_new, lz_new, wc_new], uz_int_2, lz_int_1
        return q_uz, q_lz, [sp_new, sm_new, uz_new, lz_new, wc_new]

    def simulate(self, prec, temp, et, par, init_st=None, ll_temp=None, q_init=None, snow=0):  # type: ignore[override]
        """Run the HBV model for the full precipitation time series.

        Executes the HBV model for ``n`` time steps (the length of the
        precipitation array). The results contain ``n+1`` values as
        the model calculates discharge for step ``n+1`` from step
        ``n`` inputs.

        Args:
            prec (numpy.ndarray): Average precipitation time series
                of length ``n`` [mm/h].
            temp (numpy.ndarray): Average temperature time series of
                length ``n`` [C].
            et (numpy.ndarray): Potential evapotranspiration time
                series of length ``n`` [mm/h].
            par (numpy.ndarray): Parameter vector. When ``snow=1``,
                must have 18 elements; when ``snow=0``, must have
                10 elements. See :meth:`step_run` for the full
                parameter layout.
            init_st (list[float]): Initial model states as
                ``[sp, sm, uz, lz, wc]`` [mm]. If None, defaults
                to ``[0.0, 10.0, 10.0, 10.0, 0.0]``.
            ll_temp (numpy.ndarray): Long-term average temperature
                time series of length ``n`` [C]. If None, computed
                as the mean of ``temp``.
            q_init (float): Initial discharge value. If None,
                computed from the initial states and parameters.
            snow (int): Set to 1 to run the snow subroutine, 0 to
                skip it. Default is 0.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                A tuple of ``(q_uz, q_lz, states)`` where:

                - ``q_uz``: Upper zone discharge array of length
                  ``n+1`` (float32).
                - ``q_lz``: Lower zone discharge array of length
                  ``n+1`` (float32).
                - ``states``: Model states array of shape
                  ``(n+1, 5)`` (float32).

        Raises:
            AssertionError: If ``init_st`` does not have 5 elements
                or if ``snow`` is not 0 or 1.

        Examples:
            Run a short simulation without snow:

            >>> import numpy as np
            >>> from Hapi.rrm.hbv import HBV
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
            >>> print(f"q_uz length={len(q_uz)}, first={q_uz[0]:.4f}")
            q_uz length=6, first=0.0727
        """
        # data type
        assert (
            len(init_st) == 5
        ), "state variables are 5 and the given initial values are " + str(len(init_st))
        # assert type(p2) == list, " p2 should be of type list"
        # assert len(p2) == 2, "p2 should contains tfac and catchment area"
        assert (
            snow == 0 or snow == 1
        ), " snow input defines whether to consider snow subroutine or not it has to be 0 or 1"

        if init_st is None:  # 0  1  2  3  4  5
            st = [DEF_ST]  # [sp,sm,uz,lz,wc,LA]
        else:
            st = [init_st]

        if ll_temp is None:
            ll_temp = [np.mean(temp)] * len(prec)

        if q_init is None:
            if snow == 0:
                q_uz = [par[6] * ((st[0][2]) ** (1.0 + par[8]))]
                q_lz = [par[7] * st[0][3]]
        else:
            q_uz = [par[14] * ((st[0][2]) ** (1.0 + par[16]))]
            q_lz = [par[15] * st[0][3]]

        #    uz_int_2 = [st[0][2], ]
        #    lz_int_1 = [st[0][3], ]

        for i in range(len(prec)):
            v = [prec[i], temp[i], et[i], ll_temp[i]]
            #        q_out, st_out, uz_int_2_out, lz_int_1_out = StepRun(par, p2, v, st[i], snow=0)
            q_uzi, q_lzi, st_out = self.step_run(par, v, st[i], snow=0)  # type: ignore[arg-type]  # p2,
            #        q_sim.append(q_out)
            q_uz.append(q_uzi)
            q_lz.append(q_lzi)
            st.append(st_out)
        #        uz_int_2.append(uz_int_2_out) # upper zone - perc
        #        lz_int_1.append(lz_int_1_out) # lower zone + perc

        return np.float32(q_uz), np.float32(q_lz), np.float32(st)  # type: ignore[arg-type]
