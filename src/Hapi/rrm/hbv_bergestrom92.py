"""HBV Bergestrom 1992 Lumped Conceptual Hydrological Model.

The ``Hapi.rrm.hbv_bergestrom92`` module implements the HBV-96 lumped
conceptual model based on Bergstrom (1992), using two reservoirs with
three linear responses: surface runoff, interflow, and baseflow.

The HBV model consists of precipitation, snow melt, soil moisture, and
response subroutines that convert precipitation into runoff. State
variables are updated each time step to represent specific hydrologic
behaviour of the catchment.

This version was edited based on a Master Thesis on
"Spatio-temporal simulation of catchment response based on dynamic
weighting of hydrological models" in April 2018.

Model characteristics:
    - Inputs: precipitation, evapotranspiration, temperature, initial
      state variables, and initial discharge.
    - Output: calculated discharge at time t+1.
    - Equations are solved using an explicit scheme.
    - Uses 15 parameters if the catchment has snow:
      ``[tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta, e_corr, lp,
      k, k1, k2, uzl, perc]``
    - Uses 10 parameters otherwise:
      ``[rfcf, fc, beta, e_corr, lp, k, k1, k2, uzl, perc]``
"""

from typing import Tuple
import numpy as np

from Hapi.rrm.base_model import BaseConceptualModel

DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0]
DEF_q0 = 0


class HBVBergestrom92(BaseConceptualModel):
    """HBV Bergestrom 1992 lumped conceptual hydrological model.

    This class implements the HBV-96 model variant based on
    Bergstrom (1992), featuring two groundwater reservoirs (upper and
    lower zones) with three linear outflow equations for surface
    runoff, interflow, and baseflow.

    The model inherits from
    :class:`~Hapi.rrm.base_model.BaseConceptualModel` and implements
    the ``precipitation``, ``snow``, ``soil``, ``response``,
    ``routing``, and ``simulate`` methods.

    Examples:
        >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
        >>> model = HBVBergestrom92()
    """

    def __init__(self):
        """Initialize the HBVBergestrom92 model."""
        pass

    @staticmethod
    def precipitation(prec, temp, tt, rfcf, sfcf):
        """Partition precipitation into rainfall and snowfall.

        If the temperature is lower than or equal to the threshold
        ``tt``, all precipitation is considered snowfall. If the
        temperature is higher than ``tt``, all precipitation is
        considered rainfall. Correction factors are applied to each
        component.

        Args:
            prec (float): Precipitation [mm].
            temp (float): Measured temperature [C].
            tt (float): Lower temperature threshold [C].
            rfcf (float): Rainfall correction factor [-].
            sfcf (float): Snowfall correction factor [-].

        Returns:
            tuple[float, float]: A tuple of ``(rf, sf)`` where:
                - **rf** (*float*): Rainfall [mm].
                - **sf** (*float*): Snowfall [mm].

        Examples:
            >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
            >>> rf, sf = HBVBergestrom92.precipitation(
            ...     prec=10.0, temp=-2.0, tt=0.0, rfcf=1.0, sfcf=0.8
            ... )
            >>> rf
            0.0
            >>> sf
            8.0

            When temperature exceeds the threshold, all precipitation
            becomes rainfall:

            >>> rf, sf = HBVBergestrom92.precipitation(
            ...     prec=10.0, temp=5.0, tt=0.0, rfcf=1.0, sfcf=0.8
            ... )
            >>> rf
            10.0
            >>> sf
            0.0
        """
        # if temp <= lower temp threshold
        if temp <= tt:
            # no rainfall all the precipitation will convert into snowfall
            rf = 0.0
            sf = prec * sfcf
        else:
            # temp >= tt: # if temp > upper threshold
            # no snowfall all the precipitation becomes rainfall
            rf = prec * rfcf
            sf = 0.0

        return rf, sf

    @staticmethod
    def snow(temp, rf, sf, wc_old, sp_old, tt, cfmax, cfr, cwh):
        """Compute snow accumulation, melt, and infiltration.

        The snow pack consists of two states: water content (``wc``)
        and snow pack (``sp``). The water content corresponds to the
        liquid part of the water in the snow, while the snow pack
        corresponds to the solid part.

        If the temperature is higher than the melting point, the snow
        pack will melt and the solid snow will become liquid. In the
        opposite case, the liquid part of the snow will refreeze and
        turn into solid. The water that cannot be stored by the solid
        part of the snow pack will drain into the soil as
        infiltration.

        Snowmelt is calculated with the degree-day method using
        ``cfmax``. Meltwater and rainfall are retained within the
        snowpack until they exceed the fraction ``cwh`` of the water
        equivalent of the snow. Liquid water within the snowpack
        refreezes using ``cfr``.

        Args:
            temp (float): Temperature [C].
            rf (float): Rainfall [mm].
            sf (float): Snowfall [mm].
            wc_old (float): Water content in previous state [mm].
            sp_old (float): Snow pack in previous state [mm].
            tt (float): Temperature threshold for melting [C].
            cfmax (float): Day degree factor [mm/C/timestep].
            cfr (float): Refreezing factor [-].
            cwh (float): Capacity for water holding in snow pack
                as a fraction [-].

        Returns:
            tuple[float, float, float]: A tuple of
                ``(inf, wc_new, sp_new)`` where:
                - **inf** (*float*): Infiltration into the soil [mm].
                - **wc_new** (*float*): New liquid water content in
                  the snow [mm].
                - **sp_new** (*float*): New snow pack state [mm].

        Examples:
            >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
            >>> inf, wc_new, sp_new = HBVBergestrom92.snow(
            ...     temp=5.0, rf=3.0, sf=0.0, wc_old=2.0,
            ...     sp_old=10.0, tt=0.0, cfmax=3.0, cfr=0.05,
            ...     cwh=0.1,
            ... )
            >>> sp_new
            0.0
            >>> inf > 0
            True
        """
        # if temp > melting threshold
        if temp > tt:
            # then either some snow will melt or the entire snow will melt
            if cfmax * (temp - tt) < sp_old + sf:
                # if amount of melted snow < the entire existing snow (previous amount+new)
                melt = cfmax * (temp - tt)
            else:
                # if the amount of melted snow > the entire existing snow (previous amount+new)
                # then the entire existing snow will melt (old snow pack + the current snowfall)
                melt = sp_old + sf

            sp_new = sp_old + sf - melt
            wc_int = wc_old + melt + rf
        else:
            # if temp < melting threshold,
            # then either some water will freeze or all the water willfreeze
            if cfr * cfmax * (tt - temp) < wc_old + rf:
                refr = cfr * cfmax * (tt - temp)
                # cfmax*(ttm-temp) is the melting rate of snow while cfr*cfmax*(ttm-temp)
                # is the freezing rate of melted water (rate of freezing > rate of melting)
            else:
                # if the amount of frozen water > entire water available
                refr = wc_old + rf

            sp_new = sp_old + sf + refr
            wc_int = wc_old - refr + rf

        if wc_int > cwh * sp_new:
            # if water content > holding water capacity of the snow
            inf = wc_int - cwh * sp_new
            # water content will infiltrate
            wc_new = cwh * sp_new
            # and the capacity of snow of holding water will retained
        else:  # if water content < holding water capacity of the snow
            inf = 0.0  # no infiltration
        wc_new = wc_int

        return inf, wc_new, sp_new

    @staticmethod
    def soil(temp, inf, ep, sm_old, uz_old, tm, fc, beta, e_corr, lp):
        """Compute soil moisture balance and upper zone recharge.

        The model checks the amount of water that can infiltrate the
        soil from liquid precipitation and snow pack melting. A part
        of the water is stored as soil moisture, while the rest
        becomes runoff routed to the upper zone tank.

        Actual evaporation from the soil box equals the potential
        evaporation if ``SM/FC`` is above ``LP``, while a linear
        reduction is used when ``SM/FC`` is below ``LP``.
        Groundwater recharge is added to the upper groundwater box.

        Args:
            temp (float): Temperature [C].
            inf (float): Actual infiltration [mm].
            ep (float): Potential evapotranspiration [mm].
            sm_old (float): Previous soil moisture value [mm].
            uz_old (float): Previous upper zone value [mm].
            tm (float): Average long term temperature [C].
            fc (float): Field capacity [mm].
            beta (float): Shape coefficient for effective
                precipitation separation [-].
            e_corr (float): Evapotranspiration correction factor [-].
            lp (float): Wilting point as a fraction of field
                capacity [-].

        Returns:
            tuple[float, float]: A tuple of
                ``(sm_new, uz_int_1)`` where:
                - **sm_new** (*float*): New soil moisture value [mm].
                - **uz_int_1** (*float*): New value of direct runoff
                  into the upper zone [mm].

        Examples:
            >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
            >>> sm_new, uz_int_1 = HBVBergestrom92.soil(
            ...     temp=20.0, inf=5.0, ep=3.0, sm_old=50.0,
            ...     uz_old=10.0, tm=18.0, fc=200.0, beta=2.0,
            ...     e_corr=1.0, lp=0.9,
            ... )
            >>> sm_new > 0
            True
            >>> uz_int_1 > uz_old
            True
        """
        # recharge to the upper zone
        r = ((sm_old / fc) ** beta) * inf

        # Adjusted potential evapotranspiration
        ep_int = (1.0 + (temp - tm) * e_corr) * ep

        ea = min(ep_int, (sm_old / (lp * fc)) * ep_int)

        """
        capilary flux related calculations
        # capilary rise
        # cf = c_flux*((fc - sm_old)/fc)

        # if capilary rise is more than what is available take all the available and leave it empty

        # if uz_old + r < cf:
            # cf= uz_old + r
            # uz_int_1=0
        # else:
            # uz_int_1 = uz_old + r - cf

        # sm_new = max(sm_old + inf - r + cf - ea, 0)
        """

        uz_int_1 = uz_old + r
        sm_new = max(sm_old + inf - r - ea, 0)

        return sm_new, uz_int_1

    @staticmethod
    def response(lz_old, uz_int_1, perc, k, k1, k2, uzl):
        """Compute the runoff response from upper and lower zones.

        The response routine transforms the current values of upper
        and lower zone storages into discharge. It also controls the
        recharge of the lower zone tank (baseflow).

        ``perc`` defines the maximum percolation rate from the upper
        to the lower groundwater box. Runoff from the groundwater
        boxes is computed as the sum of two or three linear outflow
        equations depending on whether the upper zone storage is
        above the threshold value ``uzl``.

        Args:
            lz_old (float): Previous lower zone value [mm].
            uz_int_1 (float): Previous upper zone value before
                percolation [mm].
            perc (float): Percolation value [mm/timestep].
            k (float): Direct runoff (surface) recession
                coefficient [-].
            k1 (float): Upper zone (interflow) recession
                coefficient [-].
            k2 (float): Lower zone (baseflow) recession
                coefficient [-].
            uzl (float): Upper zone threshold value [mm].

        Returns:
            tuple[float, float, float, float]: A tuple of
                ``(q_uz, q_lz, uz_new, lz_new)`` where:
                - **q_uz** (*float*): Upper zone discharge
                  (surface runoff + interflow) [mm/timestep].
                - **q_lz** (*float*): Lower zone discharge
                  (baseflow) [mm/timestep].
                - **uz_new** (*float*): New upper zone storage [mm].
                - **lz_new** (*float*): New lower zone storage [mm].

        Examples:
            >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
            >>> q_uz, q_lz, uz_new, lz_new = HBVBergestrom92.response(
            ...     lz_old=30.0, uz_int_1=20.0, perc=1.0, k=0.005,
            ...     k1=0.03, k2=0.015, uzl=10.0,
            ... )
            >>> q_uz > 0
            True
            >>> q_lz > 0
            True
            >>> uz_new >= 0
            True
            >>> lz_new >= 0
            True
        """
        # upper zone
        # if perc > Quz then perc = Quz and Quz = 0 if not perc = value and Quz= Quz-perc so take the min
        uz_int_2 = np.max([uz_int_1 - perc, 0.0])

        # surface runoff
        q_0 = k * np.max([uz_int_2 - uzl, 0])

        # Interflow
        q_1 = k1 * uz_int_2

        # as K & k1 are a very small values (0.005) this condition will never happen
        if q_0 + q_1 > uz_int_2:  # if q_0 =30 and UZ=20
            q_0 = uz_int_2 * 0.67  # q_0 = 20
            q_1 = uz_int_2 * 0.33

        uz_new = uz_int_2 - (q_0 + q_1)

        # lower zone tank
        # if the percolation > upper zone Q all the Quz will percolate
        lz_int_1 = lz_old + np.min([perc, uz_int_1])

        q_2 = k2 * lz_int_1

        if q_2 > lz_int_1:
            q_2 = lz_int_1

        lz_new = lz_int_1 - q_2

        q_uz = q_0 + q_1

        return q_uz, q_2, uz_new, lz_new

    @staticmethod
    def tf(maxbas):
        """Generate transfer function weights for triangular routing.

        Computes a set of normalized weights based on a triangular
        transfer function. The weights grow linearly for the first
        half of the ``maxbas`` interval and recede linearly for the
        second half.

        Args:
            maxbas (int): Number of time steps for the triangular
                transfer function.

        Returns:
            numpy.ndarray: Normalized weights for the transfer
                function, summing to 1.0.

        Examples:
            >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
            >>> import numpy as np
            >>> w = HBVBergestrom92.tf(3)
            >>> np.isclose(w.sum(), 1.0)
            True
            >>> len(w)
            3
        """
        wi = []
        for x in range(1, maxbas + 1):
            if x <= maxbas / 2.0:
                # Growing transfer
                wi.append(x / (maxbas + 2.0))
            else:
                # Receding transfer
                wi.append(1.0 - (x + 1) / (maxbas + 2.0))

        # Normalise weights
        wi = np.array(wi) / np.sum(wi)
        return wi

    def routing(self, q, maxbas=1):
        """Apply triangular transfer function routing to discharge.

        Routes the discharge signal through a triangular transfer
        function defined by the ``maxbas`` parameter. The transfer
        function weights are generated by :meth:`tf`.

        Args:
            q (numpy.ndarray): Discharge array [mm/timestep].
            maxbas (int): Transfer function length in time steps.
                Must be >= 1. Defaults to 1.

        Returns:
            numpy.ndarray: Routed discharge array with the same
                shape as ``q``.

        Raises:
            AssertionError: If ``maxbas`` is less than 1.

        Examples:
            >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
            >>> import numpy as np
            >>> model = HBVBergestrom92()
            >>> q = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
            >>> q_r = model.routing(q, maxbas=1)
            >>> len(q_r) == len(q)
            True
        """
        assert maxbas >= 1, "Maxbas value has to be larger than 1"
        # Get integer part of maxbas
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

    def simulate(
        self, prec, temp, et, ll_temp, par, init_st=None, q_init=None, snow=0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the HBV Bergestrom92 model simulation.

        Executes the HBV model for the number of time steps in the
        precipitation input. The model sequentially calls the
        precipitation, snow, soil, and response routines at each
        time step, updating state variables accordingly.

        Args:
            prec (array_like): Average precipitation [mm/timestep],
                array of length ``n``.
            temp (array_like): Average temperature [C], array of
                length ``n``.
            et (array_like): Potential evapotranspiration
                [mm/timestep], array of length ``n``.
            ll_temp (array_like): Long term average temperature [C],
                array of length ``n``.
            par (array_like): Parameter vector. When ``snow=1``,
                expects 15 parameters:
                ``[tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta,
                e_corr, lp, k, k1, k2, uzl, perc]``.
                When ``snow=0``, expects 10 parameters:
                ``[rfcf, fc, beta, e_corr, lp, k, k1, k2, uzl,
                perc]``.
            init_st (array_like, optional): Initial model states
                ``[sp, sm, uz, lz, wc]`` in mm. Defaults to
                ``[0.0, 10.0, 10.0, 10.0, 0.0]``.
            q_init (float, optional): Initial discharge value. If
                not specified, it is computed from initial states
                and parameters.
            snow (int): Flag indicating whether snow processes are
                active. Use ``1`` for snow, ``0`` for no snow.
                Defaults to 0.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                A tuple of ``(q_uz, q_lz, st)`` where:
                - **q_uz** (*numpy.ndarray*): Upper zone discharge
                  (surface runoff + interflow) for ``n+1`` time
                  steps [mm/timestep].
                - **q_lz** (*numpy.ndarray*): Lower zone discharge
                  (baseflow) for ``n+1`` time steps [mm/timestep].
                - **st** (*numpy.ndarray*): Model states array of
                  shape ``(n+1, 5)`` with columns
                  ``[sp, sm, uz, lz, wc]`` in mm.

        Examples:
            >>> from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92
            >>> import numpy as np
            >>> model = HBVBergestrom92()
            >>> n = 10
            >>> prec = np.random.uniform(0, 20, n)
            >>> temp = np.random.uniform(15, 30, n)
            >>> et = np.random.uniform(0, 5, n)
            >>> ll_temp = np.full(n, 20.0)
            >>> par = [1.0, 200.0, 2.0, 1.0, 0.9, 0.005, 0.03,
            ...        0.015, 10.0, 1.0]
            >>> q_uz, q_lz, st = model.simulate(
            ...     prec, temp, et, ll_temp, par, snow=0,
            ... )
            >>> q_uz.shape == (n + 1,)
            True
            >>> st.shape == (n + 1, 5)
            True
        """
        st = np.zeros([len(prec) + 1, 5], dtype=np.float32)
        q_0 = np.zeros([len(prec) + 1], dtype=np.float32)
        q_1 = np.zeros([len(prec) + 1], dtype=np.float32)
        q_uz = np.zeros([len(prec) + 1], dtype=np.float32)
        q_lz = np.zeros([len(prec) + 1], dtype=np.float32)

        if init_st is None:  # 0  1  2  3  4  5
            st[0, :] = DEF_ST  # [sp,sm,uz,lz,wc,LA]
        else:
            st[0, :] = init_st

        ### initial runoff
        # calculate the runoff for the first time step
        if q_init is None:
            if snow == 1:
                # upper zone
                q_0[0] = par[10] * max(st[0, 2] - par[13], 0)
                q_1[0] = par[11] * st[0, 2]
                q_uz[0] = q_0[0] + q_1[0]
                # lower zone
                q_lz[0] = par[12] * st[0, 3]

            else:
                # upper zone
                q_0[0] = par[5] * max(st[0, 2] - par[8], 0)
                q_1[0] = par[6] * st[0, 2]
                q_uz[0] = q_0[0] + q_1[0]
                # lower zone
                q_lz[0] = par[7] * st[0, 3]
        else:  # if initial runoff value is given distribute it evenlt between upper and lower responses
            q_uz[0] = q_init / 2
            q_lz[0] = q_init / 2

        ## Parse of parameters from input vector to model
        if snow == 1:
            # assert len(p) == 16, "current version of HBV (with snow) takes 18 parameter you have entered "+str(len(p))
            tt = par[0]
            rfcf = par[1]
            sfcf = par[2]
            # snow function
            cfmax = par[3]
            cwh = par[4]
            cfr = par[5]
            # soil function
            fc = par[6]
            beta = par[7]
            e_corr = par[8]
            lp = par[9]
            # response function
            k = par[10]
            k1 = par[11]
            k2 = par[12]
            uzl = par[13]
            perc = par[14]

        elif snow == 0:
            # assert len(par) >= 11, "current version of HBV (without snow) takes 11 parameter you have entered "+str(len(par))
            tt = 2.0  # very low but it does not matter as temp is 25 so it is greater than 2
            rfcf = par[0]  # 1.0 #par[16] # all precipitation becomes rainfall
            sfcf = 0.00001  # there is no snow
            # snow function
            # cfmax = 0.00001  # as there is no melting  and sp+sf=zero all the time so it doesn't matter the value of cfmax
            # cwh = 0.00001    # as sp is always zero it doesn't matter all wc will go as inf
            # cfr = 0.000001   # as temp > ttm all the time so it doesn't matter the value of cfr but put it zero
            # soil function
            fc = par[1]
            beta = par[2]
            e_corr = par[3]
            lp = par[4]
            # response function
            k = par[5]
            k1 = par[6]
            k2 = par[7]
            uzl = par[8]
            perc = par[9]

        for i in range(1, len(prec)):
            ## Parse of Inputs
            preci = prec[i]  # Precipitation [mm]
            tempi = temp[i]  # Temperature [C]
            epi = et[i]  # Long terms (monthly) Evapotranspiration [mm]
            tmi = ll_temp[i]  # Long term (monthly) average temperature [C]

            ## Parse of states
            sp_old = st[i - 1, 0]
            sm_old = st[i - 1, 1]
            uz_old = st[i - 1, 2]
            lz_old = st[i - 1, 3]
            wc_old = st[i - 1, 4]

            rf, sf = self.precipitation(preci, tempi, tt, rfcf, sfcf)

            if snow == 0:
                inf = rf
                wc_new = 0
                sp_new = 0
            else:
                inf, wc_new, sp_new = self.snow(
                    tempi, rf, sf, wc_old, sp_old, tt, cfmax, cfr, cwh
                )

            sm_new, uz_int_1 = self.soil(
                tempi, inf, epi, sm_old, uz_old, tmi, fc, beta, e_corr, lp
            )

            q_uz[i], q_lz[i], uz_new, lz_new = self.response(
                lz_old, uz_int_1, perc, k, k1, k2, uzl
            )

            st[i, :] = [sp_new, sm_new, uz_new, lz_new, wc_new]

        return q_uz, q_lz, st
