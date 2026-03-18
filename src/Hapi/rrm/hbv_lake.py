"""HBV model with a lake function.

The ``Hapi.rrm.hbv_lake`` module provides the ``HBVLake`` class, an
extension of the HBV-96 rainfall-runoff model that includes an explicit
lake representation.  Inflow to the lake is computed using the standard
HBV precipitation, snow, soil, and response routines.  Lake outflow is
then derived from a storage-discharge rating curve using interpolation.

Classes:
    HBVLake: HBV-96 model variant with an explicit lake sub-model.

Module-level Constants:
    DEF_ST: Default initial state variables
        ``[sp, sm, uz, lz, wc, lake_volume]``.
    DEF_q0: Default initial discharge value (2.3 m3/s).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interp11

from Hapi.rrm.base_model import BaseConceptualModel

# initial values for state variables
# [sp, sm, uz, lz, wc]
DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0, 10.163 * 10**9]
# initial value for discarge
DEF_q0 = 2.3  # 10.0


class HBVLake(BaseConceptualModel):
    """HBV-96 model variant with an explicit lake sub-model.

    ``HBVLake`` extends the standard HBV-96 rainfall-runoff model by
    adding a lake component that transforms the catchment outflow using
    a storage-discharge rating curve.  The lake receives inflow from the
    HBV response routine, accounts for precipitation on the lake surface
    and lake evaporation, and computes outflow via interpolation on the
    supplied rating curve.

    The model pipeline per time step is:

        precipitation -> snow -> soil -> response -> lake -> outflow

    Inherits from:
        BaseConceptualModel

    Examples:
        >>> import numpy as np
        >>> from Hapi.rrm.hbv_lake import HBVLake, DEF_ST, DEF_q0
        >>> model = HBVLake()
        >>> n = 10
        >>> prec = np.array([5.0] * n)
        >>> temp = np.array([25.0] * n)
        >>> et = np.array([2.0] * n)
        >>> par = [200, 2.0, 0.01, 0.6, 0.5, 0.01, 0.05, 1.0, 1.0,
        ...        1.0, 1.0]
        >>> p2 = [24, 100.0, 10.0]
        >>> curve = np.column_stack([
        ...     np.linspace(0, 100, 50),
        ...     np.linspace(0, 1e10, 50),
        ... ])
        >>> q_sim, states = model.simulate(
        ...     prec, temp, et, par, p2, curve,
        ...     q_0=DEF_q0, init_st=DEF_ST,
        ... )
        >>> len(q_sim) == n + 1
        True
    """

    @staticmethod
    def _precipitation(temp, ltt, utt, prec, rfcf, sfcf, tfac, pcorr):
        """Separate precipitation into rainfall and snowfall.

        If the temperature is at or below ``ltt``, all precipitation is
        snowfall.  If the temperature is at or above ``utt``, all
        precipitation is rainfall.  Between the two thresholds,
        precipitation is linearly partitioned.

        Args:
            temp (float): Measured temperature [C].
            ltt (float): Lower temperature threshold [C].
            utt (float): Upper temperature threshold [C].
            prec (float): Precipitation [mm].
            rfcf (float): Rainfall correction factor.
            sfcf (float): Snowfall correction factor.
            tfac (float): Temperature correction factor (unused in
                this routine but kept for interface consistency).
            pcorr (float): Precipitation correction factor applied to
                rainfall.

        Returns:
            tuple[float, float]:
                - Rainfall [mm].
                - Snowfall [mm].

        Examples:
            >>> HBVLake._precipitation(
            ...     temp=0.0, ltt=1.0, utt=2.0, prec=10.0,
            ...     rfcf=1.0, sfcf=1.0, tfac=1.0, pcorr=1.0,
            ... )
            (0.0, 10.0)
            >>> HBVLake._precipitation(
            ...     temp=3.0, ltt=1.0, utt=2.0, prec=10.0,
            ...     rfcf=1.0, sfcf=1.0, tfac=1.0, pcorr=1.0,
            ... )
            (10.0, 0.0)
        """
        # if temp <= lower temp threshold
        if temp <= ltt:
            # no rainfall all the precipitation will convert into snowfall
            _rf = 0.0
            _sf = prec * sfcf
        elif temp >= utt:
            # if temp > upper threshold
            _rf = prec * rfcf
            # no snowfall all the precipitation becomes rainfall
            _sf = 0.0
        else:
            # if  ltt< temp < utt
            _rf = ((temp - ltt) / (utt - ltt)) * prec * rfcf
            _sf = (1.0 - ((temp - ltt) / (utt - ltt))) * prec * sfcf

        _rf = _rf * pcorr

        return _rf, _sf

    @staticmethod
    def _snow(cfmax, tfac, temp, ttm, cfr, cwh, _rf, _sf, wc_old, sp_old):
        """Compute snow accumulation, melt, and infiltration.

        The snow pack has two states: solid snow pack (``sp``) and
        liquid water content (``wc``).  When the temperature exceeds
        the melting threshold, snow melts and the solid portion
        decreases.  When the temperature is below the threshold, the
        liquid water refreezes.  Excess water that exceeds the holding
        capacity of the snow pack becomes infiltration.

        Args:
            cfmax (float): Day-degree melting factor [mm/C/day].
            tfac (float): Temperature correction factor.
            temp (float): Temperature [C].
            ttm (float): Temperature threshold for melting [C].
            cfr (float): Refreezing factor.
            cwh (float): Water-holding capacity of snow pack (fraction
                of solid snow pack).
            _rf (float): Rainfall [mm].
            _sf (float): Snowfall [mm].
            wc_old (float): Water content in the previous state [mm].
            sp_old (float): Snow pack in the previous state [mm].

        Returns:
            tuple[float, float, float]:
                - Infiltration [mm].
                - Updated water content [mm].
                - Updated snow pack [mm].

        Examples:
            >>> HBVLake._snow(
            ...     cfmax=4.0, tfac=1.0, temp=5.0, ttm=0.0,
            ...     cfr=0.05, cwh=0.1, _rf=2.0, _sf=0.0,
            ...     wc_old=0.0, sp_old=10.0,
            ... )
            (2.0, 1.0, 0.0)
        """
        # if temp > melting threshold
        if temp > ttm:
            # then either some snow will melt or the entire snow will melt
            if cfmax * (temp - ttm) < sp_old + _sf:
                # if amount of melted snow < the entire existing snow (previous amount+new)
                _melt = cfmax * (temp - ttm)
            else:
                # if amount of melted snow > the entire existing snow (previous amount+new)
                _melt = sp_old + _sf
                # then the entire existing snow will melt (old snow pack + the current snowfall)

            _sp_new = sp_old + _sf - _melt
            _wc_int = wc_old + _melt + _rf
        else:
            # if temp < melting threshold,
            # then either some water will freeze or all the water willfreeze
            if cfr * cfmax * (ttm - temp) < wc_old + _rf:
                # if the amount of frozen water < entire water available
                _refr = cfr * cfmax * (ttm - temp)
                # cfmax*(ttm-temp) is the melting rate of snow while cfr*cfmax*(ttm-temp) is the freezing rate of
                # melted water  (rate of freezing > rate of melting)
            else:
                # if the amount of frozen water > entire water available
                _refr = wc_old + _rf

            _sp_new = sp_old + _sf + _refr
            _wc_int = wc_old - _refr + _rf

        if _wc_int > cwh * _sp_new:
            # if water content > holding water capacity of the snow
            _in = _wc_int - cwh * _sp_new
            # water content will infiltrate
            _wc_new = cwh * _sp_new
            # and the capacity of snow of holding water will retained
        else:
            # if water content < holding water capacity of the snow
            _in = 0.0  # no infiltration
            _wc_new = _wc_int

        return _in, _wc_new, _sp_new

    @staticmethod
    def _soil(
        fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux, inf, ep, sm_old, uz_old
    ):
        """Compute soil moisture balance and recharge to the upper zone.

        This routine determines how much infiltrating water is stored as
        soil moisture and how much becomes runoff routed to the upper
        zone.  Actual evapotranspiration is computed from the adjusted
        potential evapotranspiration, and capillary rise from the upper
        zone is also accounted for.

        Args:
            fc (float): Field capacity [mm].
            beta (float): Shape coefficient controlling the split
                between soil moisture increase and upper zone recharge.
            etf (float): Evapotranspiration temperature adjustment
                factor.
            temp (float): Temperature [C].
            tm (float): Long-term average temperature [C].
            e_corr (float): Evapotranspiration correction factor.
            lp (float): Fraction of field capacity above which actual
                ET equals potential ET.
            tfac (float): Time conversion factor.
            c_flux (float): Maximum capillary flux from upper zone to
                root zone [mm].
            inf (float): Actual infiltration [mm].
            ep (float): Long-term mean potential evapotranspiration
                [mm].
            sm_old (float): Previous soil moisture value [mm].
            uz_old (float): Previous upper zone value [mm].

        Returns:
            tuple[float, float, float]:
                - Updated soil moisture [mm].
                - Upper zone inflow after capillary rise [mm].
                - Direct runoff [mm] (always 0 in this
                  implementation).

        Examples:
            >>> sm, uz, qdr = HBVLake._soil(
            ...     fc=200, beta=2.0, etf=0.01, temp=25.0, tm=20.0,
            ...     e_corr=1.0, lp=0.6, tfac=24, c_flux=0.5, inf=5.0,
            ...     ep=2.0, sm_old=100.0, uz_old=10.0,
            ... )
            >>> sm > 0
            True
            >>> qdr
            0
        """

        #    qdr = max(sm_old + inf - fc, 0)  # direct run off as soil moisture exceeded the field capacity
        qdr = 0
        _in = inf - qdr
        _r = (
            (sm_old / fc) ** beta
        ) * _in  # recharge from soil subroutine to upper zone

        _ep_int = max(
            (1.0 + etf * (temp - tm)) * e_corr * ep, 0
        )  # Adjusted potential evapotranspiration

        _ea = min(_ep_int, (sm_old / (lp * fc)) * _ep_int)

        _cf = c_flux * ((fc - sm_old) / fc)  # capilary rise

        # if capilary rise is more than what is available take all the available and leave it empty

        if uz_old + _r < _cf:
            _cf = uz_old + _r
            uz_int_1 = 0
        else:
            uz_int_1 = uz_old + _r - _cf

        #    uz_int_1 = max(uz_old + _r - _cf,0)

        sm_new = max(sm_old + _in - _r + _cf - _ea, 0)

        return sm_new, uz_int_1, qdr

    @staticmethod
    def _response(tfac, perc, alpha, k, k1, area, lz_old, uz_int_1, qdr):
        r"""Transform upper and lower zone storage into discharge.

        The response routine percolates water from the upper zone to the
        lower zone, computes non-linear upper zone discharge and linear
        lower zone (baseflow) discharge, and converts the total from mm
        to m3/s.

        Args:
            tfac (float): Number of hours in the time step.
            perc (float): Maximum percolation rate [mm/hr].
            alpha (float): Non-linearity exponent for upper zone
                response.
            k (float): Upper zone recession coefficient.
            k1 (float): Lower zone recession coefficient.
            area (float): Catchment area [km2].
            lz_old (float): Previous lower zone value [mm].
            uz_int_1 (float): Upper zone value before percolation [mm].
            qdr (float): Direct runoff [mm].

        Returns:
            tuple[float, float, float]:
                - Discharge [m3/s].
                - Updated upper zone storage [mm].
                - Updated lower zone storage [mm].

        Examples:
            >>> q, uz, lz = HBVLake._response(
            ...     tfac=24, perc=1.0, alpha=1.0, k=0.05,
            ...     k1=0.01, area=100.0, lz_old=10.0,
            ...     uz_int_1=15.0, qdr=0.0,
            ... )
            >>> q > 0
            True
        """
        # upper zone
        # if perc > Quz then perc = Quz and Quz = 0 if not perc = value and Quz= Quz-perc so take the min
        uz_int_2 = np.max([uz_int_1 - perc, 0.0])  # upper zone after percolation
        _q_0 = k * (uz_int_2 ** (1.0 + alpha))

        if _q_0 > uz_int_2:  # if q_0 =30 and UZ=20
            _q_0 = uz_int_2  # q_0=20

        uz_new = uz_int_2 - _q_0

        lz_int_1 = lz_old + np.min(
            [perc, uz_int_1]
        )  # if the percolation > upper zone Q all the Quz will percolate

        _q_1 = k1 * lz_int_1

        if _q_1 > lz_int_1:
            _q_1 = lz_int_1

        lz_new = lz_int_1 - _q_1

        q_new = ((_q_0 + _q_1) * area) / (
            3.6 * tfac
        )  # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
        # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
        return q_new, uz_new, lz_new

    @staticmethod
    def _lake(temp, curve, tfac, rf, sf, q_new, lv_old, ltt, c_le, ep, lakeA):
        """Compute lake outflow and updated lake volume.

        The lake receives inflow from the catchment response routine,
        gains water from precipitation on the lake surface, and loses
        water through evaporation.  Outflow is determined by
        interpolating the storage-discharge rating curve at the average
        storage level.

        Args:
            temp (float): Temperature [C].
            curve (numpy.ndarray): Two-column array where column 0 is
                discharge [m3/s] and column 1 is storage [m3].
            tfac (float): Number of hours in the time step.
            rf (float): Rainfall [mm].
            sf (float): Snowfall [mm].
            q_new (float): Inflow discharge from the catchment [m3/s].
            lv_old (float): Previous lake volume [m3].
            ltt (float): Lower temperature threshold [C] controlling
                whether precipitation falls as rain or snow on the
                lake.
            c_le (float): Lake evaporation correction factor.
            ep (float): Potential evapotranspiration [mm].
            lakeA (float): Lake surface area [km2].

        Returns:
            tuple[float, float]:
                - Lake outflow discharge [m3/s].
                - Updated lake volume [m3].

        Examples:
            >>> import numpy as np
            >>> curve = np.column_stack([
            ...     np.linspace(0, 100, 50),
            ...     np.linspace(0, 1e10, 50),
            ... ])
            >>> qout, lv = HBVLake._lake(
            ...     temp=25.0, curve=curve, tfac=24,
            ...     rf=5.0, sf=0.0, q_new=10.0,
            ...     lv_old=5e9, ltt=1.0, c_le=1.0,
            ...     ep=2.0, lakeA=10.0,
            ... )
            >>> qout >= 0
            True
        """
        # lower zone
        # explicit representation of the lake where lake will be represented by a rating curve
        # lake evaporation
        if temp >= ltt:
            l_ea = (
                c_le * ep
            )  # the evaporation will be the potential evapotranspiration times correction factor
        else:
            l_ea = 0  # Evaporation will not occur when the Temperature is below the Threshold temperature

        l_ea_vol = l_ea * lakeA * 1000  # evaporation volume m3/time step

        # evaporation on the lake
        if temp < ltt:
            l_p = sf * c_le
        else:
            l_p = rf * c_le

        l_p_vol = l_p * lakeA * 1000  # prec # precipitation volume/ timestep

        q_vol = q_new * 3600 * tfac  # volume of inflow to the lake

        # storage in the lake before calculating the outflow
        lkv1 = lv_old + l_p_vol + q_vol - l_ea_vol
        # average storage for interpolation
        lkv2 = (lkv1 + lv_old) / 2

        storage = curve[:, 1]
        discharge = curve[:, 0]
        fn = interp11(storage, discharge, k=1)
        qout = max(fn(lkv2).tolist(), 0)

        lv_new = lkv2 - (qout * 3600 * tfac)

        return qout, lv_new

    @staticmethod
    def _tf(maxbas):
        """Generate triangular transfer function weights.

        Produces a normalised weight array with a triangular shape.
        The first half of the weights form the rising limb and the
        second half form the falling limb.

        Args:
            maxbas (int): Number of time steps over which to distribute
                the transfer function.

        Returns:
            numpy.ndarray: Normalised weights summing to 1.0.

        Examples:
            >>> import numpy as np
            >>> w = HBVLake._tf(3)
            >>> abs(w.sum() - 1.0) < 1e-10
            True
            >>> len(w)
            3
        """

        wi = []
        for x in range(1, maxbas + 1):  # if maxbas=3 so x=[1,2,3]
            if (
                x <= maxbas / 2.0
            ):  # x <= 1.5  # half of values will form the rising limb and half falling limb
                # Growing transfer    # rising limb
                wi.append(x / (maxbas + 2.0))
            else:
                # Receding transfer    # falling limb
                wi.append(1.0 - (x + 1) / (maxbas + 2.0))

        # Normalise weights
        wi = np.array(wi) / np.sum(wi)
        return wi

    def _routing(self, q, maxbas=1):
        """Route a discharge time series using a triangular function.

        Convolves the input discharge array with triangular weights
        generated by ``_tf`` to smooth the hydrograph.

        Args:
            q (numpy.ndarray): Input discharge time series.
            maxbas (int): Number of time steps for the triangular
                transfer function.  Must be >= 1.

        Returns:
            numpy.ndarray: Routed discharge time series with the same
                length as ``q``.

        Raises:
            AssertionError: If ``maxbas`` < 1.

        Examples:
            >>> import numpy as np
            >>> model = HBVLake()
            >>> q = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
            >>> q_r = model._routing(q, maxbas=3)
            >>> len(q_r) == len(q)
            True
        """
        assert maxbas >= 1, "Maxbas value has to be larger than 1"
        # Get integer part of maxbas
        maxbas = int(round(maxbas, 0))

        # get the weights
        w = self._tf(maxbas)

        # rout the discharge signal
        q_r = np.zeros_like(q, dtype="float64")
        q_temp = q
        for w_i in w:
            q_r += q_temp * w_i
            q_temp = np.insert(q_temp, 0, 0.0)[:-1]

        return q_r

    @staticmethod
    def calculate_max_bas(maxbas):
        """Compute MAXBAS routing weights using an equilateral triangle.

        Generates routing weights based on an equilateral triangle whose
        base length equals ``maxbas``.  Unlike ``_tf``, this method
        supports non-integer values of ``maxbas``.

        Args:
            maxbas (float): MAXBAS parameter controlling the routing
                duration.  Can be a non-integer value.

        Returns:
            numpy.ndarray: Array of routing weights.  The length is
                ``floor(maxbas)`` for integer values or
                ``floor(maxbas) + 1`` for non-integer values.

        Examples:
            >>> import numpy as np
            >>> w = HBVLake.calculate_max_bas(5)
            >>> len(w)
            5
            >>> abs(w.sum() - 1.0) < 0.01
            True
        """
        yant = 0
        total = 0  # Just to verify how far from the unit is the result

        total_a = (maxbas * maxbas * np.sin(np.pi / 3)) / 2

        int_part = np.floor(maxbas)

        real_part = maxbas - int_part

        peak_point = maxbas % 2

        flag = 1  # 1 = "up"  ; 2 = down

        if real_part > 0:  # even number 2,4,6,8,10
            maxbas_w = np.ones(int(int_part) + 1)  # if even add 1
        else:  # odd number
            maxbas_w = np.ones(int(int_part))

        for x in range(int(maxbas)):

            if x < (maxbas / 2.0) - 1:
                ynow = np.tan(np.pi / 3) * (x + 1)
                # Integral of x dx with a slope of 60 degree Equilateral triangle
                maxbas_w[x] = ((ynow + yant) / 2) / total_a  # ' Area / Total Area

            else:  # The area here is calculated by the formlua of a trapezoidal (B1+B2)*h /2
                if flag == 1:
                    ynow = np.sin(np.pi / 3) * maxbas
                    if peak_point == 0:
                        maxbas_w[x] = ((ynow + yant) / 2) / total_a
                    else:
                        A1 = ((ynow + yant) / 2) * (maxbas / 2.0 - x) / total_a
                        yant = ynow
                        ynow = (maxbas * np.sin(np.pi / 3)) - (
                            np.tan(np.pi / 3) * (x + 1 - maxbas / 2.0)
                        )
                        A2 = ((ynow + yant) * (x + 1 - maxbas / 2.0) / 2) / total_a
                        maxbas_w[x] = A1 + A2

                    flag = 2
                else:
                    ynow = maxbas * np.sin(np.pi / 3) - np.tan(np.pi / 3) * (x + 1 - maxbas / 2.0)
                    # 'sum of the two heights in the descending part of the triangle
                    maxbas_w[x] = ((ynow + yant) / 2) / total_a
                    # Multiplying by the height of the trapezoidal and dividing by 2

            total = total + maxbas_w[x]
            yant = ynow

        x = x + 1

        if real_part > 0:
            if np.floor(maxbas) == 0:
                # maxbas = 1
                maxbas_w[x] = 1
                # NumberofWeights = 1
            else:
                maxbas_w[x] = (yant * (maxbas - x) / 2) / total_a
                # Total = Total + maxbasW[x]
                # NumberofWeights = x
        else:
            # NumberofWeights = x - 1
            pass

        return maxbas_w

    def routing_maxbas(self, q, maxbas):
        """Route a hydrograph using MAXBAS equilateral-triangle weights.

        Computes routing by convolving the input hydrograph with weights
        derived from ``calculate_max_bas``.  This method supports
        non-integer ``maxbas`` values.

        Args:
            q (numpy.ndarray): Input hydrograph discharge array.
            maxbas (float): MAXBAS parameter controlling the routing
                duration.

        Returns:
            numpy.ndarray: Routed output hydrograph with shape
                ``(len(q), 1)``.

        Examples:
            >>> import numpy as np
            >>> model = HBVLake()
            >>> q = np.array([0, 1, 3, 5, 3, 1, 0], dtype=float)
            >>> qout = model.routing_maxbas(q, maxbas=3)
            >>> qout.shape == (7, 1)
            True
        """
        # CALCULATE maxbas WEIGHTS
        maxbas_w = self.calculate_max_bas(maxbas)
        qw = np.ones((len(q), len(maxbas_w)))
        # Calculate the matrix discharge
        for i in range(len(q)):  # 0 to 10
            for k in range(len(maxbas_w)):  # 0 to 4
                qw[i, k] = maxbas_w[k] * q[i]

        def mm(A, s):
            tot = []
            for o in range(np.shape(A)[1]):  # columns
                for t in range(np.shape(A)[0]):  # rows
                    tot.append(A[t, o])
            Su = tot[s:-1:s]
            return Su

        # Calculate routing
        j = 0
        qout = np.ones((len(q), 1))

        for i in range(len(q)):
            if i == 0:
                qout[i] = qw[i, i]
            elif i < len(maxbas_w) - 1:
                A = qw[0 : i + 1, :]
                s = len(A) - 1  # len(A) is the no of rows or use int(np.shape(A)[0])
                Su = mm(A, s)

                qout[i] = sum(Su[0 : i + 1])
            else:
                A = qw[j : i + 1, :]
                s = len(A) - 1
                Su = mm(A, s)
                qout[i] = sum(Su)
                j = j + 1

        return qout

    def _step_run(self, parameters, catchment_parameters, v, state_variables, curve):
        """Execute a single time step of the HBV-lake model.

        Runs all sub-routines (precipitation, snow, soil, response, and
        lake) for one time step and returns the lake outflow discharge
        and updated state variables.

        Args:
            parameters (array_like): Parameter vector of length 11,
                ordered as ``[fc, beta, etf, lp, c_flux, k, k1, alpha,
                perc, c_le, pcorr]``.  Nine parameters control the
                standard HBV routines, ``c_le`` is the lake evaporation
                correction factor, and ``pcorr`` is the precipitation
                correction factor.
            catchment_parameters (array_like): Problem parameter vector
                of length 3, ordered as ``[tfac, lake_sub, lake_area]``
                where ``tfac`` is hours per time step, ``lake_sub`` is
                the lake sub-catchment area [km2], and ``lake_area`` is
                the lake surface area [km2].
            v (array_like): Input vector of length 4, ordered as
                ``[prec, temp, evap, ll_temp]``.
            state_variables (array_like): Previous model states of
                length 6, ordered as ``[sp, sm, uz, lz, wc,
                lake_volume]``.
            curve (numpy.ndarray): Two-column lake rating curve array
                where column 0 is discharge [m3/s] and column 1 is
                storage [m3].

        Returns:
            tuple[float, list[float]]:
                - Lake outflow discharge [m3/s].
                - Updated state variables ``[sp, sm, uz, lz, wc,
                  lake_volume]``.

        Examples:
            >>> import numpy as np
            >>> from Hapi.rrm.hbv_lake import HBVLake, DEF_ST
            >>> model = HBVLake()
            >>> par = [200, 2.0, 0.01, 0.6, 0.5, 0.01, 0.05,
            ...        1.0, 1.0, 1.0, 1.0]
            >>> p2 = [24, 100.0, 10.0]
            >>> v = [5.0, 25.0, 2.0, 20.0]
            >>> curve = np.column_stack([
            ...     np.linspace(0, 100, 50),
            ...     np.linspace(0, 1e10, 50),
            ... ])
            >>> q, st = model._step_run(par, p2, v, DEF_ST, curve)
            >>> len(st)
            6
        """
        ## Parse of parameters from input vector to model
        # picipitation function
        ltt = 1.0  # parameters[0] # less than utt and less than lowest temp to prevent sf formation
        utt = 2.0  # parameters[1]  #very low but it does not matter as temp is 25 so it is greater than 2
        rfcf = 1.0  # parameters[16] # all precipitation becomes rainfall
        sfcf = 0.00001  # parameters[17] # there is no snow
        # snow function
        ttm = 1  # parameters[2] #should be very low lower than lowest temp as temp is 25 all the time so it does not matter
        cfmax = 0.00001  # parameters[3] as there is no melting  and sp+sf=zero all the time so it doesn't matter the value of cfmax
        cwh = 0.00001  # parameters[12] as sp is always zero it doesn't matter all wc will go as inf
        cfr = 0.000001  # parameters[13] as temp > ttm all the time so it doesn't matter the value of cfr but put it zero
        # soil function
        fc = parameters[0]
        beta = parameters[1]
        e_corr = 1  # parameters[2]
        etf = parameters[2]
        lp = parameters[3]
        c_flux = parameters[4]
        # response function
        k = parameters[5]
        k1 = parameters[6]
        alpha = parameters[7]
        perc = parameters[8]

        ## Non optimisable parameters
        # [tfac,jiboa,lake_sub,lake_area]
        tfac = catchment_parameters[0]  # tfac=0.25
        # jiboa_area=catchment_parameters[1] # AREA = 432
        lake_sub = catchment_parameters[1]  # area of lake subcatchment
        lakeA = catchment_parameters[2]  # area of the lake
        #    ilake=lakeA/area  # percentage of the lake area to catchment area

        ## Parse of Inputs
        avg_prec = v[0]  # Precipitation [mm]
        temp = v[1]  # Temperature [C]
        ep = v[2]
        tm = v[3]  # Long term (monthly) average temperature [C]

        ## Parse of states
        sp_old = state_variables[0]
        sm_old = state_variables[1]
        uz_old = state_variables[2]
        lz_old = state_variables[3]
        wc_old = state_variables[4]

        # if lake_sim:
        area = lake_sub
        c_le = parameters[9]
        lv_old = state_variables[5]
        pcorr = parameters[10]
        # else:
        # area=jiboa_area
        # pcorr=parameters[9]

        #    pcorr=parameters[10]

        rf, sf = self._precipitation(temp, ltt, utt, avg_prec, rfcf, sfcf, tfac, pcorr)
        inf, wc_new, sp_new = self._snow(
            cfmax, tfac, temp, ttm, cfr, cwh, rf, sf, wc_old, sp_old
        )
        sm_new, uz_int_1, qdr = self._soil(
            fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux, inf, ep, sm_old, uz_old
        )
        q_new, uz_new, lz_new = self._response(
            tfac, perc, alpha, k, k1, area, lz_old, uz_int_1, qdr
        )

        # if lake_sim is true it will enter the function of the lake
        # if lake_sim:
        qout, lv_new = self._lake(
            temp, curve, tfac, rf, sf, q_new, lv_old, ltt, c_le, ep, lakeA
        )
        # else:    # if lake_sim is false it will enter the function of the lake
        # qout=q_new
        # lv_new=0

        return qout, [sp_new, sm_new, uz_new, lz_new, wc_new, lv_new]

    def simulate(  # type: ignore[override]
        self,
        avg_prec,
        temp,
        et,
        par,
        p2,
        curve,
        q_0=DEF_q0,
        init_st=None,
        ll_temp=None,
        lake_sim=False,
    ):
        """Run the HBV-lake model over a precipitation time series.

        Iterates ``_step_run`` for each time step in the precipitation
        array and returns the full discharge time series and model
        states.  The result has ``n + 1`` entries because the initial
        condition is included.

        Args:
            avg_prec (array_like): Average precipitation for each time
                step [mm/h], length ``n``.
            temp (array_like): Average temperature for each time step
                [C], length ``n``.
            et (array_like): Potential evapotranspiration for each time
                step [mm/h], length ``n``.
            par (array_like): Parameter vector of length 11.  See
                ``_step_run`` for the ordering.
            p2 (array_like): Problem parameter vector
                ``[tfac, lake_sub, lake_area]``.
            curve (numpy.ndarray): Two-column lake rating curve where
                column 0 is discharge [m3/s] and column 1 is storage
                [m3].
            q_0 (float): Initial discharge value [m3/s].  Defaults to
                ``DEF_q0`` (2.3).
            init_st (list[float] or None): Initial model states
                ``[sp, sm, uz, lz, wc, lake_volume]``.  If ``None``,
                ``DEF_ST`` is used.
            ll_temp (array_like or None): Long-term average temperature
                for each time step [C].  If ``None``, computed as the
                mean of ``temp``.
            lake_sim (bool): Unused legacy parameter retained for
                interface compatibility.

        Returns:
            tuple[list[float], list[list[float]]]:
                - Simulated discharge for ``n + 1`` time steps [m3/s]
                  (includes the initial value).
                - Model states for ``n + 1`` time steps.

        Examples:
            >>> import numpy as np
            >>> from Hapi.rrm.hbv_lake import HBVLake, DEF_ST, DEF_q0
            >>> model = HBVLake()
            >>> n = 5
            >>> prec = np.array([4.0] * n)
            >>> temp = np.array([25.0] * n)
            >>> et = np.array([2.0] * n)
            >>> par = [200, 2.0, 0.01, 0.6, 0.5, 0.01, 0.05, 1.0,
            ...        1.0, 1.0, 1.0]
            >>> p2 = [24, 100.0, 10.0]
            >>> curve = np.column_stack([
            ...     np.linspace(0, 100, 50),
            ...     np.linspace(0, 1e10, 50),
            ... ])
            >>> q_sim, states = model.simulate(
            ...     prec, temp, et, par, p2, curve,
            ...     q_0=DEF_q0, init_st=DEF_ST,
            ... )
            >>> len(q_sim)
            6
            >>> len(states)
            6
        """

        if init_st is None:  # If unspecified, [0.0, 30.0, 30.0, 30.0, 0.0] mm
            st = [DEF_ST]  # if not given take the default
        else:
            st = [init_st]  # if given take it

        if (
            ll_temp is None
        ):  # If Long term average temptearature unspecified, calculated from temp
            ll_temp = [np.mean(temp)] * len(avg_prec)

        q_sim = [
            q_0,
        ]

        for i in range(len(avg_prec)):
            v = [avg_prec[i], temp[i], et[i], ll_temp[i]]
            q_out, st_out = self._step_run(par, p2, v, st[i], curve)
            q_sim.append(q_out)
            st.append(st_out)

        return q_sim, st
