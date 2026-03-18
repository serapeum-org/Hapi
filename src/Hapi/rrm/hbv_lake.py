"""HBV model with a lake function."""

from __future__ import division, print_function

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interp11

from Hapi.rrm.base_model import BaseConceptualModel

# initial values for state variables
# [sp, sm, uz, lz, wc]
DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0, 10.163 * 10**9]
# initial value for discarge
DEF_q0 = 2.3  # 10.0


class HBVLake(BaseConceptualModel):

    @staticmethod
    def _precipitation(temp, ltt, utt, prec, rfcf, sfcf, tfac, pcorr):
        """Precipitation.

            Precipitaiton routine of the HBV96 model.
            If temperature is lower than ltt, all the precipitation is considered as
            snow. If the temperature is higher than utt, all the precipitation is
            considered as rainfall. In case that the temperature is between ltt and
            utt, precipitation is a linear mix of rainfall and snowfall.

            inputs (precipitation, Temperature)
            outputs: (rainfall, snowfall)

        Parameters
        ----------
        temp : float
            Measured temperature [C]
        ltt : float
            Lower temperature treshold [C]
        utt : float
            Upper temperature treshold [C]
        prec : float
            Precipitation [mm]
        rfcf : float
            Rainfall corrector factor
        sfcf : float
            Snowfall corrector factor
        Returns
        -------
        _rf : float
            Rainfall [mm]
        _sf : float
            Snowfall [mm]
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
        """Snow.

            Snow routine of the HBV-96 model.

            The snow pack consists of two states: Water Content (wc) and Snow Pack
            (sp). The first corresponds to the liquid part of the water in the snow,
            while the latter corresponds to the solid part. If the temperature is
            higher than the melting point, the snow pack will melt and the solid snow
            will become liquid. In the opposite case, the liquid part of the snow will
            refreeze, and turn into solid. The water that cannot be stored by the solid
            part of the snow pack will drain into the soil as part of infiltration.

        Parameters
        ----------
        cfmax : float
            Day degree factor   # melting factor
        tfac : float
            Temperature correction factor
        temp : float
            Temperature [C]
        ttm : float
            Temperature treshold for Melting [C]
        cfr : float
            Refreezing factor
        cwh : float
            Capacity for water holding in snow pack
            water-holding capacity of snow (meltwater is retained in snowpack until it exceeds the WHC)
        _rf : float
            Rainfall [mm]
        _sf : float
            Snowfall [mm]
        wc_old : float
            Water content in previous state [mm]
        sp_old : float
            snow pack in previous state [mm]

        Returns
        -------
        _in : float
            Infiltration [mm]
        _wc_new : float
            Water content in posterior state [mm]
        _sp_new : float
            Snowpack in posterior state [mm]
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
        """Soil. Soil routine of the HBV-96 model.

            The model checks for the amount of water that can infiltrate the soil,
            coming from the liquid precipitation and the snow pack melting. A part of
            the water will be stored as soil moisture, while other will become runoff,
            and routed to the upper zone tank.

        Parameters
        ----------
        fc : float
            Filed capacity
        beta : float
            Shape coefficient for effective precipitation separation
            Controls the contribution of the increase in the soil moisture or
            to the response function
        etf : float
            Total potential evapotranspiration
        temp : float
            Temperature
        tm : float
            Average long term temperature
        e_corr : float
            Evapotranspiration corrector factor
        lp : float _soil

        tfac : float
            Time conversion factor
        c_flux : float
            Capilar flux in the root zone
        _in : float
            actual infiltration
        ep : float
            Long term mean potential evapotranspiration
        sm_old : float
            Previous soil moisture value
        uz_old : float
            Previous Upper zone value
        Returns
        -------
        sm_new : float
            New value of soil moisture
        uz_int_1 : float
            New value of direct runoff into upper zone

        variables
        -------
        EPInt  :
            Adjusted potential evapotranspiration
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
        r"""Response.

            The response routine of the HBV-96 model.
            The response routine is in charge of transforming the current values of
            upper and lower zone into discharge. This routine also controls the
            recharge of the lower zone tank (baseflow). The transformation of units
            also occurs in this point.

        Parameters
        ----------
        tfac : float
            Number of hours in the time step
        perc : float
            Percolation value [mm\hr]
        alpha : float
            Response box parameter
            upper zone runoff coefficient
        k : float
            Upper zone recession coefficient
            Upper zone response coefficient
        k1 : float
            Lower zone recession coefficient
            Lowe zone response coefficient
        area : float
            Catchment area [Km2]
        lz_old : float
            Previous lower zone value [mm]
        uz_int_1 : float
            Previous upper zone value before percolation [mm]
        qdr : float
            Direct runoff [mm]
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
        """lake. lake model.

        Parameters
        ----------
        temp
        curve
        tfac
        rf
        sf
        q_new
        lv_old
        ltt
        c_le
        ep
        lakeA
        """
        # lower zone
        # explicit representation of the lake where lake will be represented by a rating curve
        """
        l_ea :lake evaporation
        c_le : lake _evaporation correction factor

        """
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
        """tf.

        Transfer function weight generator in a shape of a triangle.
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
        """routing.

        This function implements the transfer function using a triangular
        function.
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
        """calculate_max_bas.

        The calculate_max_bas function calculate the maxbas Weights based on a MAXBAX number The maxbas is a HBV
        parameter that controls the routing.

        Parameters
        ----------
        maxbas: [int]
            maxbas weight

        Example
        -------
        maxbasW = calculate_max_bas(5)
        maxbasW =

            0.0800    0.2400    0.3600    0.2400    0.0800

        Notes
        -----
        - It is important to mention that this function allows one to obtain weights not only for interger values but
        from decimal values as well.
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
        """routing_maxbas.

            This function calculate the routing from a input hydrograph using the maxbas parameter from the HBV model.

        Parameters
        ----------
        q: [array]
            input hydrograph
        maxbas: []
            maxbas weight

        Returns
        -------
        Qout:
            output hydrograph
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
        """Step run. Makes the calculation of next step of discharge and states.

        Parameters
        ----------
        parameters : array_like [10]
            Parameter vector, set up as:
            [fc, beta, e_corr, etf, lp, c_flux, k, k1, alpha, perc]
            9 fixed parameters + 9 optimized parameters + 1 parameter for the lake
        catchment_parameters : array_like [3]
            Problem parameter vector setup as:
            [tfac, area, Lake_area]
        v : array_like [4]
            Input vector setup as:
            [prec, temp, evap, llt]
        state_variables : array_like [5]
            Previous model states setup as:
            [sp, sm, uz, lz, wc]

        Returns
        -------
        q_new : float
            Discharge [m3/s]
        St : array_like [5]
            Posterior model states
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

    def simulate(
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
        """Simulate.

            Run the HBV model for the number of steps (n) in precipitation. The
            resluts are (n+1) simulation of discharge as the model calculates step n+1

        Parameters
        ----------
        avg_prec : array_like [n]
            Average precipitation [mm/h]
        temp : array_like [n]
            Average temperature [C]
        et : array_like [n]
            Potential Evapotranspiration [mm/h]
        par : array_like [10]
            Parameter vector, set up as:
            [fc, beta, e_corr, etf, lp, c_flux, k, k1, alpha, perc]
        p2 : array_like [3]
            Problem parameter vector setup as:
            [tfac, area, Lake_area]
        init_st : array_like [5], optional
            Initial model states, [sp, sm, uz, lz, wc]. If unspecified,
            [0.0, 30.0, 30.0, 30.0, 0.0] mm
        ll_temp : array_like [n], optional
            Long term average temptearature. If unspecified, calculated from temp.
        q_0 : float, optional
            Initial discharge value. If unspecified set to 10.0

        Returns
        -------
        q_sim : array_like [n]
            Discharge for the n time steps of the precipitation vector [m3/s]
        st : array_like [n, 5]
            Model states for the complete time series [mm]
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
