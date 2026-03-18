"""Lumped Conceptual HBV model.

HBV is lumped conceptual model consists of precipitation, snow melt,
soil moisture and response subroutine to convert precipitation into o runoff,
where state variables are updated each time step to represent a specific
hydrologic behaviour the catchment

This version was edited based on a Master Thesis on "Spatio-temporal simulation
of catchment response based on dynamic weighting of hydrological models" on april 2018

- Model inputs are Precipitation, evapotranspiration and temperature, initial
    state variables, and initial discharge.
- Model output is Qalculated dicharge at time t+1
- Model equations are solved using explicit scheme
- model structure uses 18 parameters if the catchment has snow
    [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
    c_flux, k, k1, alpha, perc]

    otherwise it uses 10 parameters
    [rfcf, fc, beta, etf, lp, c_flux, k, k1, alpha, perc]
"""
from typing import Tuple
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

    @staticmethod
    def precipitation(temp, ltt, utt, prec, rfcf, sfcf):
        """Precipitation routine of the HBV96 model.

        If temperature is lower than ltt, all the precipitation is considered as
        snow. If the temperature is higher than utt, all the precipitation is
        considered as rainfall. In case that the temperature is between ltt and
        utt, precipitation is a linear mix of rainfall and snowfall.

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
    def snow(cfmax, temp, ttm, cfr, cwh, rf, sf, wc_old, sp_old) -> Tuple[float, float, float]:
        """Snow routine of the HBV-96 model.

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
            Day degree factor
        temp : float
            Temperature [C]
        ttm : float
            Temperature treshold for Melting [C]
        cfr : float
            Refreezing factor
        cwh : float
            Capacity for water holding in snow pack
        rf : float
            Rainfall [mm]
        sf : float
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
    def soil(fc, beta, etf, temp, tm, e_corr, lp, c_flux, inf, ep, sm_old, uz_old) -> Tuple[float, float]:  # tfac,
        """Soil routine of the HBV-96 model.

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
        etf : float
            Total potential evapotranspiration
        temp : float
            Temperature
        tm : float
            Average long term temperature
        e_corr : float
            Evapotranspiration corrector factor
        lp : float
            wilting point
        c_flux : float
            Capilar flux in the root zone
        inf : float
            actual infiltration
        ep : float
            actual evapotranspiration
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
    def response(perc, alpha, k, k1, lz_old, uz_int_1) -> Tuple[float, float, float, float]:  # tfac,area,
        r"""The response routine of the HBV-96 model.

        The response routine is in charge of transforming the current values of
        upper and lower zone into discharge. This routine also controls the
        recharge of the lower zone tank (baseflow). The transformation of units
        also occurs in this point.

        Parameters
        ----------
        perc : [float]
            Percolation value [mm\hr]
        alpha : float
            Response box parameter
        k : float
            Upper zone recession coefficient
        k1 : float
            Lower zone recession coefficient
        lz_old : float
            Previous lower zone value [mm]
        uz_int_1 : float
            Previous upper zone value before percolation [mm]
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
    def tf(maxbas) -> Tuple[float]:
        """Transfer function weight generator."""
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
        return wi


    def routing(self, q, maxbas=1):
        """Routing This function implements the transfer function using a triangular function."""
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
        """Step run.

            The function makes the calculation of next step of discharge and states.

        Parameters
        ----------
        p : array_like [18]
            Parameter vector, set up as:
            [ltt, utt, ttm, cfmax, fc, ecorr, etf, lp, k, k1,
            alpha, beta, cwh, cfr, c_flux, perc, rfcf, sfcf]
        v : array_like [4]
            Input vector setup as:
            [prec, temp, evap, llt]
        state_variable : array_like [5]
            Previous model states setup as:
            [sp, sm, uz, lz, wc]
        snow:
            1 to run the snow subroutine and 0 for not. Default is 0.

        Returns
        -------
        q_new : float
            Discharge [m3/s]
        St : array_like [5]
            Posterior model states
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
            e_corr = 1  # p[2]
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

    def simulate(self, prec, temp, et, par, init_st=None, ll_temp=None, q_init=None, snow=0):
        """Simulate Run the HBV model for the number of steps (n) in precipitation. The resluts are (n+1) simulation of discharge as the model calculates step n+1.

        Parameters
        ----------
        prec : array_like [n]
            Average precipitation [mm/h]
        temp : array_like [n]
            Average temperature [C]
        et : array_like [n]
            Potential Evapotranspiration [mm/h]
        par : array_like [18]
            Parameter vector, set up as:
            [ltt, utt, ttm, cfmax, fc, ecorr, etf, lp, k, k1,
            alpha, beta, cwh, cfr, c_flux, perc, rfcf, sfcf]
        init_st : array_like [5], optional
            Initial model states, [sp, sm, uz, lz, wc]. If unspecified,
            [0.0, 30.0, 30.0, 30.0, 0.0] mm
        ll_temp : array_like [n], optional
            Long term average temptearature. If unspecified, calculated from temp.
        q_init : float, optional
            Initial discharge value. If unspecified set to 10.0

        Returns
        -------
        q_sim : array_like [n]
            Discharge for the n time steps of the precipitation vector [m3/s]
        st : array_like [n, 5]
            Model states for the complete time series [mm]
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
            q_uzi, q_lzi, st_out = self.step_run(par, v, st[i], snow=0)  # p2,
            #        q_sim.append(q_out)
            q_uz.append(q_uzi)
            q_lz.append(q_lzi)
            st.append(st_out)
        #        uz_int_2.append(uz_int_2_out) # upper zone - perc
        #        lz_int_1.append(lz_int_1_out) # lower zone + perc

        return np.float32(q_uz), np.float32(q_lz), np.float32(st)
