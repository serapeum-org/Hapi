"""Lumped Conceptual HBV model.

HBV is lumped conceptual model consists of precipitation, snow melt,
soil moisture and response subroutine to convert precipitation into o runoff,
where state variables are updated each time step to represent a specific
hydrologic behaviour the catchment

This version was edited based on a Master Thesis on "Spatio-temporal simulation
of catchment response based on dynamic weighting of hydrological models" on april 2018

- Main Model inputs are Precipitation, evapotranspiration and temperature,
  conversion vector P2 = [time_conversion, catchment_area], parameters.
- Optional model inputs are: initial state variables, initial discharge
  Long term average temptearature .

- Model output is Qalculated dicharge at time t+1
- Model equations are solved using explicit scheme


Parameters
----------
precipitation : array_like [n]
    Average precipitation [mm/h]
evapotranspitration : array_like [n]
    Potential Evapotranspiration [mm/h]
temperature : array_like [n]
    Average temperature [C]
init_st : array_like [5], optional
    Initial model states, [sp, sm, uz, lz, wc]. If unspecified,
    [0.0, 30.0, 30.0, 30.0, 0.0] mm
q_init : float, optional
    Initial discharge value. If unspecified set to 10.0
ll_temp : array_like [n], optional
    Long term average temptearature. If unspecified, calculated from temp.
p2 : array_like [2]
    Problem parameter vector setup as:
    [tfac, area]
par : array_like [10]
    Parameter vector, set up as:
    [fc, beta, e_corr, etf, lp, c_flux, k, k1, alpha, perc]

Returns
-------
q_sim : array_like [n]
    Discharge for the n time steps of the precipitation vector [m3/s]
st : array_like [n, 5]
    Model states for the complete time series [mm]

- model structure uses 18 parameters if the catchment has snow
    [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
    c_flux, k, k1, alpha, perc]

    otherwise it uses 10 parameters
    [rfcf, fc, beta, etf, lp, c_flux, k, k1, alpha, perc]


Parameters
----------
ltt : float
    Lower temperature treshold [C]
utt : float
    Upper temperature treshold [C]
rfcf : float
    Rainfall corrector factor
sfcf : float
    Snowfall corrector factor
ttm : float
    Temperature treshold for Melting [C]
cfmax : float
    Day degree factor
cwh : float
    Capacity for water holding in snow pack
cfr : float
    Refreezing factor
fc : float
    Filed capacity
beta : float
    Shape coefficient for effective precipitation separation
e_corr : float
    Evapotranspiration corrector factor
etf : float
    Total potential evapotranspiration
lp : float _soil
    wilting point
c_flux : float
    Capilar flux in the root zone
k : float
    Upper zone recession coefficient
    Upper zone response coefficient
k1 : float
    Lower zone recession coefficient
    Lowe zone response coefficient
alpha : float
    Response box parameter
    upper zone runoff coefficient
perc: float
    percolation
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interp11

# import sklearn.metrics as error

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


def Precipitation(temp, ltt, utt, prec, rfcf, sfcf, tfac):
    """Precipitation Precipitaiton routine of the HBV96 model.

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


def Snow(cfmax, tfac, temp, ttm, cfr, cwh, rf, sf, wc_old, sp_old):
    """Snow Snow routine of the HBV-96 model.

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


def Soil(fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux, inf, ep, sm_old, uz_old):
    """Soil Soil routine of the HBV-96 model.

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
    lp : float _soil
        wilting point
    tfac : float
        Time conversion factor
    c_flux : float
        Capilar flux in the root zone
    _in : float
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


def Response(tfac, perc, alpha, k, k1, area, lz_old, uz_int_1):
    r"""Response The response routine of the HBV-96 model.

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

    uz_new = uz_int_2 - (_q_0)

    lz_int_1 = lz_old + np.min(
        [perc, uz_int_1]
    )  # if the percolation > upper zone Q all the Quz will percolate

    _q_1 = k1 * lz_int_1

    if _q_1 > lz_int_1:
        _q_1 = lz_int_1

    lz_new = lz_int_1 - (_q_1)

    q_new = ((_q_0 + _q_1) * area) / (
        3.6 * tfac
    )  # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
    # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
    return q_new, uz_new, lz_new


def Lake(temp, curve, tfac, rf, sf, q_new, lv_old, ltt, c_le, ep, lakeA):
    """Lake subroutine.

        lower zone
        explicit representation of the lake where lake will be represented by a rating curve

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


def Step_run(p, p2, v, St, curve, lake_sim, snow=0):
    """Step run.

    Makes the calculation of next step of discharge and states

    Parameters
    ----------
    p : array_like [10]
        Parameter vector, set up as:
        [fc, beta, e_corr, etf, lp, c_flux, k, k1, alpha, perc]
        9 fixed parameters + 9 optimized parameters + 1 parameter for the lake
    p2 : array_like [3]
        Problem parameter vector setup as:
        [tfac, area, Lake_area]
    v : array_like [4]
        Input vector setup as:
        [prec, temp, evap, llt]
    St : array_like [5]
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
    # [tfac,main_catchment_area,lake_sub,lake_area]
    tfac = p2[0]

    ## Parse of Inputs
    prec = v[0]  # Precipitation [mm]
    temp = v[1]  # Temperature [C]
    ep = v[2]
    tm = v[3]  # Long term (monthly) average temperature [C]

    ## Parse of states
    sp_old = St[0]
    sm_old = St[1]
    uz_old = St[2]
    lz_old = St[3]
    wc_old = St[4]

    if lake_sim:
        area = p2[2]  # area of lake subcatchment
        lakeA = p2[3]  # area of the lake
        c_le = p[9]
        lv_old = St[5]
    else:
        area = p2[1]

    rf, sf = Precipitation(temp, ltt, utt, prec, rfcf, sfcf, tfac)
    inf, wc_new, sp_new = Snow(cfmax, tfac, temp, ttm, cfr, cwh, rf, sf, wc_old, sp_old)
    sm_new, uz_int_1 = Soil(
        fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux, inf, ep, sm_old, uz_old
    )
    q_new, uz_new, lz_new = Response(tfac, perc, alpha, k, k1, area, lz_old, uz_int_1)

    # if lake_sim is true it will enter the function of the lake
    if lake_sim:
        qout, lv_new = Lake(
            temp, curve, tfac, rf, sf, q_new, lv_old, ltt, c_le, ep, lakeA
        )
    else:  # if lake_sim is false it will enter the function of the lake
        qout = q_new
        lv_new = 0

    return qout, [sp_new, sm_new, uz_new, lz_new, wc_new, lv_new]


def Simulate(
    prec,
    temp,
    et,
    par,
    p2,
    init_st=None,
    ll_temp=None,
    q_init=None,
    snow=0,
    lake_sim=False,
    curve=None,
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
    ### inputs validation
    # data type
    assert len(init_st) == 5, (
        "state variables are 5 and the given initial values are " + str(len(init_st))
    )
    assert type(p2) == list, " p2 should be of type list"
    assert len(p2) == 2, "p2 should contains tfac and catchment area"
    assert snow == 0 or snow == 1, (
        " snow input defines whether to consider snow subroutine or not it has to be 0 or 1"
    )

    if init_st is None:  # If unspecified, [0.0, 30.0, 30.0, 30.0, 0.0] mm
        st = [DEF_ST]  # if not given take the default
    else:
        st = [init_st]  # if given take it

    if (
        ll_temp is None
    ):  # If Long term average temptearature unspecified, calculated from temp
        ll_temp = [np.mean(temp)] * len(prec)

    if q_init is None:
        q_sim = [DEF_q0]
    else:
        q_sim = [q_init]

    # run the step by step function
    for i in range(len(prec)):
        v = [prec[i], temp[i], et[i], ll_temp[i]]
        q_out, st_out = Step_run(par, p2, v, st[i], curve, lake_sim, snow=0)
        q_sim.append(q_out)
        st.append(st_out)

    return q_sim, st
