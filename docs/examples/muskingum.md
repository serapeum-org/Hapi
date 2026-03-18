# Muskingum
Muskingum is a hydrologic-routing method which employs the equation of continuity to predict magnitude, volume and temporal patterns of flow as it translates downstream of a channel.

.. math::
    𝐼−𝑄 = \frac{𝑑𝑆}{𝑑𝑡}


![muskingum](../img/muskingum1.png)

![muskingum1](../img/muskingum2.png)


Channel routing functions of inflow, outflow and storage where storage can be considered as two parts, prism & wedge storage.

.. math::
    𝑆  = 𝐾∗[𝑥∗𝐼^{\m} +(1−𝑥)∗𝑄^{\𝑚}]


Where `k` is the travel time constant and `x` are weighting coefficient to determine the linearity of the water surface, and it ranges between 0 & 0.5, and `m` is an exponential constant varies from 0.6 for rectangle channel to 1.


For Muskingum version of the channel routing equation `m` equals one which made the relation between `S` and `I`, `Q`. Using coefficient `k` & `x` three weights can be calculated as follow:

.. math::
    C1 = \left(\frac{𝛥𝑡−2𝐾𝑋}{2𝐾(1−𝑋)+𝛥𝑡}\right)\label{eq:C1}
    C2 = \left(\frac{𝛥𝑡+2𝐾𝑋}{2𝐾(1−𝑋)+𝛥𝑡}\right)\label{eq:C2}
    C3 = \left(\frac{2𝐾(1-𝑋)-𝛥𝑡}{2𝐾(1−𝑋)+𝛥𝑡}\right)\label{eq:C3}

To route the inflow hydrograph

.. math::
    Q = \left(C1 * I_{j+1} + C2 * I_{j} + C3 * Q_{j} }\right)\label{eq:Q}