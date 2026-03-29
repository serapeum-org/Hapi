# Muskingum

Muskingum is a hydrologic-routing method which employs the equation
of continuity to predict magnitude, volume and temporal patterns of
flow as it translates downstream of a channel.

$$
I - Q = \frac{dS}{dt}
$$

![muskingum](../img/muskingum1.png)

![muskingum1](../img/muskingum2.png)

Channel routing functions of inflow, outflow and storage where
storage can be considered as two parts, prism & wedge storage.

$$
S = K \cdot [x \cdot I^{m} + (1 - x) \cdot Q^{m}]
$$

Where `k` is the travel time constant and `x` are weighting
coefficient to determine the linearity of the water surface, and it
ranges between 0 & 0.5, and `m` is an exponential constant varies
from 0.6 for rectangle channel to 1.

For Muskingum version of the channel routing equation `m` equals one
which made the relation between `S` and `I`, `Q`. Using coefficient
`k` & `x` three weights can be calculated as follow:

$$
C_1 = \frac{\Delta t - 2KX}{2K(1 - X) + \Delta t}
$$

$$
C_2 = \frac{\Delta t + 2KX}{2K(1 - X) + \Delta t}
$$

$$
C_3 = \frac{2K(1 - X) - \Delta t}{2K(1 - X) + \Delta t}
$$

To route the inflow hydrograph:

$$
Q = C_1 \cdot I_{j+1} + C_2 \cdot I_{j} + C_3 \cdot Q_{j}
$$
