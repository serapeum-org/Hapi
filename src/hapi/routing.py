"""Routing module for hydrograph routing in the Hapi hydrological model.

This module provides channel routing methods used to translate and
attenuate flood hydrographs as they travel through the river network.
It includes:

1. Muskingum-Cunge routing (iterative and vectorized variants).
2. Triangular (MAXBAS) routing using transfer function weights.
"""
from __future__ import annotations

import numpy as np


class Routing:
    """Routing methods for translating and attenuating discharge hydrographs.

    This class provides static methods for two families of routing
    approaches:

    - **Muskingum routing**: classical linear channel routing using the
      Muskingum storage equation, available in both iterative
      (`muskingum`) and vectorized (`muskingum_v`) forms.
    - **Triangular routing**: MAXBAS-based transfer-function routing
      that distributes discharge over a triangular weighting kernel.
      Two variants are provided: `triangular_routing_1` supports
      fractional MAXBAS values; `triangular_routing_2` uses integer
      MAXBAS values.
    """

    def __init__(self):
        """Initialize the Routing instance.

        The Routing class uses only static methods, so no parameters
        are required for instantiation.
        """
        pass

    @staticmethod
    def muskingum(inflow, Qinitial, k, x, dt):
        """Route an inflow hydrograph using the Muskingum method.

        Applies the Muskingum linear storage routing equation to
        translate and attenuate an inflow time series. The three
        Muskingum coefficients (c1, c2, c3) are derived from the
        travel time ``k``, the weighting factor ``x``, and the time
        step ``dt``.

        Args:
            inflow (numpy.ndarray): Time series of inflow discharge
                values.
            Qinitial (float): Initial outflow value at the first time
                step.
            k (float): Channel travel time in the same units as
                ``dt`` (typically hours).
            x (float): Weighting factor for inflow versus storage,
                ranging from 0 (maximum attenuation) to 0.5 (no
                attenuation).
            dt (float): Computational time step in the same units as
                ``k``.

        Returns:
            numpy.ndarray: Routed outflow hydrograph with the same
            length as ``inflow``, rounded to four decimal places.

        Examples:
            >>> import numpy as np
            >>> from hapi.routing import Routing
            >>> inflow = np.array([0, 1, 3, 7, 10, 9, 6, 3, 1, 0])
            >>> q_routed = Routing.muskingum(
            ...     inflow, Qinitial=0, k=2, x=0.2, dt=1
            ... )
        """
        c1 = (dt - 2 * k * x) / (2 * k * (1 - x) + dt)
        c2 = (dt + 2 * k * x) / (2 * k * (1 - x) + dt)
        c3 = (2 * k * (1 - x) - dt) / (2 * k * (1 - x) + dt)

        #    if c1+c2+c3!=1:
        #        raise("sim of c1,c2 & c3 is not 1")

        outflow = np.zeros_like(inflow)
        outflow[0] = Qinitial

        for i in range(1, len(inflow)):
            outflow[i] = c1 * inflow[i] + c2 * inflow[i - 1] + c3 * outflow[i - 1]

        outflow = np.round(outflow, 4)

        return outflow

    @staticmethod
    def muskingum_v(
        inflow: np.ndarray,
        Qinitial: int | float,
        k: int | float,
        x: int | float,
        dt: int | float,
    ) -> np.ndarray:
        """Route an inflow hydrograph using a vectorized Muskingum method.

        This is a performance-optimized variant of `muskingum` that
        pre-computes the c1 and c2 terms in a vectorized manner before
        applying the recursive c3 correction. Negative outflow values
        that would result from the c3 term are suppressed.

        Args:
            inflow (numpy.ndarray): Time series of inflow discharge
                values.
            Qinitial (int | float): Initial outflow value at the
                first time step.
            k (int | float): Channel travel time in the same
                units as ``dt`` (typically hours).
            x (int | float): Weighting factor for inflow versus
                storage, ranging from 0 (maximum attenuation) to 0.5
                (no attenuation).
            dt (int | float): Computational time step in the
                same units as ``k``.

        Returns:
            numpy.ndarray: Routed outflow hydrograph with the same
            length as ``inflow``.

        Examples:
            >>> import numpy as np
            >>> from hapi.routing import Routing
            >>> inflow = np.array([0, 1, 3, 7, 10, 9, 6, 3, 1, 0])
            >>> q_routed = Routing.muskingum_v(
            ...     inflow, Qinitial=0, k=2, x=0.2, dt=1
            ... )
        """
        c1 = (dt - 2 * k * x) / (2 * k * (1 - x) + dt)
        c2 = (dt + 2 * k * x) / (2 * k * (1 - x) + dt)
        c3 = (2 * k * (1 - x) - dt) / (2 * k * (1 - x) + dt)

        #    if c1+c2+c3!=1:
        #        raise("sim of c1,c2 & c3 is not 1")

        Q = np.zeros_like(inflow)
        Q[0] = Qinitial
        Q[1:] = c1 * np.asarray(inflow[1:]) + c2 * np.asarray(inflow[0:-1])

        for i in range(1, len(inflow)):
            # only if the
            if not Q[i] + c3 * Q[i - 1] < 0:
                Q[i] = Q[i] + c3 * Q[i - 1]

        return Q

    @staticmethod
    def tf(maxbas):
        """Generate triangular transfer-function weights.

        Builds a normalized weight array shaped as a triangle with a
        rising limb for the first half and a falling limb for the
        second half. The weights sum to 1 and are used by
        `triangular_routing_2` to distribute discharge across
        ``maxbas`` time steps.

        Args:
            maxbas (int): Number of time steps over which to spread
                the discharge. Must be >= 1.

        Returns:
            numpy.ndarray: Array of normalized weights with length
            ``maxbas`` that sum to 1.0.

        Examples:
            >>> from hapi.routing import Routing
            >>> weights = Routing.tf(5)
            >>> print(weights.sum())
            1.0
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

    @staticmethod
    def triangular_routing_2(q, maxbas=1):
        """Route discharge using a triangular transfer function (integer MAXBAS).

        Convolves the input discharge time series with a triangular
        weighting kernel whose width is determined by ``maxbas``. Only
        integer values of ``maxbas`` are supported; the value is
        rounded to the nearest integer internally. Weights are
        generated by `tf`.

        Args:
            q (numpy.ndarray): Time series of discharge values to be
                routed.
            maxbas (int): Number of time steps for the triangular
                routing kernel. Must be >= 1. Defaults to 1.

        Returns:
            numpy.ndarray: Routed discharge time series with the same
            length as ``q``.

        Raises:
            AssertionError: If ``maxbas`` is less than 1.

        Examples:
            >>> import numpy as np
            >>> from hapi.routing import Routing
            >>> q = np.array([0.0, 1.0, 3.0, 7.0, 10.0, 9.0, 6.0])
            >>> q_routed = Routing.triangular_routing_2(q, maxbas=3)
        """
        # input data validation
        assert maxbas >= 1, "Maxbas value has to be larger than 1"

        # Get integer part of maxbas
        maxbas = int(round(maxbas, 0))

        # get the weights
        w = Routing.tf(maxbas)

        # rout the discharge signal
        q_r = np.zeros_like(q, dtype="float64")
        q_temp = np.float32(q)
        for w_i in w:
            q_r += q_temp * w_i
            q_temp = np.insert(q_temp, 0, 0.0)[:-1]  # type: ignore[assignment]

        return q_r

    @staticmethod
    def calculate_weights(maxbas):
        """Calculate triangular routing weights for a given MAXBAS value.

        Computes normalized weights based on the area under an
        equilateral-triangle transfer function. Unlike `tf`, this
        method supports fractional (non-integer) MAXBAS values by
        computing exact trapezoidal areas under the triangle curve.

        Args:
            maxbas (float): The MAXBAS routing parameter controlling
                the number of time steps over which discharge is
                distributed. Can be an integer or a decimal value.

        Returns:
            numpy.ndarray: Array of normalized routing weights. The
            length is ``floor(MAXBAS)`` for integer values, or
            ``floor(MAXBAS) + 1`` for non-integer values.

        Examples:
            >>> from hapi.routing import Routing
            >>> weights = Routing.calculate_weights(5)
            >>> print(weights)
            [0.08 0.24 0.36 0.24 0.08]
        """
        yant = 0
        total = 0  # Just to verify how far from the unit is the result

        total_area = (maxbas * maxbas * np.sin(np.pi / 3)) / 2
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
                # Integral of x dx with slope of 60 degree Equilateral triangle
                ynow = np.tan(np.pi / 3) * (x + 1)
                # Area / total_area
                maxbas_w[x] = ((ynow + yant) / 2) / total_area
            else:  # The area here is calculated by the formlua of a trapezoidal (B1+B2)*h /2
                if flag == 1:
                    ynow = np.sin(np.pi / 3) * maxbas
                    if peak_point == 0:
                        maxbas_w[x] = ((ynow + yant) / 2) / total_area
                    else:
                        a1 = ((ynow + yant) / 2) * (maxbas / 2.0 - x) / total_area
                        yant = ynow
                        ynow = (maxbas * np.sin(np.pi / 3)) - (
                            np.tan(np.pi / 3) * (x + 1 - maxbas / 2.0)
                        )
                        a2 = ((ynow + yant) * (x + 1 - maxbas / 2.0) / 2) / total_area
                        maxbas_w[x] = a1 + a2

                    flag = 2
                else:
                    # 'sum of the two height in the descending part of the triangle
                    ynow = maxbas * np.sin(np.pi / 3) - np.tan(np.pi / 3) * (x + 1 - maxbas / 2.0)
                    # Multiplying by the height of the trapezoidal and dividing by 2
                    maxbas_w[x] = ((ynow + yant) / 2) / total_area

            total = total + maxbas_w[x]
            yant = ynow

        x = int(maxbas)
        # x = x + 1

        if real_part > 0:
            if np.floor(maxbas) == 0:
                maxbas = 1
                maxbas_w[x] = 1
                # NumberofWeights = 1
            else:
                maxbas_w[x] = (yant * (maxbas - (x)) / 2) / total_area
                total = total + maxbas_w[x]
                # NumberofWeights = x
        else:
            # NumberofWeights = x - 1
            pass

        return maxbas_w

    @staticmethod
    def triangular_routing_1(Q, MAXBAS):
        """Route discharge using triangular weights (fractional MAXBAS).

        Distributes the input hydrograph over time using MAXBAS
        triangular weights computed by `calculate_weights`. Unlike
        `triangular_routing_2`, this method supports fractional
        (non-integer) MAXBAS values.

        The routing is performed by constructing a weighted discharge
        matrix and summing along the anti-diagonals to produce the
        output hydrograph.

        Args:
            Q (numpy.ndarray): Input discharge time series to be
                routed.
            MAXBAS (float): The MAXBAS routing parameter. Can be an
                integer or a decimal value. Controls the number of
                time steps over which the discharge is spread.

        Returns:
            numpy.ndarray: Routed output hydrograph with the same
            length as ``Q``.

        Examples:
            >>> import numpy as np
            >>> from hapi.routing import Routing
            >>> Q = np.array([0.0, 1.0, 3.0, 7.0, 10.0, 9.0, 6.0])
            >>> q_out = Routing.triangular_routing_1(Q, MAXBAS=5)
        """
        # CALCULATE MAXBAS WEIGHTS
        maxbasW = Routing.calculate_weights(MAXBAS)

        Qw = np.ones((len(Q), len(maxbasW)))
        # Calculate the matrix discharge
        for i in range(len(Q)):  # 0 to 10
            for k in range(len(maxbasW)):  # 0 to 4
                Qw[i, k] = maxbasW[k] * Q[i]

        def mm(A, s):
            tot = []
            for o in range(np.shape(A)[1]):  # columns
                for t in range(np.shape(A)[0]):  # rows
                    tot.append(A[t, o])
            Su = tot[s:-1:s]
            return Su

        # Calculate routing
        j = 0
        Qout = np.ones(shape=(len(Q)))

        for i in range(len(Q)):
            if i == 0:
                Qout[i] = Qw[i, i]
            elif i < len(maxbasW) - 1:
                A = Qw[0 : i + 1, :]
                s = len(A) - 1  # len(A) is the no of rows or use int(np.shape(A)[0])
                Su = mm(A, s)

                Qout[i] = sum(Su[0 : i + 1])
            else:
                A = Qw[j : i + 1, :]
                s = len(A) - 1
                Su = mm(A, s)
                Qout[i] = sum(Su)
                j = j + 1

        return Qout  # ,maxbasW
