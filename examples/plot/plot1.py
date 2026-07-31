"""Created on Sat Apr 17 20:16:53 2021.

@author: mofarrag
"""

# import matplotlib.pyplot as plt
import numpy as np
from Hapi.visualizer import Visualize

distance = [
    19.5,
    43.0,
    71.0,
    123.5,
    164.5,
    204.5,
    212.5,
    222.0,
    251.0,
    343.0,
    483.5,
    679.0,
    921.5,
]

wl1 = [
    136.82,
    132.7,
    123.05,
    115.27,
    101.56,
    81.12,
    68.76,
    66.36,
    65.84,
    50.06,
    39.65,
    29.59,
    22.66,
]
wl1 = np.transpose([distance, wl1])

wl2 = [
    134.64,
    134.11,
    120.87,
    115.89,
    94.8,
    79.13,
    67.85,
    65.63,
    64.61,
    46.64,
    37.93,
    27.99,
    22.02,
]
wl2 = np.transpose([distance, wl2])

diff = np.random.uniform(-0.5, 0.5, size=len(wl1))
diff = np.transpose([distance, diff])

diff2 = np.random.uniform(-0.5, 0.5, size=len(wl1))
diff2 = np.transpose([distance, diff2])

OT1 = np.random.uniform(5000, 20000, size=len(wl1))
OT1 = np.transpose([distance, OT1])

OT2 = np.random.uniform(7000, 16000, size=len(wl1))
OT2 = np.transpose([distance, OT2])
# %%
Y1_2 = wl2

Y2_2 = diff2

# label = ['YYY']
"you have to include the x-axis as first column in the array"
Points1 = OT2

"at which Y value the points are going to be plotted"
PointsY = 13
PointsY1 = [15]

Visualize.Plot_Type1(
    wl1,
    diff,
    OT1,
    PointsY,
    PointMaxSize=200,
    PointMinSize=1,
    X_axis_label="Distance",
    LegendNum=5,
    LegendLoc=(1.3, 1),
    PointLegendTitle="Output 2",
    Ylim=[0, 180],
    Y2lim=[-2, 16],
    color1="#27408B",
    color2="#DC143C",
    color3="grey",
    linewidth=4,
    Y1_2=Y1_2,
    Y2_2=Y2_2,
    Points1=Points1,
    PointsY1=PointsY1,
)
