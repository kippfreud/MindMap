"""
This will perform the RatSLAM algorithm on the data specified in options.py
"""

# -----------------------------------------------------------------------

import imageio
import matplotlib.pyplot as plt

from ratSLAM.data_simulation import generate_dummy_dataset
from ratSLAM.ratSLAM import RatSLAM
from ratSLAM.utilities import showTiming
from utils.get_MJ_dataset import get_mj_dataset
from utils.logger import root_logger

# ------------------------------------------------------------------------

if __name__ == "__main__":

    root_logger.debug("Starting RatSLAM...")

    slam = RatSLAM(absolute_rot=True)

    data = get_mj_dataset()
    x = []
    y = []
    with imageio.get_writer("NeuroSLAM-Full.gif", mode="I") as writer:
        for i, d in enumerate(data):
            if i < 1:
                continue
            if i > 750:
                break
            slam.step(d)
            if i % 1 == 0 and i > 0:
                slam.experience_map.plot(writer)
    plt.scatter(x, y)
    plt.show()
    showTiming()
    exit(0)
