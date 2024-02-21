"""
This will perform the RatSLAM algorithm on the data specified in options.py
"""

# -----------------------------------------------------------------------

import argparse

import imageio
import matplotlib.pyplot as plt

from ratSLAM.data_simulation import generate_dummy_dataset
from ratSLAM.ratSLAM import RatSLAM
from ratSLAM.utilities import showTiming
from utils.get_MJ_dataset import get_mj_dataset
from utils.logger import logger
from utils.use_DI_model import MODEL

# ------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5file", required=True, help="h5 file to run brain on")
    parser.add_argument(
        "--model_path", type=str, required=True, help="The path of the model to use"
    )
    args = parser.parse_args()

    logger.debug("Starting RatSLAM...")

    MODEL.setup_model(args.h5file, args.model_path)
    slam = RatSLAM(absolute_rot=True)

    data = get_mj_dataset(args.h5file)
    x = []
    y = []
    # ..todo: Fix gif writer for windows
    with imageio.get_writer("BrainSLAM_windows.gif", mode="I") as writer:
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
