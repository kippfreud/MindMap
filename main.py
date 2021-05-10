"""
This will perform the RatSLAM algorithm on the video file given in ...

"""

# -----------------------------------------------------------------------

from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

from ratSLAM.data_simulation import generate_dummy_dataset
from ratSLAM.ratSLAM import RatSLAM
from ratSLAM.input import DummyInput
from ratSLAM.utilities import showTiming
from utils.get_MJ_dataset import get_mj_dataset
from utils.logger import root_logger

#------------------------------------------------------------------------

if __name__ == "__main__":

    root_logger.debug("Starting RatSLAM...")

    slam = RatSLAM(absolute_rot=True)

    data = get_mj_dataset()
    # data = generate_dummy_dataset(n_loops=4,
    #                               steps_per_loops=54,
    #                               noise_rots=1/10,
    #                               noise_trans=0.,#1/100,
    #                               noise_templates=10,
    #                               len_data_templates=20,
    #                               template_mean=10000,
    #                               template_sd=10,
    #                               step_len=0.00001
    #                               )
    x = []
    y = []
    with imageio.get_writer('NeuroSLAM-Full.gif', mode='I') as writer:
        for i, d in enumerate(data):
            print(i)
            if i > 1000:
                break
            slam.step(d)
            if i%1 == 0 and i>0:
                slam.experience_map.plot(writer)
    plt.scatter(x,y)
    plt.show()
    showTiming()
    print("done")
