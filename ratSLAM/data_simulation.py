"""
This will generate a DummyInput dataset of a rat travelling in a perfect square.
Given noise parameters inform how noisy odometry readings will be.

"""

# -----------------------------------------------------------------------

import cv2
import numpy as np

from ratSLAM.ratSLAM import RatSLAM
from ratSLAM.input import DummyInput
from ratSLAM.utilities import showTiming

#------------------------------------------------------------------------


def generate_dummy_dataset(n_loops=5,
                           steps_per_loops=54,
                           noise_rots=1/100,
                           noise_trans=1/100,
                           noise_templates=1/10,
                           len_data_templates=20,
                           template_mean=10000,
                           template_sd=10,
                           step_len=0.1
                           ):
    """
    Generates a DummyInput dataset of a rat travelling in a perfect square.
    """
    tot_len = n_loops*steps_per_loops
    templates = []
    for i in range(steps_per_loops):
        templates.append(np.random.normal(template_mean,template_sd,len_data_templates))
    visuals = templates * n_loops
    trans = [step_len]*tot_len
    rots = ([0.]*int(round((steps_per_loops-4)/4)) + [np.pi/2]) * 3
    for i in range(steps_per_loops - len(rots)):
        rots.append(0.)
    rots[-1] = np.pi/2
    rots = rots * n_loops
    print(len(rots)==len(trans))
    print(len(rots)==len(visuals))
    for i in range(len(rots)):
        rots[i] += np.random.randn(1)[0]*noise_rots
    for i in range(len(trans)):
        trans[i] += np.random.randn(1)[0]*noise_trans
    for i in range(len(templates)):
        templates[i] += np.random.normal(0,noise_templates,len_data_templates)
    data = []
    for i in range(len(rots)):
        app = (visuals[i], (trans[i], rots[i]))
        data.append(DummyInput(app))
    return data
