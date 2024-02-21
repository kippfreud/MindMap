"""
Class containing the main ratSLAM class
"""

# -----------------------------------------------------------------------

import numpy as np

from ratSLAM.experience_map import ExperienceMap
from ratSLAM.input import Input
from ratSLAM.odometry import Odometry
from ratSLAM.pose_cells import PoseCells
from ratSLAM.utilities import timethis
from ratSLAM.view_cells import ViewCells
from utils.logger import logger

# -----------------------------------------------------------------------

X_DIM = 8
Y_DIM = 8
TH_DIM = 36

# -----------------------------------------------------------------------


class RatSLAM(object):
    """
    RatSLAM module.

    Divided into 4 submodules: odometry, view cells, pose
    cells, and experience map.
    """

    def __init__(self, absolute_rot=False):
        """
        Initializes the ratslam modules.
        """
        self.odometry = Odometry()
        self.view_cells = ViewCells()
        self.pose_cells = PoseCells(X_DIM, Y_DIM, TH_DIM)
        self.experience_map = ExperienceMap(X_DIM, Y_DIM, TH_DIM)

        self.absolute_rot = absolute_rot
        self.last_pose = None
        # TRACKING -------------------------------
        self.prev_trans = []
        self.ma_trans = 1
        self.prev_rot = []
        self.ma_rot = 1

    # ---------------------------------------------------------
    # Public Methods
    # ---------------------------------------------------------

    @timethis
    def step(self, input):
        """
        Performs a step of the RatSLAM algorithm by analysing given input data.
        """
        if not isinstance(input, Input):
            logger.error("ERROR: input is not instance of Input class")
            raise TypeError("Input is not instance of Input class")
        x_pc, y_pc, th_pc = self.pose_cells.active_cell
        # Get activated view cell
        view_cell = self.view_cells.observe_data(input, x_pc, y_pc, th_pc)
        # Get odometry readings
        vtrans, vrot = self.odometry.observe_data(input, absolute_rot=self.absolute_rot)
        # Perform moving average smoothing
        self.prev_trans = [vtrans] + self.prev_trans
        if len(self.prev_trans) > self.ma_trans:
            self.prev_trans = self.prev_trans[: self.ma_trans]
        self.prev_rot = [vrot] + self.prev_rot
        if len(self.prev_rot) > self.ma_rot:
            self.prev_rot = self.prev_rot[: self.ma_rot]
        vtrans = np.mean(self.prev_trans)
        vrot = np.mean(self.prev_rot)
        # Update pose cell network, get index of most activated pose cell
        x_pc, y_pc, th_pc = self.pose_cells.step(view_cell, vtrans, vrot)
        # Execute iteration of experience map
        self.experience_map.step(
            view_cell,
            vtrans,
            vrot,
            x_pc,
            y_pc,
            th_pc,
            true_pose=(input.raw_data[1][0], input.raw_data[1][1]),
            true_odometry=(input.raw_data[1][2], input.raw_data[1][1]),
        )
        # true_odometry=(input.raw_data[1][3], input.raw_data[1][2])) #ORIGINAL
        self.last_pose = (input.raw_data[1][0], input.raw_data[1][2])
