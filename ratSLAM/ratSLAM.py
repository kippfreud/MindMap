"""
Class containing the main ratSLAM class
"""

# -----------------------------------------------------------------------

import numpy as np

from ratSLAM.experience_map import ExperienceMap
from ratSLAM.odometry import Odometry
from ratSLAM.pose_cells import PoseCells
from ratSLAM.view_cells import ViewCells
from ratSLAM.input import Input
from ratSLAM.utilities import timethis

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
    def __init__(self):
        """
        Initializes the ratslam modules.
        """
        self.odometry = Odometry()
        self.view_cells = ViewCells()
        self.pose_cells = PoseCells(X_DIM, Y_DIM, TH_DIM)
        self.experience_map = ExperienceMap(X_DIM, Y_DIM, TH_DIM)

        # TRACKING -------------------------------
        #self.odometry = self.odometry.odometry
        #self.active_pc = self.pose_cells.active_cell

    ###########################################################
    # Public Methods
    ###########################################################

    @timethis
    def step(self, input):
        """
        Performs a step of the RatSLAM algorithm by analysing given input data.
        """
        if not isinstance(input, Input):
            print("ERROR: input is not instance of Input class")
            exit(0)
        x_pc, y_pc, th_pc = self.pose_cells.active_cell
        # Get activated view cell
        view_cell = self.view_cells.observe_data(input, x_pc, y_pc, th_pc)
        # Get odometry readings
        vtrans, vrot = self.odometry.observe_data(input)
        # Update pose cell network, get index of most activated pose cell
        x_pc, y_pc, th_pc = self.pose_cells.step(view_cell, vtrans, vrot)
        # Execute iteration of experience map
        self.experience_map.step(view_cell, vtrans, vrot, x_pc, y_pc, th_pc)
