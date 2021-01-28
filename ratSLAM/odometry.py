"""
File for Odometry class.
"""

# -----------------------------------------------------------------------

import numpy as np

from ratSLAM.utilities import timethis

# -----------------------------------------------------------------------

class Odometry(object):
    """
    Odometry Module - for making inferences about speed and direction.
    """
    def __init__(self):
        """
        Instantiates Odometry module.
        """
        # We know we start facing some direction, so we set current
        # direction to forwards.
        self.odometry = [0., 0., np.pi/2]

        # Instantiate initial templates for translation and rotation

        self.old_input = None

    ###########################################################
    # Public Methods
    ###########################################################

    @timethis
    def observe_data(self, input):
        """
        Observe new data and return inferred odometry readings.

        :param img: Most recently observed data.
        :return: inferred translation and rotation
        """
        # Calculate translation
        translation, rotation = self._get_odometry(input)
        # Save input as old input
        self.old_input = input

        # Update raw odometry
        #..todo:: ALSO NOT VALID FOR NEW NEURAL RATSLAM!
        self.odometry[2] += rotation
        self.odometry[0] += translation * np.cos(self.odometry[2])
        self.odometry[1] += translation * np.sin(self.odometry[2])

        return translation, rotation

    ###########################################################
    # Private Methods
    ###########################################################

    def _get_odometry(self, input):
        """
        Infer translation and rotation by comparing new data to old translation data.

        :param input: Data to infer translation from
        :return: translation value
        """
        translation, rotation = input.compareOdometry(self.old_input)
        return translation, rotation
