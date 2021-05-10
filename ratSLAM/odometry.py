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
        self.odometry = [0., 0., 0.]

        # Instantiate initial templates for translation and rotation

        self.old_input = None
        self.old_rotation = None

    ###########################################################
    # Public Methods
    ###########################################################

    @timethis
    def observe_data(self, input, absolute_rot=False):
        """
        Observe new data and return inferred odometry readings.

        :param img: Most recently observed data.
        :return: inferred translation and rotation
        """
        translation, r = self._get_odometry(input)

        if self.old_rotation is None or absolute_rot is False:
            rotation = r
        else:
            rotation = self._get_angle_diff(r, self.old_rotation)

        # Save input as old input
        self.old_rotation = r
        self.old_input = input

        # Update raw odometry
        #..todo:: ALSO NOT VALID FOR NEW NEURAL RATSLAM!
        if absolute_rot:
            self.odometry[2] = rotation
        else:
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

    def _get_angle_diff(self, angle1, angle2):
        """
        Returns the difference between angle1 and angle 2, clipped between
        pi and -pi.

        :param angle1: First angle to be compared
        :param angle2: Second angle to be compared
        :return: a float; the different between these two angles, clipped between -pi and pi
        """
        angle_diff = self._clip_angle_pi(angle2 - angle1)
        absolute_angle_diff = abs(self._clip_angle_2pi(angle1) - self._clip_angle_2pi(angle2))
        # If absolute_angle_diff is less than pi
        if absolute_angle_diff < (2 * np.pi - absolute_angle_diff):
            if angle_diff > 0:
                # and angle difference is positive; return absolute_angle_diff
                angle = absolute_angle_diff
            else:
                # and angle difference is positive; return -absolute_angle_diff
                angle = -absolute_angle_diff
        # If absolute_angle_diff is more than or equal to pi
        else:
            if angle_diff > 0:
                # and angle difference is positive; return 2*pi - absolute_angle_diff
                angle = 2 * np.pi - absolute_angle_diff
            else:
                # and angle difference is negative; return -(2*pi - absolute_angle_diff)
                angle = -(2 * np.pi - absolute_angle_diff)
        return angle

    def _clip_angle_pi(self, angle):
        """
        Will clip the given angle between -PI and PI, return clipped angle.
        E.g _clip_angle_pi(3*np.pi) returns np.pi
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def _clip_angle_2pi(self, angle):
        """
        Will clip the given angle between -PI and PI, return clipped angle.
        E.g _clip_angle_2pi(-np.pi) returns np.pi
        """
        while angle < 0:
            angle += 2 * np.pi
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        return angle