"""
Contains a wrapper class for input taken by RatSLAM algorithm,

Input must have a way to be "templatized", and have a method for comparing
templates.
"""

# -----------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np

from utils.logger import logger
from utils.use_DI_model import MODEL

# -----------------------------------------------------------------------


class Input(ABC):
    """
    Abstract base class for RatSLAM input
    """

    def __init__(self, data):
        """
        instantiates input wrapper.

        :param data: raw data to be wrapped.
        """
        self.raw_data = data
        self.template = self._getTemplate(data)

    @abstractmethod
    def _getTemplate(self, data):
        """
        Templateizes the raw data, returns template.
        """
        logger.error("ERROR: Function should be overwritten by derived class.")
        raise NotImplementedError

    @abstractmethod
    def compareSimilarity(self, other_data):
        """
        Compares this inputs template to another - returns similarity score.
        """
        logger.error("ERROR: Function should be overwritten by derived class.")
        raise NotImplementedError

    @abstractmethod
    def compareOdometry(self, other_template):
        """
        Compares this inputs template to another - returns odometry values
        """
        logger.error("ERROR: Function should be overwritten by derived class.")
        raise NotImplementedError


# -----------------------------------------------------------------------


class DummyInput(Input):
    """
    Abstract base class for dummy RatSLAM input; this is input with odometry data annotated.
    """

    def __init__(self, data):
        """
        instantiates input wrapper.

        :param data: raw data to be wrapped, should be a 2-tuple with (data, odometry)
        """
        if not isinstance(data, tuple):
            logger.error("ERROR: data should be a 2 element tuple")
            raise TypeError("ERROR: data should be a 2 element tuple")
        if len(data) != 2:
            logger.error("ERROR: data should be a 2 element tuple")
            raise TypeError("ERROR: data should be a 2 element tuple")
        self.raw_data = data
        self.template = self._getTemplate(data)

    def _getTemplate(self, data):
        """
        Template-izes the raw data, returns template.
        """
        return data[0]

    def compareSimilarity(self, other_data):
        """
        Compares this inputs template to another - returns similarity score.
        """
        return np.sum(np.abs(self.template - other_data.template))

    def compareOdometry(self, other_template):
        """
        Compares this inputs template to another - returns odometry values
        """
        return self.raw_data[1]


# -----------------------------------------------------------------------


class MattJonesDummyInput(Input):
    """
    Abstract base class for Matt Jones RatSLAM input. The input data is annotated with truth
    values for position and angle so we can cheat at loop closure.
    """

    def __init__(self, data):
        """
        instantiates input wrapper.

        :param data: raw data to be wrapped, should be a 2-tuple with (data, odometry)
        """
        if not isinstance(data, tuple):
            logger.error("ERROR: data should be a 2 element tuple")
            raise TypeError("ERROR: data should be a 2 element tuple")
        if len(data) != 2:
            logger.error("ERROR: data should be a 2 element tuple")
            raise TypeError("ERROR: data should be a 2 element tuple")
        self.raw_data = data
        self.template = self._getTemplate(data)

    def _getTemplate(self, data):
        """
        Templateizes the raw data, just returns the input data here.
        """
        return data[0]

    def compareSimilarity(self, other_data):
        """
        Compares this inputs template to another - returns similarity score.
        This dummy version is cheating, and uses true location and angle labels.
        See class below for non-cheating class.
        """
        current_loc = self.raw_data[1][0]
        current_ang = self.raw_data[1][1]
        other_loc = other_data.raw_data[1][0]
        other_ang = other_data.raw_data[1][1]
        loc_diff = (
            (current_loc[0] - other_loc[0]) ** 2 + (current_loc[1] - other_loc[1]) ** 2
        ) ** 0.5
        ang_diff = np.abs(current_ang - other_ang) * (np.pi / 180.0)
        return loc_diff + ang_diff

    def compareOdometry(self, other_template):
        """
        Compares this inputs template to another - returns odometry values in the form (speed, angle in radians)
        """
        return MODEL.get_odometry(self.template)


# -----------------------------------------------------------------------


class MattJonesInput(Input):
    """
    Matt Jones RatSLAM input.
    """

    def __init__(self, data):
        """
        instantiates input wrapper.

        :param data: raw data to be wrapped, should be a 2-tuple with (data, odometry)
        """
        if not isinstance(data, tuple):
            logger.error("ERROR: data should be a 2 element tuple")
            raise TypeError("ERROR: data should be a 2 element tuple")
        if len(data) != 2:
            logger.error("ERROR: data should be a 2 element tuple")
            raise TypeError("ERROR: data should be a 2 element tuple")
        self.raw_data = data
        self.odom = None
        self.template = self._getTemplate(data)

    def _getTemplate(self, data):
        """
        Templateizes the raw data, just returns the input data here.
        """
        return data[0]

    def compareSimilarity(self, other_data):
        """
        Compares this inputs template to another - returns similarity score.
        """
        if self.odom is None:
            self.odom = MODEL.get_odometry(self.template, True)
        current_loc = self.odom[2]
        other_loc = other_data.odom[2]
        loc_diff = (
            (current_loc[0] - other_loc[0]) ** 2 + (current_loc[1] - other_loc[1]) ** 2
        ) ** 0.5
        return loc_diff

    def compareOdometry(self, other_template):
        """
        Compares this inputs template to another - returns odometry values in the form (speed, angle in radians)
        """
        if self.odom is None:
            speed, angle, pos = MODEL.get_odometry(self.template, True)
            self.odom = (speed, angle, pos)
        return self.odom[0], self.odom[1]
