"""
Contains a wrapper class for input taken by RatSLAM algorithm,

Input must have a way to be "templatized", and have a method for comparing
templates.
"""

# -----------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np

from utils.use_DI_model import get_odometry

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
        print("ERROR: Function should be overwritten by derived class.")
        exit(0)

    @abstractmethod
    def compareSimilarity(self, other_data):
        """
        Compares this inputs template to another - returns similarity score.
        """
        print("ERROR: Function should be overwritten by derived class.")
        exit(0)

    @abstractmethod
    def compareOdometry(self, other_template):
        """
        Compares this inputs template to another - returns odometry values
        """
        print("ERROR: Function should be overwritten by derived class.")
        exit(0)

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
            print("ERROR: data should be a 2 element tuple")
            exit(0)
        if len(data) != 2:
            print("ERROR: data should be a 2 element tuple")
            exit(0)
        self.raw_data = data
        self.template = self._getTemplate(data)

    def _getTemplate(self, data):
        """
        Templateizes the raw data, returns template.
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
            print("ERROR: data should be a 2 element tuple")
            exit(0)
        if len(data) != 2:
            print("ERROR: data should be a 2 element tuple")
            exit(0)
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

        ..todo: Implement this by cheating!
        """
        current_loc = self.raw_data[1][0]
        current_ang = self.raw_data[1][1]
        other_loc = other_data.raw_data[1][0]
        other_ang = other_data.raw_data[1][1]
        loc_diff = ( (current_loc[0] - other_loc[0])**2 + (current_loc[1] - other_loc[1])**2 ) ** 0.5
        ang_diff = np.abs(current_ang - other_ang)*(np.pi/180.)
        return loc_diff + ang_diff
        #return 1.


    def compareOdometry(self, other_template):
        """
        Compares this inputs template to another - returns odometry values in the form (speed, angle in radians)
        """
        return get_odometry(self.template)
