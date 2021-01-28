"""
Contains a wrapper class for input taken by RatSLAM algorithm,

Input must have a way to be "templatized", and have a method for comparing
templates.
"""

# -----------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np

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