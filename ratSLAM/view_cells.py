"""
View Cell and View Cells classes.
"""

# -----------------------------------------------------------------------

import numpy as np

from ratSLAM.input import Input
from ratSLAM.utilities import timethis

# -----------------------------------------------------------------------


class ViewCell(object):
    """
    A single view cell.
    """
    def __init__(self, input, x_pc, y_pc, th_pc):
        """
        :param template: data associated with this view cell.
        :param x_pc: x index of pose cell associated with this view cell.
        :param y_pc: y index of pose cell associated with this view cell.
        :param th_pc: theta index of pose cell associated with this view cell.
        """
        self.input = input
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        # Keep track of if this cell is a new addition or a previously encountered
        # view cell; always true on instantiation.
        self.new_view_cell = True
        # view cells must keep track of their "decay" values -
        # the decay parameter decreases the amount of energy
        # injected by repeated exposure to the same visual scene. Variables for this process are
        # defined here. Also see get_decay and set_active functions below.
        # ..todo: are these variables good? Is this the best way of handling decay?
        self._active_decay = 0.5
        self._global_decay = 0.01
        self._decay_value = self._active_decay
        # Keep track of the experiences associated with the view cell.
        self._associated_experience_nodes = []

    ###########################################################
    # Public Methods
    ###########################################################

    def get_experience(self, index):
        """
        Returns the experience in position index from the _associated_experience_nodes var.
        """
        return self._associated_experience_nodes[index]

    def iter_experiences(self):
        """
        Iterate through _associated_experience_nodes.
        """
        for experience in self._associated_experience_nodes:
            yield experience

    def get_num_associated_experiences(self):
        """
        Returns the number of experience nodes associated with this view cell]
        """
        return len(self._associated_experience_nodes)

    def add_associated_experience(self, experience):
        """
        Adds the given experience to this classes _associated_experience_nodes param.

        :param experience: An ExperienceNode associated with this view cell.
        """
        self._associated_experience_nodes.append(experience)

    def get_decay(self):
        """
        Returns the _decay_value param.

        ..todo: again, not sure if this is best way of handling decay - see above.
        """
        return self._decay_value

    def activated(self):
        """
        Called whenever cell is activated - adds the _active_decay param to the
        decay value.

        ..todo: again, not sure if this is best way of handling decay - see above.
        """
        self._decay_value += self._active_decay

    def apply_global_decay(self):
        """
        Reduce the decay value of this cell by the _global_decay param.

        ..todo: again, not sure if this is best way of handling decay - see above.
        """
        self._decay_value -= self._global_decay

# -----------------------------------------------------------------------


class ViewCells(object):
    """
    View Cell Module
    """
    def __init__(self):
        """
        Instantiates the View Cell module
        """
        self.size = 0
        self.cells = []
        self.active_cell = None

    ###########################################################
    # Public Methods
    ###########################################################

    @timethis
    def observe_data(self, input, x_pc, y_pc, th_pc):#
        """
        Observe data, either
        - Add new view cell and strengthen it's connections to currently /
        active pose cells (x_pc, y_pc, th_pc) and return it, or
        - Take view cell corresponding to data most similar to that found /
        in img and return that.

        :param img: Most recently observed data.
        :param x_pc: x index of currently active pose cell.
        :param y_pc: y index of currently active pose cell.
        :param th_pc: theta index of currently active pose cell.
        :return: view cell active after observing this data.
        """
        if not isinstance(input, Input):
            print("ERROR: input is not instance of Input class.")
            exit(0)
        # Get similarity scores comparing this data to all others
        scores = self._get_similarity_scores(input)
        # Decrease the decay value of each view cell
        self._global_decay()
        # Using these scores, we not decide whether to create a new cell,
        # or activate the most similar cell which exists already
        if self._should_I_make_new_cell(scores):
            # Make new view cell with connections to the currently active
            # view cells.
            cell = self._create_cell(input, x_pc, y_pc, th_pc)
            # Set new cell as the active cell and return it.
            self.active_cell = cell
            return cell
        else:
            # Find most similar view cell and save that as the currently
            # active cell, then return it.
            index_of_most_similar_cell = np.argmin(scores)
            cell = self.cells[index_of_most_similar_cell]
            # Call the activated function of the view cell - this changes the
            # cells decay value.
            cell.activated()
            # As we are using an old cell here, either it is the same as the old view cell
            # (in which case it may be a brand new view cell), or it is different from the
            # old view cell, in which case it is definitely not a brand new cell, so we will
            # set its new_view_cell param to false.
            if self.active_cell != cell:
                cell.new_view_cell = False
            # Set new cell as the active cell and return it.
            self.active_cell = cell
            return cell

    ###########################################################
    # Private Methods
    ###########################################################

    def _create_cell(self, input, x_pc, y_pc, th_pc):
        """
        Create a new View Cell and add it to registry of view cells, then return it.

        :param input: data associated with this view cell.
        :param x_pc: x index of pose cell associated with this view cell.
        :param y_pc: y index of pose cell associated with this view cell.
        :param th_pc: theta index of pose cell associated with this view cell.
        """
        cell = ViewCell(input, x_pc, y_pc, th_pc)
        self.cells.append(cell)
        self.size += 1
        return cell

    def _get_similarity_scores(self, input):
        """
        Computes similarity of the given template to the templates associated with \
        all other view cells

        :param template: data to be compared to templates from existing view cells.
        :return: 1D numpy array of similarity scores.
        """
        scores = []
        for view_cell in self.cells:
            sim_score = input.compareSimilarity(view_cell.input)
            scores.append(sim_score)
        return scores

    def _should_I_make_new_cell(self, scores):
        """
        Decides whether to create a new view cell or activate a pre \
        existing view cell.

        :param scores: Similarity scores between currently observed data templates \
        and all data templates associated with currently existing view cells.
        :return: True if we should create a new view cell, False otherwise.
        """
        #..todo:: AGAIN, TAKEN FROM ORIGINAL RATSLAM-PYTHON WITH MINOR ALTERATION, NOT VERY GOOD!
        #..todo:: THRESHOLD SHOULD INCREASE WITH NUMBER OF VIEW CELLS.
        if self.size == 0:
            return True
        if np.min(scores) > 20: #..todo: global param
            return True
        return False

    def _global_decay(self):
        """
        Decreases the decay value of each view cell.

        ..todo: Not sure if this is best way of handling decay - see view cell class.
        """
        for cell in self.cells:
            cell.apply_global_decay()
