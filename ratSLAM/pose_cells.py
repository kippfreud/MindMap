"""
Class containing pose cell information.
"""

# -----------------------------------------------------------------------

import numpy as np
import itertools

from ratSLAM.utilities import timethis

# -----------------------------------------------------------------------

class PoseCells(object):
    """
    Pose Cell Module
    """
    def __init__(self,
                 x_dim,
                 y_dim,
                 theta_dim,
                 local_excitation_dimension=7,
                 local_excitation_variance=1,
                 local_inhibition_dimension=5,
                 local_inhibition_variance=2,
                 global_inhibition_value=0.00002,
                 activation_center_cells_to_average=8
                 ):
        """
        Instantiates Pose Cell module.
        """
        # Instantiate pose cell matrix
        self.cells = np.zeros([x_dim, y_dim, theta_dim])
        # We know rat starts in one place, so set all activation to central cell
        x, y, t = [round(x_dim/2), round(y_dim/2), round(theta_dim/2)]
        self.active_cell = np.array([x, y, t])
        self.cells[x, y, t] = 1
        # save dimensions
        self.dims = (x_dim, y_dim, theta_dim)
        # save global inhibition value; amount of activation to be removed from each pose
        # cell at each time step.
        self.global_inhibition_value = global_inhibition_value
        # We must create the following variables for use in the _update_activation
        # function - they are used from local excitation and local inhibition; the
        # 'spreading' and 'anti-spreading' of activation.
        self.local_excitation_dimension = local_excitation_dimension
        self.local_excitation_variance = local_excitation_variance
        self.local_inhibition_dimension = local_inhibition_dimension
        self.local_inhibition_variance = local_inhibition_variance
        self.excitatory_wrapped_xy = self._get_wrapped_array(self.dims[0], self.local_excitation_dimension)
        self.excitatory_wrapped_th = self._get_wrapped_array(self.dims[2], self.local_excitation_dimension)
        self.excitatory_spread_matrix = self._create_pose_cell_spread_matrix(self.local_excitation_dimension,
                                                                             self.local_excitation_variance)
        self.inhibitory_wrapped_xy = self._get_wrapped_array(self.dims[0], self.local_inhibition_dimension)
        self.inhibitory_wrapped_th = self._get_wrapped_array(self.dims[2], self.local_inhibition_dimension)
        self.inhibitory_spread_matrix = self._create_pose_cell_spread_matrix(self.local_inhibition_dimension,
                                                                             self.local_inhibition_variance)
        # We create the following variables to help when finding the center of activation
        # of the network in _get_center_of_activation
        self.activation_center_cells_to_average = activation_center_cells_to_average
        self.activation_center_wrapped_xy = list(range(self.dims[0] - self.activation_center_cells_to_average,
                                                  self.dims[0])) + list(range(self.dims[0])) + \
                                                    list(range(self.activation_center_cells_to_average))
        self.activation_center_wrapped_th = list(range(self.dims[2] - self.activation_center_cells_to_average,
                                                  self.dims[2])) + list(range(self.dims[2])) + \
                                                    list(range(self.activation_center_cells_to_average))
        self.xy_sin_lookup = np.sin(np.multiply(range(1, self.dims[0] + 1), (2 * np.pi) / self.dims[0]))
        self.xy_cos_lookup = np.cos(np.multiply(range(1, self.dims[0] + 1), (2 * np.pi) / self.dims[0]))
        self.th_sin_lookup = np.sin(np.multiply(range(1, self.dims[2] + 1), (2 * np.pi) / self.dims[2]))
        self.th_cos_lookup = np.cos(np.multiply(range(1, self.dims[2] + 1), (2 * np.pi) / self.dims[2]))

    ###########################################################
    # Public Methods
    ###########################################################

    @timethis
    def step(self, view_cell, vtrans, vrot):
        """
        Execute an iteration of the pose cell attractor network.

        :param view_cell: the last most activated view cell.
        :param vtrans: the translation of the robot given by odometry.
        :param vrot: the rotation of the robot given by odometry.
        :return: a 3D-tuple with the (x, y, th) index of most active pose cell.
        """
        # if the activated view cell given is not a new cell (and thus is a previously encountered
        # view cell) then add energy to the pose cell associated to that view cell
        if not view_cell.new_view_cell:
            # get indexes of activated pose cell (i.e fix dims between 0 and max)
            # ..todo: is this necessary? why not just act_x = view_cel.x_pc etc?
            act_x = np.min([np.max([int(np.floor(view_cell.x_pc)), 1]), self.dims[0]])
            act_y = np.min([np.max([int(np.floor(view_cell.y_pc)), 1]), self.dims[1]])
            act_th = np.min([np.max([int(np.floor(view_cell.th_pc)), 1]), self.dims[2]])
            # inject energy into this pose cell
            self._inject_energy(view_cell, act_x, act_y, act_th)
        # Update values in activation matrix (local excitation, local inhibition,
        # global inhibition, normalization).
        self._update_activation()
        # Path integration - must now change activation depending on odometry readings.
        self._path_integration(vtrans, vrot)
        # Get index of pose cell with maximum activation and return it
        self.active_cell = self._get_center_of_activation(self.activation_center_wrapped_xy,
                                                          self.activation_center_wrapped_th)
        return self.active_cell

    ###########################################################
    # Private Methods
    ###########################################################

    @timethis
    def _inject_energy(self, cell, x, y, th):
        """
        This will inject some energy into the pose cell attractor network.

        ..todo: currently taken from openratslam - unclear why declared variables are hardcoded

        :param cell: The view cell which has activated the pose cell of given index.
        :param x: x index of activated pose cell.
        :param y: y index of activated pose cell.
        :param th: theta index of activated pose cell.
        """
        pc_vt_inject_energy = 0.1
        energy = pc_vt_inject_energy * (1. / 30.) * (30 - np.exp(1.2 * cell.get_decay()))
        if energy > 0:
            self.cells[x, y, th] += energy

    @timethis
    def _update_activation(self):
        """
        This function will first spread out existing activation in the pose cell network.
        Then will perform local inhibition - removing activation from and around active
        pose cells.
        These first two steps ensure stabilization of energy packets.
        Global inhibition is then applied.
        Normalization is then applied and new cell activation matrix is returned.

        :return: new cell activation matrix.
        """
        # STEP 1: Spread out existing activation in pose cell network
        # pca_new is the new activation values
        new_activations = np.zeros([self.dims[0], self.dims[1], self.dims[2]])
        # For cells with non zero activation we spread their activation out using
        # the matrix returned by _create_pose_cell_spread_matrix.
        indices = np.nonzero(self.cells)
        for i, j, k in zip(*indices):
            # Spread the activation centered at i,j,k out, then add this activation
            # matrix to pca_new
            new_activations[np.ix_(self.excitatory_wrapped_xy[i:i + self.local_excitation_dimension],
                            self.excitatory_wrapped_xy[j:j + self.local_excitation_dimension],
                            self.excitatory_wrapped_th[k:k + self.local_excitation_dimension])] += \
                self.cells[i, j, k] * self.excitatory_spread_matrix
        # STEP 2: Compute local inhibition and subtract this from activation values.
        # Inhibitions are local inhibition values for each cell - to be subtracted from new_activations.
        inhibitions = np.zeros([self.dims[0], self.dims[1], self.dims[2]])
        for i, j, k in zip(*indices):
            # Spread the activation centered at i,j,k out, then add this activation.
            # matrix to pca_new
            inhibitions[np.ix_(self.inhibitory_wrapped_xy[i:i + self.local_inhibition_dimension],
                        self.inhibitory_wrapped_xy[j:j + self.local_inhibition_dimension],
                        self.inhibitory_wrapped_th[k:k + self.local_inhibition_dimension])] += \
                self.cells[i, j, k] * self.inhibitory_spread_matrix
        # subtract local inhibition values from new activations.
        self.cells = new_activations - inhibitions
        # STEP 3: Global inhibition and normalization.
        self.cells[self.cells < self.global_inhibition_value] = 0
        self.cells[self.cells >= self.global_inhibition_value] -= self.global_inhibition_value
        self.cells = self.cells / np.sum(self.cells)
        # return new cell activations
        return self.cells

    def _create_pose_cell_spread_matrix(self, dim, var):
        """
        Creates a matrix encoding a "ball of activation", with higher values towards
        the center of the matrix, and activation spread out according to the var
        param. A dim x dim x dim matrix will be returned. This matrix sums to one
        and so can be used to "spread out" activation in one location by multiplication
        with submatrices of larger networks.
        E.g if you had a n x n x n matrix with all activation in some cell, multiplying
        the area centered around that cell by the matrix returned by this function would
        spread the activation out to the nearest dim cells with variance var.

        :param dim: The dimension of the square matrix to create and return.
        :param var: The spread of the activation in the returned network.
        """
        # find center of activation
        center = int(np.floor(dim / 2.))
        # set weights to zero
        weights = np.zeros([dim, dim, dim])
        # iterate through all weight values
        for x, y, z in itertools.product(range(dim), range(dim), range(dim)):
            # give each cell weight a value based on that cells distance from the center of activation
            dx = -(x - center) ** 2
            dy = -(y - center) ** 2
            dz = -(z - center) ** 2
            weights[x, y, z] = 1.0 / (var * np.sqrt(2 * np.pi)) * np.exp((dx + dy + dz) / (2. * var ** 2))
        # normalize weights
        weights = weights / np.sum(weights)
        return weights

    def _get_wrapped_array(self, dim, wrap_dimension):
        """
        Will return an extended range(dim) with wrapped padding of length
        int(np.floor(wrap_dimension / 2.)).
        E.g self._get_wrap_array(5,1) will return [4,0,1,2,3,4,0]
        """
        return list(range(dim - int(np.floor(wrap_dimension / 2.)), dim)) + \
               list(range(dim)) + list(range(int(np.floor(wrap_dimension / 2.))))

    @timethis
    def _get_center_of_activation(self, wrapped_xy, wrapped_th):
        """
        Finds the x,y,th center of activation in the network, uses weighted
        average (with wrapping) to calculate.

        ..todo: Did not thoroughly read this function.

        ..todo: Better comments in this function

        :return: (x,y,th) coords.
        """
        x, y, z = np.unravel_index(np.argmax(self.cells), self.cells.shape)
        z_posecells = np.zeros([self.dims[0], self.dims[1], self.dims[2]])
        zval = self.cells[np.ix_(
            wrapped_xy[x:x + self.activation_center_cells_to_average * 2],
            wrapped_xy[y:y + self.activation_center_cells_to_average * 2],
            wrapped_th[z:z + self.activation_center_cells_to_average * 2]
        )]
        z_posecells[np.ix_(
            wrapped_xy[x:x + self.activation_center_cells_to_average * 2],
            wrapped_xy[y:y + self.activation_center_cells_to_average * 2],
            wrapped_th[z:z + self.activation_center_cells_to_average * 2]
        )] = zval
        # get the sums for each axis
        x_sums = np.sum(np.sum(z_posecells, 2), 1)
        y_sums = np.sum(np.sum(z_posecells, 2), 0)
        th_sums = np.sum(np.sum(z_posecells, 1), 0)
        th_sums = th_sums[:]
        # now find the (x, y, th) using population vector decoding to handle
        # the wrap around
        x = (np.arctan2(np.sum(self.xy_sin_lookup * x_sums),
                        np.sum(self.xy_cos_lookup * x_sums)) * \
             self.dims[0] / (2 * np.pi)) % (self.dims[0])
        y = (np.arctan2(np.sum(self.xy_sin_lookup * y_sums),
                        np.sum(self.xy_cos_lookup * y_sums)) * \
             self.dims[1] / (2 * np.pi)) % (self.dims[1])
        th = (np.arctan2(np.sum(self.th_sin_lookup * th_sums),
                         np.sum(self.th_cos_lookup * th_sums)) * \
              self.dims[2] / (2 * np.pi)) % (self.dims[2])
        return x, y, th

    @timethis
    def _path_integration(self, translation, rotation):
        """
        Will move activation in network according to the given translation and rotation.

        ..todo: the maths behind the activity matrix is transformed is difficult to \
            understand, spend more time understanding it.

        ..todo: better comments in this function
        """
        # PART 1: Transform activity matrix according to translation
        # Calculate size of the directional bins
        direction_bin_size = (2.*np.pi)/self.dims[2]
        # loop over all theta bins
        for direction_bin_index in range(self.dims[2]):
            direction = float(direction_bin_index-1) * direction_bin_size
            pca90 = np.rot90(self.cells[:, :, direction_bin_index],
                             int(np.floor(direction * 2 / np.pi)))
            dir90 = direction - int(np.floor(direction * 2 / np.pi)) * np.pi / 2
            pca_new = np.zeros([self.dims[0] + 2, self.dims[1] + 2])
            pca_new[1:-1, 1:-1] = pca90
            weight_sw = (translation ** 2) * np.cos(dir90) * np.sin(dir90)
            weight_se = translation * np.sin(dir90) - \
                        (translation ** 2) * np.cos(dir90) * np.sin(dir90)
            weight_nw = translation * np.cos(dir90) - \
                        (translation ** 2) * np.cos(dir90) * np.sin(dir90)
            weight_ne = 1.0 - weight_sw - weight_se - weight_nw
            pca_new = pca_new * weight_ne + \
                      np.roll(pca_new, 1, 1) * weight_nw + \
                      np.roll(pca_new, 1, 0) * weight_se + \
                      np.roll(np.roll(pca_new, 1, 1), 1, 0) * weight_sw
            pca90 = pca_new[1:-1, 1:-1]
            pca90[1:, 0] = pca90[1:, 0] + pca_new[2:-1, -1]
            pca90[1, 1:] = pca90[1, 1:] + pca_new[-1, 2:-1]
            pca90[0, 0] = pca90[0, 0] + pca_new[-1, -1]
            # unrotate the pose cell xy layer
            self.cells[:, :, direction_bin_index] = np.rot90(pca90,
                                                             4 - int(np.floor(direction * 2 / np.pi)))
        # PART 2: Transform activity matrix according to rotation
        # Shift the pose cells +/- theta given by rotation
        if rotation != 0:
            weight = (np.abs(rotation) / direction_bin_size) % 1
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(rotation) * int(np.floor(abs(rotation) / direction_bin_size)))
            shift2 = int(np.sign(rotation) * int(np.ceil(abs(rotation) / direction_bin_size)))
            self.cells = np.roll(self.cells, shift1, 2) * (1.0 - weight) + \
                         np.roll(self.cells, shift2, 2) * (weight)

