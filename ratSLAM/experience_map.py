"""
Class containing experience map functionality.
"""

# -----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import imageio

from ratSLAM.utilities import timethis
from utils.misc import rotate

# -----------------------------------------------------------------------

class ExperienceNode(object):
    """
    Class for a node on the experience map graph.
    An Experience node is used to to represent a single point in the experience
    map, thus it must store its position in the map and corresponding activations
    of pose and view cell modules.
    """
    def __init__(self, x_pc, y_pc, th_pc, x_em, y_em, th_em, view_cell):
        """
        Instantiates the experience node.

        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :param x_em: the position of axis x in the experience map.
        :param y_em: the position of axis x in the experience map.
        :param th_em: the orientation of the experience, in radians.
        :param view_cell: the last most activated view cell.
        """
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.x_em = x_em
        self.y_em = y_em
        self.th_em = th_em
        self.view_cell = view_cell
        self.links = []

    ###########################################################
    # Public Methods
    ###########################################################

    def link_to(self, child, diff_x, diff_y, theta):
        """
        Create an ExperienceLink object connecting this node to the node given
        by the child parameter.

        :param child: The experience node to be linked to.
        :param diff_x: The estimated x difference between this node and the given child node
        :param diff_y: The estimated th difference between this node and the given child node
        :param theta: The estimated theta value
        """
        dist = np.sqrt(diff_x**2 + diff_y**2)
        # Calculate absolute angle of link.
        absolute_th = self._get_angle_diff(self.th_em, theta)
        # Calculate relative angle between two experiences
        relative_th = self._get_angle_diff(self.th_em, np.arctan2(diff_y, diff_x))
        # Create link and add to list of links
        link = ExperienceLink(self, child, absolute_th, dist, relative_th)
        self.links.append(link)

    ###########################################################
    # Private Methods
    ###########################################################

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


class ExperienceLink(object):
    """
    Class for a link between ExperienceNode instances on the experience map graph
    """
    def __init__(self, parent, child, absolute_th, dist, relative_th):
        """
        Instantiates the ExperienceLink.

        :param parent: the Experience object that owns this link.
        :param child: the target Experience object.
        :param absolute_th: The current angle of the agent.
        :param dist: the euclidean distance between the Experiences.
        :param relative_th: the angle between the experiences.
        """
        self.parent = parent
        self.child = child
        self.dist = dist
        self.absolute_th = absolute_th
        self.relative_th = relative_th


class ExperienceMap(object):
    """
    Experience map module.
    """
    def __init__(self, x_dim, y_dim, th_dim, dist_threshold=10,
                 graph_relaxation_cycles=100, correction_rate=0.5):
        """
        Instantiates the Experience Map.

        :param x_dim: x-dimension of pose cell network.
        :param y_dim: y-dimension of pose cell network.
        :param th_dim: th-dimension of pose cell network.
        :param dist_threshold: Minimum distance from active experience node \
            required to create new experience node.
        :param graph_relaxation_cycles: The number of complete experience map \
            graph relaxation cycles to perform per iteration
        :param correction_rate: ..todo: add docs for this.
        """
        # Instantiate networkX graph object
        self.G = nx.Graph()
        # save dimensions
        self.dims = (x_dim, y_dim, th_dim)
        # save number of graph relaxation cycles and correction rate
        self._graph_relaxation_cycles = graph_relaxation_cycles
        self._correction_rate = correction_rate
        # number of experience nodes in experience map
        self.size = 0
        # currently active experience node
        self.current_exp = None
        # currently active view cell
        self.current_view_cell = None
        # best estimate of current position relative to the currently active
        # experience node.
        self.accum_diff_x = 0
        self.accum_diff_y = 0
        self.accum_th = 0#np.pi / 2
        # history of previously visited experience nodes
        self.history = []
        # Minimum distance from active experience node required to create
        # new experience node.
        self._dist_threshold = dist_threshold
        # Keep track of initial position and angle for debugging purposes
        self.initial_pose = None
        self.true_pose = None
        self.true_speed = None
        self.prev_visited = []

        # Plotting
        self.fig = plt.figure(figsize=(10., 4.))
        self.position_ax = self.fig.add_subplot(121, facecolor='#E6E6E6')
        self.position_ax.set_xlim([0, 750])
        self.position_ax.set_ylim([0, 600])
        self.compass_ax = self.fig.add_subplot(122, polar=True, facecolor='#E6E6E6')
        self.compass_ax.set_ylim(0, 5)
        self.compass_ax.set_yticks(np.arange(0, 5, 1.0))
        # radar green, solid grid lines
        plt.rc('grid', color='#316931', linewidth=1, linestyle='-')
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.ion()

    ###########################################################
    # Public Methods
    ###########################################################

    def plot(self, writer):
        """
        Plots experience map

        ..todo::make this better
        """
        self.position_ax.clear()
        self.compass_ax.clear()
        self.position_ax.set_xlim([-300, 600])
        self.position_ax.set_ylim([-300, 600])
        self.compass_ax.set_ylim(0, 0.02)
        self.compass_ax.set_yticks(np.arange(0, 0.2, 0.05))

        # POSITION AXIS
        true_p_adj = rotate(self.true_pose[0] - self.initial_pose[0], degrees=self.initial_pose[1])
        self.prev_visited.append((true_p_adj[0], true_p_adj[1]))
        self.position_ax.scatter([true_p_adj[0]], [true_p_adj[1]], c="green")
        self.position_ax.scatter([t[0] for t in self.prev_visited], [t[1] for t in self.prev_visited], c="pink")
        pos = {e: (e.x_em, e.y_em) for e in self.G.nodes}
        cols = ["#004650" if e==self.current_exp else "#933A16" for e in self.G.nodes]
        nx.draw(self.G, pos=pos, node_color=cols, node_size=50, ax=self.position_ax)

        # COMPASS AXIS
        self.compass_ax.arrow(0, 0,
                             # -5.,5.,
                             self.true_pose[1], self.true_speed,
                             alpha=0.5, width=0.1,
                             edgecolor='black', facecolor='green', lw=1.3, zorder=3)
        self.compass_ax.arrow(0, 0,
                              # -5.,5.,
                              self.accum_th, self.true_speed,
                              alpha=0.5, width=0.1,
                              edgecolor='black', facecolor='red', lw=1.3, zorder=3)

        plt.pause(0.005)

        image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        writer.append_data(image)

    @timethis
    def step(self, view_cell, translation, rotation, x_pc, y_pc, th_pc, true_pose=None, true_odometry=None):
        """
        Execute an iteration of the experience map.

        :param view_cell: the current most activated view cell.
        :param x_pc: x index of the current pose cell.
        :param y_pc: y index of the current pose cell.
        :param th_pc: th index of the current pose cell.
        :param translation: the translation of the robot given by odometry.
        :param rotation: the rotation of the robot given by odometry.
        """
        if self.initial_pose is None:
            self.initial_pose = true_pose

        if self.true_pose is None:
            self.true_speed = 0
        else:
            self.true_speed = ((self.true_pose[0][0] - true_pose[0][0])**2 + (self.true_pose[0][1] - true_pose[0][1])**2)**0.5
        self.true_pose = true_pose


        #translation = min(translation*10000,5)
        #if true_odometry is not None:
            #print("INFO:: Not using true odometry")
            #translation = true_odometry[0]
            #rotation = true_odometry[1]

        print(f"Rotation is {rotation}")
        print(f"Translation is {translation}")
        # Use translation and rotation to update current position estimates of agent.
        self.accum_th = self._clip_angle_pi(self.accum_th - rotation)
        #self.accum_th = self._clip_angle_pi(rotation)
        self.accum_diff_x += translation * np.cos(self.accum_th)
        self.accum_diff_y += translation * np.sin(self.accum_th)
        # Get distance between last experience and new location
        if self.current_exp is None:
            distance = 0
        else:
            distance = np.sqrt(
                self._min_dist(self.current_exp.x_pc, x_pc, self.dims[0])**2 + \
                self._min_dist(self.current_exp.y_pc, y_pc, self.dims[1])**2 #+ \
                #self._min_dist(self.current_exp.th_pc, th_pc, self.dims[2])**2
            )
        # Keep track of whether to adjust map (in case of loop closure)
        adjust_map = False
        # If we have moved above a threshold distance from the previous experience node,
        # or if the currently active view cell is new and thus has no corresponding
        # experience, we will create one.
        # ..todo: I do not believe the second part of this logical is necessary. \
        #       it uses dist threshold for 2 separate things and is never triggered here usefully (so far).
        if view_cell.get_num_associated_experiences() == 0 or distance > self._dist_threshold:
            # Make a new experience
            exp = self._create_experience(x_pc, y_pc, th_pc, view_cell)
            # Set this experience to the currently active experience
            self.current_exp = exp
            # Set accum_difference x and y variables to 0
            self.accum_diff_x = 0
            self.accum_diff_y = 0
            # Set accum_th variable to true experience th value.
            self.accum_th = self.current_exp.th_em
        # If the view cell has changed, but is not new, we search for the given view cells
        # matching experience.
        elif view_cell != self.current_exp.view_cell:
            # Find the experience associated with the current view cell and which is under
            # the threshold distance from the center of the pose cell activation (i.e is not
            # associated to a place which is too far away from estimated position).
            adjust_map = True
            matched_exp = None
            distances_from_center = []
            num_candidate_matches = 0
            for experience in view_cell.iter_experiences():
                distance = np.sqrt(
                    self._min_dist(experience.x_pc, x_pc, self.dims[0])**2 + \
                    self._min_dist(experience.y_pc, y_pc, self.dims[1])**2 #+ \
                    #self._min_dist(experience.th_pc, th_pc, self.dims[2])**2
                )
                distances_from_center.append(distance)
                if distance < self._dist_threshold:
                    num_candidate_matches += 1
            # Currently there is no implementation for when there are multiple matching nodes
            # ..todo: Implement an algorithm for dealing with this situation
            if num_candidate_matches > 1:
                print("WARNING: Multiple matching experience nodes, no implementation for dealing with this")
            else:
                # If there is at most one candidate experience node match
                min_distance_index = np.argmin(distances_from_center)
                min_distance = distances_from_center[min_distance_index]
                if min_distance < self._dist_threshold:
                    # If the min distance is below the threshold, then it is a match.
                    matched_exp = view_cell.get_experience(min_distance_index)
                    # Check if a link already exists from the previous experience to the matched experience
                    link_exists = False
                    for linked_exp in [l.child for l in self.current_exp.links]:
                        if linked_exp == matched_exp:
                            link_exists = True
                    # If a link does not exist, we create one.
                    if not link_exists:
                        self._link(self.current_exp, matched_exp, self.accum_diff_x,
                                   self.accum_diff_y, self.accum_th)
                # If there is no matched experience node, create a new experience node.
                if matched_exp is None:
                    matched_exp = self._create_experience(x_pc, y_pc, th_pc, view_cell)
                self.current_exp = matched_exp
                self.accum_diff_x = 0
                self.accum_diff_y = 0
                # ..todo: removed this as experiences are no longer all the same theta!
                #self.accum_th = self.current_exp.th_em
        # Add current experience node to history
        self.history.append(self.current_exp)
        # If we do not need to adjust the map, return.
        if not adjust_map:
            return
        # if we do need to adjust the map, do it now.

        #self._adjust_map()
        return

    ###########################################################
    # Private Methods
    ###########################################################

    def _adjust_map(self):
        """
        Adjust map; fix errors, merge nodes. Called after loop closures detected.

        ..todo:: this does not merge nodes - it should.
        """
        for i in range(0, self._graph_relaxation_cycles):
            for experience_0 in self.G.nodes:
                for link in experience_0.links:
                    experience_1 = link.child
                    # Calculate where experience_1 thinks experience_2 should be based
                    # on the information contained in the link.
                    link_x = experience_0.x_em + link.dist * np.cos(experience_0.th_em + link.relative_th)
                    link_y = experience_0.y_em + link.dist * np.sin(experience_0.th_em + link.relative_th)
                    # Correct locations of experience_0 and experience_1 by equal but opposite amounts.
                    # A 0.5 correction parameter means that e0 and e1 will be  fully corrected
                    # based on e0's link information.
                    experience_0.x_em = experience_0.x_em + (experience_1.x_em - link_x) * self._correction_rate
                    experience_0.y_em = experience_0.y_em + (experience_1.y_em - link_y) * self._correction_rate
                    experience_1.x_em = experience_1.x_em - (experience_1.x_em - link_x) * self._correction_rate
                    experience_1.y_em = experience_1.y_em - (experience_1.y_em - link_y) * self._correction_rate
                    # Determine the difference between the angle experience_0 thinks experience_1 is facing and
                    # the angle that experience_1 thinks that it is facing
                    th_diff = self._get_angle_diff(experience_0.th_em + link.relative_th,
                                                   experience_1.th_em)
                    # Again, correct estimated angles of experience_0 and experience_1 by equal and
                    # opposite amounts
                    experience_0.th_em = self._clip_angle_pi(experience_0.th_em + th_diff*self._correction_rate)
                    experience_1.th_em = self._clip_angle_pi(experience_1.th_em - th_diff*self._correction_rate)
        return

    def _create_experience(self, x_pc, y_pc, th_pc, view_cell):
        """
        Creates a new Experience Node and adds it to the experience map.
        Necessary links are created.

        :param x_pc: x index of currently active pose cell.
        :param y_pc: y index of currently active pose cell.
        :param th_pc: th index of currently active pose cell.
        :param view_cell: Currently most activated view cell.
        :return: The created ExperienceNode.
        """
        self.size += 1
        # Get x, y, and theta estimates for experience node to be created.
        x_em = self.accum_diff_x
        y_em = self.accum_diff_y
        th_em = self._clip_angle_pi(self.accum_th)
        if self.current_exp is not None:
            x_em += self.current_exp.x_em
            y_em += self.current_exp.y_em
        # Instantiate new experience node and add it to networkx graph object
        experience = ExperienceNode(x_pc, y_pc, th_pc, x_em, y_em, th_em, view_cell)
        self.G.add_node(experience)
        # Add links if current experience is not None.
        if self.current_exp is not None:
            self._link(self.current_exp, experience, self.accum_diff_x, self.accum_diff_y, self.accum_th)
        # Add to view cells list of associated experience nodes
        view_cell.add_associated_experience(experience)
        # Return created experience
        return experience

    def _link(self, parent, child, x_dif, y_dif, th):
        """
        Instantiates a link between two experience nodes, from parent to child.

        :param parent: The Experience node parent of the link.
        :param child: The Experience node child of the link.
        :param x_dif: The difference in x coordinates from parent to child.
        :param y_dif: The difference in y coordinates from parent to child
        :param th: The angle the agent is heading when at the child node.
        """
        if self.G.has_edge(parent, child):
            print("WARNING: This edge already exists!")
        self.G.add_edge(parent, child, weight=(x_dif**2 + y_dif**2)**0.5)
        parent.link_to(child, x_dif, y_dif, th)

    def _min_dist(self, d1, d2, wrap_at):
        """
        Returns the minimum distance between d1 and d2, where numbers
        "wrap around" starting at wrap_at.
        E.g _min_dist(0,5,6) would return 1.
        """
        delta = np.min([np.abs(d1 - d2), wrap_at - np.abs(d1 - d2)])
        return delta

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