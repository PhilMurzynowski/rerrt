"""
Base RRT Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from trees.nodes import RRTNode

from visuals.helper import printProgressBar


class RRT:
    """
    Class to grow tree. Makes use of RRTNode class.
    start       :nparray: (nx x 1)      desired start of trajectory
    goal        :nparray: (nx x 1)      desired end of trajectory
    system      :System:                System object to provide dynamics
    input       :Input:                 Input object configured for desired input
                                        sampling
    scene       :Scene:                 Scene object used to organized region and
                                        obstacles
    Required options:
        min_dist
        max_iter
        direction
        track_children
        extend_by
        goal_sample_rate
        sample_dim
        distanceMetric

    To run tree growth call tree.treeExpansion where tree is an :RRT: instance.
    Note: Currently functional for 2D, in process of generalizing.
    """


    def __init__(self, start, goal, system, input_, scene, opts):
        """Initalization, parameters descirbed above. Options (opts) passed in
        to configure distanceMetric.
        """
        self.start = start
        self.goal = goal
        self.system = system
        self.input = input_
        self.scene = scene
        self.region = scene.region
        self.obstacles = scene.obstacles
        self.poly = [self.region] + self.obstacles
        # default 
        self.setDistanceMetric(opts)

    def inObstacle(self, point):
        """Method to check if point is within any obstacle.
        Currently not optimized based on location data, etc.
        point   :nparray: (? x 1)       part of state
        """
        for o in self.obstacles:
            if o.inPoly(point):
                return True
        return False


    def inRegion(self, point):
        """Check if point is within valid physical region.
        Note: could be expanded to include limits on remaining states.
        E.g. for Car could place limits on speed.
        point   :nparray: (? x 1)       part of state
        """
        return self.region.inPoly(point)

    def validState(self, x):
        """Check used in treeExpansion. Given state x, perform one full check
        to ascertain whether the state is in free space, i.e. in the valid region
        and outside obstacles.
        x       :nparray: (nx x 1)      state
        """
        if not self.inRegion(x[:2]) or self.inObstacle(x[:2]):
            return False
        return True

    def sample(self, opts):
        """Sample valid region, meaning within region but excluding obstacles.
        Can provide a desired frequency of sampling the goal to favor growth
        towards the goal.
        Required options:
            goal_sample_rate
            sample_dim
        """
        if np.random.rand() > opts['goal_sample_rate']:
            # sample region outside of obstacles
            rnd = None
            while rnd is None or self.inObstacle(rnd):
                rnd = np.random.uniform(self.region.v1, self.region.v4, (opts['sample_dim'], 1))
        else:
            # sample goal
            # add check to ensure goal is appropriate number of dimensions
            # can also remove slicing if correct number of dimensions
            rnd = self.goal[:opts['sample_dim'], :]
        return rnd

    def euclidean2D(self, p1, p2):
        """Calculate 2 dimensional euclidean distance between two points.
        """
        #p1_xy = p1[:2].reshape(2, 1) if p1.shape != (2, 1) else p2
        #p2_xy = p2[:2].reshape(2, 1) if p2.shape != (2, 1) else p2
        #return np.linalg.norm(p1_xy - p2_xy)
        p1r = p1.reshape(-1, 1) if (p1.ndim == 1 or p1.shape != (-1, 1)) else p1
        p2r = p2.reshape(-1, 1) if (p2.ndim == 1 or p2.shape[1] != 1) else p2
        return np.sqrt((p1r[0, 0]-p2r[0, 0])**2+(p1r[1, 0]-p2r[1, 0])**2)

    def setDistanceMetric(self, opts):
        """Can implement different distance metrics for nonlinear systems.
        Definied during initialization. Default is euclidean2D."""
        if opts['distanceMetric'] is None:
            self.distanceMetric = self.euclidean2D


    def distToGoal(self, p):
        """Distance from point p to goal using desired distanceMetric
        p       :nparray: (? x 1)       part of state
        """
        return self.distanceMetric(p, self.goal)


    def nearestNode(self, new_location, get_dist=False):
        """Find node in tree nearest to new_location in terms of distanceMetric.
        Note: Currently inefficient, looks through all nodes in tree.
        Note: Can swap in more efficient implementation with kd Tree perhaps.
        However, for nonlinear dynamics euclidean distance doesn't always make
        sense so the kd Tree would perhaps also be based off distanceMetric in
        some fashion.
        """
        # if system does not move on first timestep, will encounter error
        # will keep returning start node, as first in list
        L = len(self.node_list)
        closest_distance = np.Inf
        closest_node = None
        for i in range(L):
            idx = L-1-i
            #idx = i
            distance = self.distanceMetric(new_location, self.node_list[idx].x)
            if distance < closest_distance:
                closest_distance = distance
                closest_node = self.node_list[idx]
        if get_dist:
            return closest_node, closest_distance
        return closest_node

    def steer(self, from_node, to_location, opts):
        """Function that is bulk of the extension operation, used after obtaining
        a random sample and determining the nearest node to the sample. Simulates
        the system from the nearest node (from_node) with multiple control
        inputs for multiple timesteps, chooses the control input which results in
        a state closest to the desired partial state (to_location) according to
        distanceMetric, and returns nodes and input for that simulation. Possible
        control inputs are configured by self.input which is an Input object
        passed in to initalize RRT.
        from_node       :RRTNode:           node to steer from
        to_location     :nparray: (? x 1)   partial state
        Required options:
            extend_by
            direction
        """
        best_u = None
        best_proximity = np.inf
        for k in range(self.input.numsamples):
            key, action = self.input.getAction(k)
            steered_to = self.system.simulate(from_node.x, action, opts['extend_by'], opts['direction'])
            #plt.plot([from_node.x[0], steered_to[0]], [from_node.x[1], steered_to[1]])
            proximity = self.distanceMetric(to_location, steered_to)
            if proximity < best_proximity:
                best_key, best_proximitiy = key, proximity
        parent = from_node
        new_nodes = []
        for i in range(opts['extend_by']):
            xsim = self.system.simulate(parent.x, self.input.actions[best_key], 1, opts['direction'])
            new_nodes.append(RRTNode(xsim, parent, opts=opts))
            parent = new_nodes[-1]
        return new_nodes, self.input.actions[key]

    def extend(self, opts):
        """Extends the tree by sampling from within the valid region of space
        and steering the node that is closest to the random sample to determine
        how to grow the tree.
        Returns new_nodes which will later be checked before being added to the
        tree, new_u is the control input used and can be recorded.
        """
        samp = self.sample(opts)
        closest = self.nearestNode(samp)
        new_nodes, new_u = self.steer(closest, samp, opts)
        return new_nodes, new_u

    def treeExpansion(self, opts):
        """Main function to grow the tree. Will grow for max_iter or until the
        path from start to goal or vice versa is within min_dist of destination.
        Configures tree and calls correct expansion function depending on desired
        direction of growth.
        Required options:
            direction
        """
        if opts['direction'] == 'forward':
            self.start = RRTNode(self.start, opts=opts) # overwrite self.start
            self.node_list = [self.start]
            self.treeForwardExpansion(opts)
        elif opts['direction'] == 'backward':
            # switch start and goal
            # to grow the tree bacwards list the end point as the start
            tmp = np.copy(self.start)
            self.start = RRTNode(self.goal, opts=opts)
            self.node_list = [self.start]
            self.goal = tmp
            self.treeBackwardExpansion(opts)

    def treeForwardExpansion(self, opts):
        """Expand the tree forward in time from the start to the goal.
        """
        raise NotImplementedError

    def treeBackwardExpansion(self, opts):
        """Expand the tree backward in time from the goal to the start
        (goal is made the start and start the goal).
        While have not hit the maximum number of iterations or close to
        destination, attempts to extend the tree. If the states of the extending
        nodes are not valid, then the extension is rejected.
        Required options;
            min_dist
            max_iter
            track_children
        """
        iter_step = 0
        initial_dist = self.distToGoal(self.start.x)
        best_dist = initial_dist
        best_start_node = None
        while best_dist > opts['min_dist'] and iter_step < opts['max_iter']:
            iter_step+=1
            new_nodes, new_u = self.extend(opts) # u not needed, so kept as _
            valid_extension = True
            # checks if nodes in extension are in valid states
            for new_node in new_nodes:
                if not self.validState(new_node.x):
                    valid_extension = False
                    break
            # rejects node here
            if not valid_extension: continue
            self.node_list.extend(new_nodes)
            for new_node in new_nodes:
                new_node.setU(new_u)
                if opts['track_children']: new_node.parent.addChild(new_node)
                new_dist = self.distToGoal(new_node.x[:2])
                if new_dist < best_dist: best_dist, best_start_node = new_dist, new_node
            printProgressBar('Iterations complete', iter_step, opts['max_iter'])
            printProgressBar('| Distance covered', initial_dist-best_dist, initial_dist, writeover=False)
        assert best_start_node is not None, 'Did not find good node to start from'

    def getPath(self, endnode, reverse=True, gen=False):
        """Given node returns list of previous generations for the node, a path.
        Note: use generators eventually
        endnode     :RRTNode:       node to which get path for
        reverse     :bool:          whether to reorder path
        gen         :bool:          not implemented, use generators
        """
        path = [endnode]
        node = endnode
        while node.parent is not None:
            node = node.parent
            path.append(node)
        if reverse: path = path[::-1]
        return path


    def finalPath(self):
        """Determines which node is closest to the goal by distanceMetric and
        then returns the path to that node.
        """
        final = self.nearestNode(self.goal)
        return self.getPath(final)



