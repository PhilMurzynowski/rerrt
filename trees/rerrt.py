"""
RERRT Implementation
"""

# python libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# custom
from trees.nodes import RERRTNode
from trees.rrt import RRT
from visuals.helper import printProgressBar


class RERRT(RRT):
    """
    Class to grow robust tree. Makes use of RERRTNode class.
    start           :nparray: (nx x 1)      desired start of trajectory
    goal            :nparray: (nx x 1)      desired end of trajectory
    system          :System:                System object to provide dynamics
    input           :Input:                 Input object configured for desired
                                            input sampling
    scene           :Scene:                 Scene object used to organized region
                                            and obstacles
    collision_func  :CollisionDetection:    function to use for collision
                    :       method     :    detection of ellipse and obstacles

    Required options:
        min_dist
        max_iter
        direction
        track_children
        extend_by
        goal_sample_rate
        sample_dim
        D
        E0
        Q   (if using TVLQR)
        R   (if using TVLQR)

    To run tree growth call tree.treeExpansion where tree is an :RERRT: instance.
    Note: Currently functional for 2D, in process of generalizing.
    """


    def __init__(self, start, goal, system, input_, scene, dist_func, collision_func):
        super().__init__(start, goal, system, input_, scene, dist_func)
        self.setCollisionFunc(collision_func)

    def setCollisionFunc(self, func):
        """Wrapper to set desired collision function
        func    :function:      function to use for collision detection
        """
        self.collision = func

    def nodeCollision(self, node):
        """Checks whether ellipse of node is in collision with obstacles.
        Collision detection method determined during initialization.
        node    :RERRTNode:     node whose ellipse to check for collision
        """
        for o in self.obstacles:
            if self.collision(node.ellipse, o): return True
        return False

    def branchCollision(self, branch):
        """Note: Currently unused as individually checking nodes more efficient.
        Check if ellipses for a branch of nodes is in collision with obstacles.
        branch      :[RERRTNode,]:      list,set,etc. of nodes to check
        """
        for node in branch:
            if self.nodeCollision(node): return True
        return False

    def nearestReachable(self, new_location):
        """Determine the reachable state closest to the given partial state.
        new_location    :nparray: (? x 1)   partial state
        Returns the parameters necessary to identify a reachable state, the node
        which the reachable state originates from and its key for node.reachable,
        as well as distance.
        """
        L = len(self.node_list)
        smallest_distance = np.Inf
        reaching_node = None
        best_reach = None
        for i in range(L):
            for key, reach in self.node_list[L-1-i].reachable.items():
                distance = self.distanceMetric(new_location, reach, self.system)
                if distance < smallest_distance:
                    smallest_distance = distance
                    reaching_node = self.node_list[L-1-i]
                    best_reach = key
        return reaching_node, best_reach, smallest_distance

    def extend(self, opts):
        """Very different function from base class RRT extend.
        Resamples until getting a sample that is closer to a reachable state
        rather than a node that is already in the tree, effectively rebiasing the
        voronoi regions of nodes for faster exploration. Working off of RGRRT.
        Once ascertaining reachable state closest to sample, converts that
        reachable state to an extension of nodes (new_nodes). The reachable state
        can be converted to multiple nodes as it may be a simulation of many time
        steps forward, while currently nodes are always separated by one
        timestep.
        Required options:
            extend_by
            direction
        Returns extension (new_nodes), control input, and num extra samples used
        """
        best_reach_dist = np.Inf
        best_node_dist = np.Inf
        extra_attempts = -1
        while best_node_dist <= best_reach_dist:
            samp = self.sample(opts)
            reaching_node, key, best_reach_dist = self.nearestReachable(samp)
            node, best_node_dist = self.nearestNode(samp, get_dist=True)
            extra_attempts += 1
        new_nodes = []
        # remove reachable state, inefficient because recalculating here
        # after conversion to node no longer counted as reachable state
        # abusing key as an index here as it is an int, whoops
        reaching_node.popReachable(key)
        parent = reaching_node
        for i in range(opts['extend_by']):
            xsim = self.system.simulate(parent.x, self.input.actions[key], 1, opts['direction'])
            new_nodes.append(RERRTNode(xsim, parent, opts=opts))
            parent = new_nodes[-1]
        return new_nodes, self.input.actions[key], extra_attempts

    def treeExpansion(self, opts):
        """Modified RRT's treeExpansion.
        Main function to grow the tree. Will grow for max_iter or until the
        path from start to goal or vice versa is within min_dist of destination.
        Configures tree and calls correct expansion function depending on desired
        direction of growth.
        Required options:
            direction
        """
        if opts['direction'] == 'forward':
            self.start = RERRTNode(self.start, opts=opts) # overwrite self.start
            self.node_list = [self.start]
            self.start.setEi(opts['E0'])
            self.start.setHi(np.zeros((self.system.nx, self.system.nw)))
            self.treeForwardExpansion(opts)
        elif opts['direction'] == 'backward':
            # switch start and goal
            # to grow the tree bacwards list the end point as the start
            tmp = np.copy(self.start)
            self.start = RERRTNode(self.goal, opts=opts)
            self.node_list = [self.start]
            self.goal = tmp
            # reminder, though setting S to be 0s at self.start
            # growing backwards so actually setting 0s at destination
            self.start.setSi(np.zeros((self.system.nx, self.system.nx)))
            self.start.getReachable(self.system, self.input, opts)
            self.treeBackwardExpansion(opts)

    def treeForwardExpansion(self, opts):
        """Expand the tree forward in time from the goal to the start.
        While have not hit the maximum number of iterations or close to
        destination, attempts to extend the tree. If the states of the extending
        nodes are not valid or propagated ellipses collide with obstacles then
        the extension is rejected.
        Required options;
            min_dist
            max_iter
            track_children
        """
        raise NotImplementedError('Update pending')
        #iter_step = 0
        #best_distToGoal = self.distToGoal(self.start.x[0:2])
        #while best_distToGoal > opts['min_dist'] and iter_step < opts['max_iter']:
        #    iter_step+=1
        #    printProgressBar(iter_step, opts['max_iter'])
        #    new_node, new_u  = self.extend(opts)
        #    # check if centerpoint is in valid region
        #    if not self.inRegion(new_node.x[0:2]) or self.inObstacle(new_node.x[0:2]):
        #        # invalid node
        #        continue
        #    new_node.parent.setU(new_u)
        #    new_node.parent.getJacobians(self.system)
        #    branch = self.calcEllipsesGivenEndNode(new_node, opts)
        #    # check for new collisions after recalculating ellipsoids
        #    if not self.branchCollision(branch):
        #        self.node_list.append(new_node)
        #        new_dist = self.distToGoal(new_node.x[0:2])
        #        if new_dist < best_distToGoal: best_distToGoal = new_dist

    def treeBackwardExpansion(self, opts):
        """Expand the tree backward in time from the goal to the start.
        (goal has been made the start and start the goal).
        While have not hit the maximum number of iterations or close to
        destination, attempts to extend the tree. If the states of the extending
        nodes are not valid or propagated ellipses collide with obstacles then
        the extension is rejected.
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
            # obtain extension
            new_nodes, new_u, extra_attempts = self.extend(opts)
            iter_step += extra_attempts
            valid_extension = True
            # checks if nodes in extension are valid states
            for new_node in new_nodes:
                if not self.validState(new_node.x):
                    valid_extension = False
            # can reject node here if states not valid
            # proceeds with more expensive robust ellipse check
            if not valid_extension: continue
            # obtain parameters necessary to attempt propogating ellipse
            for new_node in new_nodes:
                new_node.setU(new_u)
                new_node.getJacobians(self.system)
                new_node.calcSi(opts['Q'], opts['R'], new_node.parent)
                new_node.calcKi(opts['R'], new_node.parent)
            robust = self.robustnessCheck(new_nodes[-1], opts)
            # if fails robustness check:
            # rejects extension continuing to attempting next extension
            if robust:
                self.node_list.extend(new_nodes)
                for new_node in new_nodes:
                    if opts['track_children']: new_node.parent.addChild(new_node)
                    new_node.getReachable(self.system, self.input, opts)
                    new_dist = self.distToGoal(new_node.x[:2])
                    if new_dist < best_dist: best_dist, best_start_node = new_dist, new_node
            printProgressBar('Iterations complete', iter_step, opts['max_iter'])
            printProgressBar('| Distance covered', initial_dist-best_dist, initial_dist, writeover=False)
        # repoprogate from best start node for accurate graphing of final path
        assert best_start_node is not None, "Didn't find good node to start from"
        final_propogation_valid = self.robustnessCheck(best_start_node, opts)
        # below assertion triggered when best start node is node from
        # mid-extension and collision method allowed bad ellipses
        assert final_propogation_valid, 'Collision checking likely not thorough enough'

    def robustnessCheck(self, tipnode, opts):
        """Critical function in RERRT, checking robustness.
        Warning: Currently this method is only valid for backwards RRT.
        For forward RRT:
            NotImplemented
        For backward RRT:
            Given node (tipnode) at the very end of newest extension (which has
            not yet been added to the tree), considers this node as a starting
            node from which ellipsoids will be propogated, setting initial E0
            and H, and proceeding to propogate ellipsoids through previous
            generations of parents. Collision is checked immediately after each
            propagation up one generation meaning do not need to propogate and
            check all nodes back to root of the tree if there is a collision
            before it.
        tipnode     :RERRTNode:     node at the tip of newest attempted extension
                                    do not need to pass in entire extension as
                                    rest of extension are simply previous
                                    generations of tipnode, i.e. tipnode.parent,
                                    tipnode.parent.parent etc., number of prev
                                    generations dependent on extend_by
        Required options:
            direction
        Returns True if propogated ellipsoids do not collide, False otherwise.
        """
        if opts['direction'] == 'forward':
            raise NotImplementedError
        elif opts['direction'] == 'backward':
            tipnode.setHi(np.zeros((self.system.nx, self.system.nw)))
            tipnode.setEi(opts['E0'])
            # check if starting ellipsoid is in collision
            if self.nodeCollision(tipnode): return False
            node = tipnode
            # propagate up one generation and check if ellipsoid collides
            while node.parent is not None:
                node.propogateEllipse(opts['D'], node.parent)
                if self.nodeCollision(node.parent): return False
                node = node.parent
            return True

    def calcEllipsesGivenEndNode(self, endnode, opts):
        """Note: Currently not in use, inefficient to create path [a list].
                 May be put back into use once getPath updated with generators.
        """
        branch = self.getPath(endnode)
        self.calcEllipseGivenPath(branch, opts)
        return branch

    def calcEllipseGivenPath(self, path, opts):
        """Note: Currently not in use, inefficient to create path [a list].
                 May be put back into use once getPath updated with generators.
        """
        N = path[-1].n+1
        path[N-1].setSi(np.zeros((self.system.nx, self.system.nx)))
        path[0].setEi(opts['E0'])
        path[0].setHi(np.zeros((self.system.nx, self.system.nw)))

        for i in range(N-1):
            path[i].getJacobians(self.system)
        for i in range(N-1, 0, -1):
            path[i-1].calcSi(opts['Q'], opts['R'], path[i])
        for i in range(N-1):
            path[i].calcKi(opts['R'], path[i+1])
            path[i].propogateEllipse(opts['D'], path[i+1])




