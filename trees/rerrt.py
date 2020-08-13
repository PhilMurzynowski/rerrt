"""
RERRT Implementation
"""

# python libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#from pydrake.common.containers import namedview
#import pydrake.math as m

# custom classes
from trees.nodes import RRTNode, RERRTNode
from trees.rrt import RRT
from utils.shapes import Rectangle, Ellipse
from utils.collision import CollisionDetection
from utils.general import printProgressBar,  pickRandomColor
from utils.visual import Scene
from utils.systems import System
from utils.math import isPSD, isSymmetric, getNearPSD, getNearPD

class RERRT(RRT):


    def __init__(self, start, goals, system, scene, collision_function):
        self.start = start
        # multiple goals only currently used for backwards RRT
        self.goals = goals
        # set first goal state to default goal
        self.goal = goals[0]
        self.system = system
        self.scene = scene
        self.region = scene.region
        self.obstacles = scene.obstacles
        self.poly = [self.region] + self.obstacles
        self.collision = collision_function

    def nodeCollision(self, node):
        for o in self.obstacles:
            if self.collision(node.ellipse, o): return True
        return False

    def branchCollision(self, branch):
        # collision method passed in
        for node in branch:
            if self.nodeCollision(node): return True
        return False

    def nearestReachableState(self, new_location):
        # returns node and key identifying best reachable state
        L = len(self.node_list)
        smallest_distance = np.Inf
        reaching_node = None
        best_reach = None
        for i in range(L):
            for key, reach in self.node_list[L-1-i].reachable.items():
                distance = self.distance_metric(new_location, reach)
                if distance < smallest_distance:
                    smallest_distance = distance
                    reaching_node = self.node_list[L-1-i]
                    best_reach = key
        return reaching_node, best_reach, smallest_distance

    def extendReachable(self, opts):
        # performs sampling and returns nearest reachable state
        # if nearest state is a node not a reachable state, tries again
        best_reach_dist = np.Inf
        best_node_dist = np.Inf
        extra_attempts = -1
        while best_node_dist <= best_reach_dist:
            samp = self.sample(opts)
            reaching_node, key, best_reach_dist = self.nearestReachableState(samp)
            node, best_node_dist = self.nearest_node(samp, get_dist=True)
            extra_attempts += 1
        new_node = RERRTNode(reaching_node.popReachable(key), reaching_node, opts=opts)
        # after conversion to node no longer counted as reachable state
        # abusing key and idx here, whoops
        return new_node, opts['input_actions'][key], extra_attempts

    def extendReachableMultiTimeStep(self, opts):
        # performs sampling and returns nearest reachable state
        # if nearest state is a node not a reachable state, tries again
        best_reach_dist = np.Inf
        best_node_dist = np.Inf
        extra_attempts = -1
        while best_node_dist <= best_reach_dist:
            samp = self.sample(opts)
            reaching_node, key, best_reach_dist = self.nearestReachableState(samp)
            node, best_node_dist = self.nearest_node(samp, get_dist=True)
            extra_attempts += 1
        new_nodes = []
        # remove reachable state, inefficient because recalculating here
        # after conversion to node no longer counted as reachable state
        # abusing key and idx here, whoops
        # not sure if using popReachable here could lead to bugs, shouldnt
        reaching_node.popReachable(key)
        parent = reaching_node
        for i in range(opts['extend_by']):
            xsim = self.system.simulate(parent.x, opts['input_actions'][key], 1, opts['direction'])
            new_nodes.append(RERRTNode(xsim, parent, opts=opts))
            parent = new_nodes[-1]
        return new_nodes, opts['input_actions'][key], extra_attempts

    def extend(self, opts):
        # returns a node and the control input used
        samp = self.sample(opts)
        closest = self.nearest_node(samp)
        new_x, new_u = self.steer(closest, samp, opts)
        new_node = RERRTNode(new_x, closest, opts=opts)
        return new_node, new_u

    def extendMultiTimeStep(self, opts):
        # returns a node and the control input used
        samp = self.sample(opts)
        closest = self.nearest_node(samp)
        extension, new_u = self.steerMultiTimeStep(closest, samp, opts)
        new_nodes = [RERRTNode(extension[0], closest, opts=opts)]
        for i in range(1, opts['extend_by']):
            new_nodes.append(RERRTNode(extension[i], new_nodes[-1], opts=opts))
        return new_nodes, new_u

    def ellipseTreeExpansion(self, opts):
        if opts['direction'] == 'forward':
            self.start = RERRTNode(self.start, opts=opts) # overwrite self.start
            self.node_list = [self.start]
            self.start.setEi(opts['E0'])
            self.start.setHi(np.zeros((opts['nx'], opts['nw'])))
            self.ellipseTreeForwardExpansion(opts)
        elif opts['direction'] == 'backward':
            # switch start and goal
            # to grow the tree bacwards list the end point as the start
            self.goal = np.copy(self.start)
            self.starts = [RERRTNode(x, opts=opts) for x in self.goals]
            self.node_list = self.starts.copy()
            # self.starts are the goals here, growing backwards, apologies aha
            for i in range(len(self.starts)):
                self.starts[i].setSi(np.zeros((opts['nx'], opts['nx'])))
                self.starts[i].calcReachableMultiTimeStep(self.system, opts)
                #self.starts[i].plotNode(new_figure=True)
            self.ellipseTreeBackwardExpansion(opts)

    def ellipseTreeForwardExpansion(self, opts):
        iter_step = 0
        best_dist_to_goal = self.dist_to_goal(self.start.x[0:2])
        while best_dist_to_goal > opts['epsilon'] and iter_step < opts['max_iter']:
            iter_step+=1
            printProgressBar(iter_step, opts['max_iter'])
            new_node, new_u  = self.extend(opts)
            # check if centerpoint is in valid region
            if not self.inRegion(new_node.x[0:2]) or self.inObstacle(new_node.x[0:2]):
                # invalid node
                continue
            new_node.parent.set_u(new_u)
            new_node.parent.getJacobians(self.system)
            branch = self.calcEllipsesGivenEndNode(new_node, opts)
            # check for new collisions after recalculating ellipsoids
            if not self.branchCollision(branch):
                self.node_list.append(new_node)
                new_dist = self.dist_to_goal(new_node.x[0:2])
                if new_dist < best_dist_to_goal: best_dist_to_goal = new_dist

    def ellipseTreeBackwardExpansion(self, opts):
        iter_step = 0
        initial_dist = self.dist_to_goal(self.starts[0].x[:2])
        best_dist = initial_dist
        best_start_node = None
        while best_dist > opts['epsilon'] and iter_step < opts['max_iter']:
            iter_step+=1
            new_nodes, new_u, extra_attempts = self.extendReachableMultiTimeStep(opts)
            iter_step += extra_attempts
            valid_extension = True
            for new_node in new_nodes:
                if not self.inRegion(new_node.x[0:2]) or self.inObstacle(new_node.x[0:2]):
                    # invalid node
                    valid_extension = False
            if not valid_extension: continue
            for new_node in new_nodes:
                new_node.set_u(new_u)
                new_node.getJacobians(self.system)
                new_node.calcSi(opts['Q'], opts['R'], new_node.parent)
                new_node.calcKi(opts['R'], new_node.parent)
            valid_propogation = self.repropagateEllipses(new_nodes[-1], opts)
            if valid_propogation:
                self.node_list.extend(new_nodes)
                for new_node in new_nodes:
                    if opts['track_children']: new_node.parent.addChild(new_node)
                    new_node.calcReachableMultiTimeStep(self.system, opts)
                    new_dist = self.dist_to_goal(new_node.x[:2])
                    if new_dist < best_dist: best_dist, best_start_node = new_dist, new_node
            printProgressBar('Iterations complete', iter_step, opts['max_iter'])
            printProgressBar('| Distance covered', initial_dist-best_dist, initial_dist, writeover=False)
        # repoprogate from best start node for accurate graphing of final path
        assert best_start_node is not None, 'Did not find good node to start from'
        final_propogation_valid = self.repropagateEllipses(best_start_node, opts)
        # below assertion triggered when best start node is node from mid-extension, hlfmtx pts allowed bad ellipses
        assert final_propogation_valid, 'Collision checking likely not thorough enough'

    def repropagateEllipses(self, startnode, opts):
        # currently this method is only valid for backwards RRT
        # avoid creating lists by checking collision immediately after propogating ellipse
        startnode.setHi(np.zeros((opts['nx'], opts['nw'])))
        startnode.setEi(opts['E0'])
        if self.nodeCollision(startnode): return False
        node = startnode
        while node.parent is not None:
            node.propogateEllipse(opts['D'], node.parent)
            if self.nodeCollision(node.parent): return False
            node = node.parent
        return True

    def calcEllipsesGivenEndNode(self, endnode, opts):
        # can optimize later so don't need to create list for path
        branch = self.getPath(endnode)
        self.calcEllipseGivenPath(branch, opts)
        return branch

    def calcEllipseGivenPath(self, path, opts):

        N = path[-1].n+1
        path[N-1].setSi(np.zeros((opts['nx'], opts['nx'])))
        path[0].setEi(opts['E0'])
        path[0].setHi(np.zeros((opts['nx'], opts['nw'])))

        for i in range(N-1):
            path[i].getJacobians(self.system)
        for i in range(N-1, 0, -1):
            path[i-1].calcSi(opts['Q'], opts['R'], path[i])
        for i in range(N-1):
            path[i].calcKi(opts['R'], path[i+1])
            path[i].propogateEllipse(opts['D'], path[i+1])

    def drawEllipsoids(self, nodes, hlfmtxpts=False, color='gray', fraction=1.00):
        freq = 1/fraction
        for i, n in enumerate(nodes):
            if i%freq==0:
                if n.ellipse is None:
                    # if a goalstate was never propogated from will not have an ellipse set
                    continue
                n.ellipse.convertFromMatrix()
                n.ellipse.drawEllipse(color=color)
                if hlfmtxpts:
                    halfmtx_pts = n.ellipse.getHalfMtxPts()
                    plt.scatter(halfmtx_pts[0, :], halfmtx_pts[1, :])

    def drawEllipsoidTree(self, opts):
        # ellipses at each node currently only keep the last propogated ellipse
        # otherwise would be extremely memory intensive
        # so if the path branches only the last propogated value will be kept
        # basic way to draw all ellipses with backwards RRT must:
        # find all valid start nodes and for each start node:
        # reprogate from that start node and draw ellipses
        if not opts['track_children']:
            raise RuntimeError('Enable track_children')
        if opts['direction'] == 'backward':
            startnodes = (n for n in self.node_list if len(n.children)==0)
            for startnode in startnodes:
                valid_propagation = self.repropagateEllipses(startnode, opts)
                assert valid_propagation, 'BUG'
                path = self.getPath(startnode, reverse=False)
                self.drawEllipsoids(path, color=pickRandomColor())
        elif opts['direction'] == 'forward':
            raise NotImplementedError('Not implemented yet for forward RRT')


    def drawReachable(self, nodes, color='limegreen', fraction=1.00):
        freq = 1/fraction
        plotnum = 0
        for node in nodes:
            for key, reach in node.reachable.items():
                plotnum += 1
                if plotnum%freq==0:
                    plt.plot([node.x[0], reach[0]], [node.x[1], reach[1]], color=color)
