"""
RRT-Dirtrel Implementation
"""

# python libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#from pydrake.common.containers import namedview
#import pydrake.math as m

# custom classes
from rrt import RRT
from shapes import Rectangle, Ellipse
from collision import CollisionDetection
from setup import printProgressBar, Scene, MySystem



class RRT_Dirtrel(RRT):


    class DirtrelNode(RRT.Node):


        def __init__(self, x, parent=None, u=None):
            super().__init__(x, parent)
            # can linearize system at parent since control input is known
            # and propogate Ei
            self.u = None   # control input used at node
            self.A = None   # memory intensive to store matrices
            self.B = None   # but can examine values, variation easily
            self.G = None   # as well as save computation
            self.S = None
            self.K = None
            self.H = None
            self.E = None
            self.ellipse = None
            self.reachable = {}

        def createEllipse(self):
            # let EE bet E^1/2
            EE = scipy.linalg.sqrtm(self.E)
            # take first 2 dimensions and project
            A = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
            EEw = np.linalg.pinv(A@EE)
            ellipse_projection = np.linalg.inv(EEw.T@EEw)
            self.ellipse =  Ellipse(self.x[0:2], ellipse_projection)

        def setEi(self, Ei):
            self.E = Ei
            self.createEllipse()

        def setHi(self, Hi):
            self.H = Hi

        def setSi(self, Si):
            self.S = Si

        def set_u(self, u):
            self.u = u

        def getJacobians(self, system):
            # wrapper function
            self.A, self.B, self.G = system.getJacobians(self.x, self.u)

        def calcSi(self, Q, R, nextNode):
            p = nextNode
            self.S = Q + self.A.T@p.S@self.A - self.A.T@p.S@self.B@np.linalg.inv(R + self.B.T@p.S@self.B)@self.B.T@p.S@self.A

        def calcKi(self, R, nextNode):
             p = nextNode
             self.K = np.linalg.inv(R + self.B.T@p.S@self.B)@self.B.T@p.S@self.A

        def propogateEllipse(self, D, nextNode):
            abk = self.A-self.B@self.K
            En = abk@self.E@abk.T
            En += abk@self.H@self.G.T + self.G@self.H.T@abk.T
            En += self.G@D@self.G.T
            Hn = abk@self.H + self.G@D

            nextNode.setHi(Hn)
            nextNode.setEi(En)

        def calcReachable(self, system, opts):
            for i, action in enumerate(opts['input_actions']):
                if opts['direction'] == 'forward':
                    self.reachable[i] = system.nextState(self.x, action)
                elif opts['direction'] == 'backward':
                    self.reachable[i] = system.prevState(self.x, action)

        def popReachable(self, idx):
            return self.reachable.pop(idx)

        def calcCost(self):
            pass


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
        for i in range(L):
            for key, reach in self.node_list[L-1-i]:
                distance = self.distance_metric(new_location, reach)
                if distance < smallest_distance:
                    smallest_distance = distance
                    reaching_node = self.node_list[L-1-i]
                    closest_reach = key
        return reaching_node, key, smallest_distance

    def extendReachableState(self, opts):
        # performs sampling and returns nearest reachable state
        # if nearest state is a node not a reachable state, tries again
        best_reach_dist = np.Inf
        best_node_dist = np.Inf
        while best_node_dist <= best_reach_dist:
            samp = self.sample(opts)
            reaching_node, key, best_reach_dist = self.nearestReachableState(samp)
            node, best_node_dist = self.nearest_node(samp, get_dist=True)
        raise NotImplementedError

    def extend(self, opts):
        # returns a node and the control input used
        samp = self.sample(opts)
        closest = self.nearest_node(samp)
        new_x, new_u = self.steer(closest, samp, opts)
        new_node = self.DirtrelNode(new_x, closest)
        return new_node, new_u

    def extendMultiTimeStep(self, opts):
        # returns a node and the control input used
        samp = self.sample(opts)
        closest = self.nearest_node(samp)
        extension, new_u = self.steerMultiTimeStep(closest, samp, opts)
        new_nodes = [self.DirtrelNode(extension[0], closest)]
        for i in range(1, opts['extend_by']):
            new_nodes.append(self.DirtrelNode(extension[i], new_nodes[-1]))
        return new_nodes, new_u

    def ellipseTreeExpansion(self, opts):
        if opts['direction'] == 'forward':
            self.start = self.DirtrelNode(self.start) # overwrite self.start
            self.node_list = [self.start]
            self.start.setEi(opts['E0'])
            self.start.setHi(np.zeros((opts['nx'], opts['nw'])))
            self.ellipseTreeForwardExpansion(opts)
        elif opts['direction'] == 'backward':
            # switch start and goal
            # to grow the tree bacwards list the end point as the start
            self.goal = np.copy(self.start)
            self.starts = [self.DirtrelNode(x) for x in self.goals]
            self.node_list = self.starts.copy()
            # self.starts are the goals here, growing backwards, apologies aha
            for i in range(len(self.starts)):
                self.starts[i].setSi(np.zeros((opts['nx'], opts['nx'])))
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
            printProgressBar('Iterations complete', iter_step, opts['max_iter'])
            printProgressBar('| Distance covered', initial_dist-best_dist, initial_dist, writeover=False)
            new_nodes, new_u = self.extendMultiTimeStep(opts)
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
            valid_propogation = self.repropogateEllipses(new_nodes[-1], opts)
            if valid_propogation:
                self.node_list.extend(new_nodes)
                for new_node in new_nodes:
                    new_node.calcReachable(self.system, opts)
                    new_dist = self.dist_to_goal(new_nodes[-1].x[0:2])
                    if new_dist < best_dist: best_dist, best_start_node = new_dist, new_node
        # repoprogate from best start node for accurate graphing
        if best_start_node is not None:
            final_propogation_valid = self.repropogateEllipses(best_start_node, opts)
            assert final_propogation_valid

    def repropogateEllipses(self, startnode, opts):
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

    def drawEllipsoids(self, nodes, hlfmtxpts=False):
        for n in nodes:
            n.ellipse.convertFromMatrix()
            n.ellipse.drawEllipse()
        if hlfmtxpts == True:
            for n in nodes:
                halfmtx_pts = n.ellipse.getHalfMtxPts()
                plt.scatter(halfmtx_pts[0, :], halfmtx_pts[1, :])


