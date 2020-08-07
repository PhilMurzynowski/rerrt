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
            # used to store values when checking for collision
            # these values are committed if propogation of branch valid
            # tmd for timid
            self.H_timid, self.E_timid, self.ellipse_timid = None, None, None

        def createEllipse(self, E):
            # let EE bet E^1/2
            EE = scipy.linalg.sqrtm(E)
            # take first 2 dimensions and project
            A = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
            EEw = np.linalg.pinv(A@EE)
            ellipse_projection = np.linalg.inv(EEw.T@EEw)
            return Ellipse(self.x[0:2], ellipse_projection)

        def setEi(self, Ei):
            self.E = Ei
            self.ellipse = self.createEllipse(self.E)

        def setHi(self, Hi):
            self.H = Hi

        #def setTimidValues(self, Hi, Ei):
        #    self.H_timid = Hi
        #    self.E_timid = Ei
        #    self.ellipse_timid = self.createEllipse(self.E_timid)

        #def adoptTimidValues(self):
        #    self.H = self.H_timid.copy()
        #    self.E = self.E_timid.copy()
        #    self.ellipse = self.ellipse_timid.copy()

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

        def propogateEllipse(self, D, nextNode, timid=False):
            # timid option determines whether to commit propogation
            # to main variable i.e. setEi or store the values aside (e.g. to check collision before committing) 
            # let n be next
            E = self.E_timid if timid else self.E
            H = self.H_timid if timid else self.H
            abk = self.A-self.B@self.K
            En = abk@E@abk.T
            En += abk@H@self.G.T + self.G@H.T@abk.T
            En += self.G@D@self.G.T
            Hn = abk@H + self.G@D

            if timid:
                nextNode.setTimidValues(Hn, En)
            else:
                nextNode.setHi(Hn)
                nextNode.setEi(En)

        def calcCost(self):
            pass


    def __init__(self, start, goal, system, scene, collision_function):
        super().__init__(start, goal, system, scene)
        self.collision = collision_function

    def nodeCollision(self, node, timid=False):
        ellipse = node.ellipse_timid if timid else node.ellipse
        for o in self.obstacles:
            if self.collision(ellipse, o): return True
        return False

    def branchCollision(self, branch, timid=False):
        # collision method passed in
        for node in branch:
            if self.nodeCollision(node, timid): return True
        return False

    def extend(self, opts):
        # returns a node and the control input used
        samp = self.sample(opts)
        closest = self.nearest_node(samp)
        new_x, new_u = self.steer(closest, samp, opts)
        new_node = self.DirtrelNode(new_x, closest)
        return new_node, new_u

    def ellipseTreeExpansion(self, opts):
        if opts['direction'] == 'forward':
            self.start = self.DirtrelNode(self.start) # overwrite self.start
            self.node_list = [self.start]
            self.ellipseTreeForwardExpansion(opts)
        elif opts['direction'] == 'backward':
            # switch start and goal
            # to grow the tree bacwards list the end point as the start
            tmp1 = np.copy(self.start)
            tmp2 = np.copy(self.goal)
            self.start = self.DirtrelNode(tmp2) # overwrite self.start
            self.goal = tmp1
            self.node_list = [self.start]
            self.ellipseTreeBackwardExpansion(opts)

    def ellipseTreeForwardExpansion(self, opts):
        iter_step = 0
        self.start.setEi(opts['E0'])
        self.start.setHi(np.zeros((opts['nx'], opts['nw'])))
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
        # self.start is the goal here, growing backwards, apologies aha
        self.start.setSi(np.zeros((opts['nx'], opts['nx'])))
        best_dist_to_goal = self.dist_to_goal(self.start.x[0:2])
        while best_dist_to_goal > opts['epsilon'] and iter_step < opts['max_iter']:
            iter_step+=1
            printProgressBar(iter_step, opts['max_iter'])
            new_node, new_u = self.extend(opts)
            if not self.inRegion(new_node.x[0:2]) or self.inObstacle(new_node.x[0:2]):
                # invalid node
                continue
            new_node.set_u(new_u)
            new_node.getJacobians(self.system)
            valid_propogation = self.repropogateEllipses(new_node, opts)
            if valid_propogation:
                self.node_list.append(new_node)
                #self.adoptNewEllipses(new_node)
                new_dist = self.dist_to_goal(new_node.x[0:2])
                if new_dist < best_dist_to_goal: best_dist_to_goal, best_start_node = new_dist, new_node
        # repoprogate from best start node for accurate graphing
        final_propogation_valid = self.repropogateEllipses(best_start_node, opts)
        assert final_propogation_valid

    def repropogateEllipses(self, startnode, opts):
        # currently this method is only valid for backwards RRT
        # avoid creating lists by checking collision immediately after propogating ellipse
        startnode.calcSi(opts['Q'], opts['R'], startnode.parent)
        startnode.calcKi(opts['R'], startnode.parent)
        # timid option does not commit new ellipse values yet
        timid = False
        #startnode.setTimidValues(np.zeros((opts['nx'], opts['nw'])), opts['E0'])
        startnode.setHi(np.zeros((opts['nx'], opts['nw'])))
        startnode.setEi(opts['E0'])
        if self.nodeCollision(startnode, timid=timid): return False
        node = startnode
        while node.parent is not None:
            node.propogateEllipse(opts['D'], node.parent, timid=timid)
            if self.nodeCollision(node.parent, timid=timid): return False
            node = node.parent
        return True

    def adoptNewEllipses(self, startnode):
        # when repropogating ellipses new ellipses only adopted
        # after all ellipses have been checked for collision
        node = startnode
        while node.parent is not None:
            node.parent.adoptTimidValues()
            node = node.parent

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


