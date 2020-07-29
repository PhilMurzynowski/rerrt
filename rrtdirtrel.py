"""
RRT-Dirtrel Implementation
"""

# python libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#from pydrake.common.containers import namedview
#import pydrake.math as m

# custom classes
from rrt import RRT
from shapes import Rectangle, Ellipse
from collision import CollisionDetection
from setup import Scene, MySystem



class RRT_Dirtrel(RRT):


    class DirtrelNode(RRT.Node):


        def __init__(self, x, parent=None, u=None):
            super().__init__(x, parent)
            # can linearize system at parent since control input is known
            # and propogate Ei
            self.u = None   # control input used at node
            self.A = None   # memory intensive to store matrices
            self.B = None   # but can examine values, variation easily
            self.G = None
            self.S = None
            self.K = None
            self.H = None
            self.E = None

        def createEllipse(self):
            self.ellipse = Ellipse(self.x[0:2], self.E[:2, :2])

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
            #self.S = Q + p.A.T@p.S@p.A - p.A.T@p.S@p.B@np.linalg.inv(R + p.B.T@p.S@p.B)@p.B.T@p.S@p.A
            self.S = Q + self.A.T@p.S@self.A - self.A.T@p.S@self.B@np.linalg.inv(R + self.B.T@p.S@self.B)@self.B.T@p.S@self.A

        def calcKi(self, R):
            self.K = np.linalg.inv(R + self.B.T@self.S@self.B)@self.B.T@self.S@self.A

        # def calcKi(self, R, nextNode):
        #     p = nextNode
        #     self.K = np.linalg.inv(R + self.B.T@p.S@self.B)@self.B.T@p.S@self.A

        def propogateEllipse(self, D, nextNode):
            # let n be next
            abk = self.A-self.B@self.K
            En = abk@self.E@abk.T
            En += abk@self.H@self.G.T + self.G@self.H.T@abk.T
            En += self.G@D@self.G.T
            Hn = abk@self.H + self.G@D

            nextNode.setHi(Hn)
            nextNode.setEi(En)
            nextNode.createEllipse()

        def calcCost(self):
            pass


    def __init__(self, start, goal, system, scene, collision_function):
        super().__init__(start, goal, system, scene)
        self.start = self.DirtrelNode(start) # overwrite self.start
        self.collision = collision_function


    def tree_expansion(self, options):
        iter_step = 0 
        self.node_list = [self.start] 
        best_dist_to_goal = self.dist_to_goal(self.start.x[0:2])

        while best_dist_to_goal > options['epsilon'] and iter_step <= options['max_iter']:
            samp = self.sample(options)
            closest = self.nearest_node(samp)
            x_hat, u_hat = self.steer(closest, samp, options)
            n_min = self.DirtrelNode(x_hat, closest)

            if self.inRegion(x_hat[0:2]) and not self.inObstacle(x_hat[0:2]):
                # n_min only added to tree if not in obstacle
                self.node_list.append(n_min)
                n_min.parent.set_u(u_hat)
                dist = self.dist_to_goal(n_min.x[0:2])
                if dist < best_dist_to_goal:
                    best_dist_to_goal = dist
            iter_step+=1

    def ellipsetree_expansion(self, opts):
        iter_step = 0
        self.start.setEi(opts['E0'])
        self.start.setHi(np.zeros((opts['nx'], opts['nw'])))
        self.node_list = [self.start]
        best_dist_to_goal = self.dist_to_goal(self.start.x[0:2])

        while best_dist_to_goal > opts['epsilon'] and iter_step <= opts['max_iter']:
            #print progress for helpful visual
            print("\rprogress: {prog:>5}%".format(prog=round(100 * iter_step / opts['max_iter'], 3)), end='')
            samp = self.sample(opts)
            closest = self.nearest_node(samp)
            x_hat, u_hat = self.steer(closest, samp, opts)
            n_min = self.DirtrelNode(x_hat, closest)

            # check if centerpoint is in valid region
            if self.inRegion(x_hat[0:2]) and not self.inObstacle(x_hat[0:2]):

                n_min.parent.set_u(u_hat)
                n_min.parent.getJacobians(self.system)
                branch = self.calcEllipseGivenEndNode(n_min, opts)
                # collision method passed in
                valid = True
                for node in branch:
                    for o in self.obstacles:
                        collides = self.collision(node.ellipse, o)
                        if collides:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    self.node_list.append(n_min)

                dist = self.dist_to_goal(n_min.x[0:2])
                if dist < best_dist_to_goal:
                    best_dist_to_goal = dist
            iter_step+=1
        #assert best_dist_to_goal <= epsilon

    def calcEllipseGivenEndNode(self, endnode, opts):
        # can optimize later so don't need to create list for path
        branch = self.getPath(endnode)
        self.calcEllipseGivenPath(branch, opts)
        return branch
        # dont have to set Si and Hi for start as those will never change
        # endnode.setSi(np.zeros((opts['nx'], opts['nx'])))
        # node = endnode
        # while node.parent is not None:
        #     node.calcSi(opts['Q'], opts['R'], node.parent)
        #     node.calcKi(opts['R'])
        #     node = node.parent


    def calcEllipseGivenPath(self, path, opts):

        N = path[-1].n+1
        path[N-1].setSi(np.zeros((opts['nx'], opts['nx'])))
        path[0].setEi(opts['E0'])
        path[0].setHi(np.zeros((opts['nx'], opts['nw'])))
        #print(f'N: {N}')

        for i in range(N-1):
            path[i].getJacobians(self.system)
        for i in range(N-1, 0, -1):
            path[i-1].calcSi(opts['Q'], opts['R'], path[i])
        for i in range(N-1):
            path[i].calcKi(opts['R'])
            path[i].propogateEllipse(opts['D'], path[i+1])

    def drawEllipsoids(self, nodes, hlfmtxpts=False):
        for n in nodes:
            n.ellipse.convertFromMatrix()
            n.ellipse.drawEllipse()
        if hlfmtxpts == True:
            for n in nodes:
                halfmtx_pts = n.ellipse.getHalfMtxPts()
                plt.scatter(halfmtx_pts[0, :], halfmtx_pts[1, :])


