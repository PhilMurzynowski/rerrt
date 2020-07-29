"""
RRT-Dirtrel Implementation
"""

# python libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pydrake.systems.framework import (BasicVector_, LeafSystem_)
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.common.containers import namedview
import pydrake.math as m

from pydrake.all import (AutoDiffXd, autoDiffToGradientMatrix, initializeAutoDiff)

# custom classes
from rrt import RRT
from shapes import Rectangle, Ellipse
from collision import CollisionDetection



class Scene:
    
    def __init__(self, start, goal, region, obstacles):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.region = region
        self.obstacles = obstacles

    def plot_scene(self):
        ax = plt.gca()
        for r in self.obstacles:
            rect = patches.Rectangle(r.v1, r.w, r.h, r.angle, color='cyan')
            ax.add_artist(rect)
        plt.axis([self.region.v1[0]-0.5, self.region.v4[0]+0.5, self.region.v1[1]-0.5, self.region.v4[1]+0.5])
        plt.plot(self.start[0], self.start[1], "xr", markersize=10)
        plt.plot(self.goal[0], self.goal[1], "xb", markersize=10)
        plt.legend(('start', 'goal'), loc='upper left')
        plt.gca().set_aspect('equal')
    
    def show_scene(self):
        plt.figure()
        plot_scene()
        plt.tight_layout()

"""Dynamics 

Using simplified custom system class
"""

class MySystem():

    def __init__(self, sys_opts):
        self.dt = sys_opts['dt']
        self.nx = sys_opts['nx']
        self.nu = sys_opts['nu']
        self.nw = sys_opts['nw']

    def dynamics(self, state, input, uncertainty=None):

        if uncertainty is None:
            uncertainty = np.zeros((self.nw, 1))

        #["x_pos", "y_pos", "heading", "speed", "steer_angle"]
        x_next = np.array([
            state[0] + self.dt*(state[3]*np.cos(state[2])),
            state[1] + self.dt*(state[3]*np.sin(state[2])),
            state[2] + self.dt*(state[3]*np.tan(state[4] + uncertainty[0])),
            state[3] + self.dt*(input[0]),
            state[4] + self.dt*(input[1] + uncertainty[1])])
        return x_next

    def nextState(self, state, input):
        # wrapper
        return self.dynamics(state, input)

    def getJacobians(self, x, u, w=None):

        if w is None:
            w = np.zeros((self.nw, 1))

        # format for autodiff
        xuw = np.vstack((x, u, w))
        xuw_autodiff = initializeAutoDiff(xuw)
        # name and split here for readability
        x_autodiff = xuw_autodiff[:self.nx, :]
        u_autodiff = xuw_autodiff[self.nx:self.nx+self.nu, :]
        w_autodiff = xuw_autodiff[self.nx+self.nu:, :]
        x_next_autodiff = self.dynamics(x_autodiff, u_autodiff, w_autodiff)
        # nice function organize for us and return gradient matrix
        x_next_gradient = autoDiffToGradientMatrix(x_next_autodiff)
        # split into Ai, Bi, Gi
        Ai = x_next_gradient[:, 0:self.nx]
        Bi = x_next_gradient[:, self.nx:self.nx+self.nu]
        Gi = x_next_gradient[:, self.nx+self.nu:]

        return Ai, Bi, Gi


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
            if iter_step%options['plot_freq']==0 or best_dist_to_goal <= options['epsilon']:
                self.draw_sceneandtree(options['plot_size'])
                plt.title(f'Iteration: {iter_step}\nDistance to goal: {best_dist_to_goal}')
            
            iter_step+=1
        #assert best_dist_to_goal <= epsilon

    def ellipsetree_expansion(self, opts):
        iter_step = 0 
        self.start.setEi(opts['E0'])
        self.start.setHi(np.zeros((opts['nx'], opts['nw'])))
        self.node_list = [self.start] 
        best_dist_to_goal = self.dist_to_goal(self.start.x[0:2])

        while best_dist_to_goal > opts['epsilon'] and iter_step <= opts['max_iter']:
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
                        #print(collides)
                        if collides:
                            valid = False
                            break
                    if not valid:
                        break
                #print(iter_step)
                if valid:
                    #print('b')
                    self.node_list.append(n_min)

                dist = self.dist_to_goal(n_min.x[0:2])
                if dist < best_dist_to_goal:
                    best_dist_to_goal = dist
            if iter_step%opts['plot_freq']==0 or best_dist_to_goal <= opts['epsilon']:
                self.draw_sceneandtree(opts['plot_size'])
                plt.title(f'Iteration: {iter_step}\nDistance to goal: {best_dist_to_goal}')
            
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

def get_rotation_mtx(angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


# Initialize start, goal, bounds on area
start = [10, 10]
start_state = np.array(start + [-np.pi*2/3, 5, 0]).reshape(5, 1)     
goal = np.array([[0], [0]])                               
region = Rectangle([-5, -5], 20, 20)                        


# initialize obstacles
rects = []
ang1 = 120.0
ang2 = 30.0
# arguments: rectangle bottom left corner, width, height, angle from horizontal (deg)
rects.append(Rectangle([2, 3], 3, 1.5, angle=ang1))
rects.append(Rectangle([7, 4], 2.5, 1.5, angle=ang2))
# Scene describes the start, goal, obstacles, mostly for plotting
scene = Scene(start, goal, region, rects)

sys_opts = {
    'dt': 0.05,
    'nx': 5,
    'nu': 2,
    'nw': 2
    }
sys = MySystem(sys_opts)

col = CollisionDetection()
collision_function = col.selectCollisionChecker('erHalfMtxPts')

# initialize RRT_Dirtrel
tree = RRT_Dirtrel(start_state, scene.goal, sys, scene, collision_function)

# run RRT_Dirtrel
run_options = {
    'epsilon':          1,                              # min dist to goal
    'max_iter':         400,                            # iterations
    'plot_freq':        200,                            # how often to plot tree expansion (num iterations)
    'plot_size':        (10, 10),                       # plot size
    'goal_sample_rate': 0.5,                            # favor tree expansion towards goal
    'input_max':        5,                              # max magnitude of input in any one dimension
    'input_samples':    10,                             # number of random inputs to sample in steering method
    'nx':               sys_opts['nx'],
    'nu':               sys_opts['nu'],
    'nw':               sys_opts['nw'],
    'D':                0.05*np.eye(sys_opts['nw']),
    'E0':               0.05*np.eye(sys_opts['nx']),
    'Ql':               np.eye(sys_opts['nx']),         # use if cost added
    'Rl':               np.eye(sys_opts['nu']),         # ...
    'QlN':              np.eye(sys_opts['nx']),         # ...
    'Q':                np.eye(sys_opts['nx']),
    'R':                np.eye(sys_opts['nu'])}

tree.ellipsetree_expansion(run_options)
final_path = tree.final_path()
tree.draw_path(final_path)
# hlfmtxpts drawing currently is slow
tree.drawEllipsoids(final_path, hlfmtxpts=True)
##tree.drawEllipsoids(tree.node_list)

print('hi')
# tree.tree_expansion(run_options)
# final_path = tree.final_path()
# tree.draw_path(final_path)
# tree.calcEllipseGivenPath(final_path, run_options)

# for n in final_path:
#     print('Testing')
#     print(f'E before: {n.E[:2, :2]}')
#     n.ellipse.convertFromMatrix()
#     print(f'w: {n.ellipse.w}, h: {n.ellipse.h}, angle: {n.ellipse.angle}')
#     n.ellipse.convertToMatrix()
#     print(f'E after:  {n.ellipse.mtx}')
#     print(f'w: {n.ellipse.w}, h: {n.ellipse.h}, angle: {n.ellipse.angle}')
#     n.ellipse.convertFromMatrix()
#     print(f'w: {n.ellipse.w}, h: {n.ellipse.h}, angle: {n.ellipse.angle}')
#     n.ellipse.convertToMatrix()
#     print(f'E after2:  {n.ellipse.mtx}')
    # print(f'i: {n.n}, S: {n.S}, K: {n.K}')
    # #cost is decreasing so makes sense
    # print(f"i: {n.n}, S norm: {np.linalg.norm(n.S, 'fro')}")
    # norm monotonic so something is wrong with drawing perhaps
    # print(f"i: {n.n}, E norm: {np.linalg.norm(n.E, 'fro')}, {np.linalg.norm(n.ellipse.mtx, 'fro')}")

#tree.drawEllipsoids(final_path)
