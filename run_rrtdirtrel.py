"""
Run RRT Dirtrel
"""

# python libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#from pydrake.common.containers import namedview
#import pydrake.math as mr

# custom classes
from rrt import RRT
from shapes import Rectangle, Ellipse
from collision import CollisionDetection
from setup import Scene, MySystem
from rrtdirtrel import RRT_Dirtrel


# Initialize start, goal, bounds on area
start = [12.5, 12.5]
goal = [0, 0]

# start_state used for forward expansion as start
# goal_states used for backward expansion as start
#   multiple goals states allow for better tree growth using backwards RRT with nonlinear systems
# currently RRT does not require both starting state
# and ending state to be satisfied
# one must be satisfied and returns solution closest to other desired state
start_state = np.array(start + [-np.pi*2/3, 5, 0]).reshape(5, 1)
num_goal_states = 20
eps = 1e-4
goal_states = [np.array([goal[0]+eps*np.cos(theta)]+[goal[1]+eps*np.sin(theta)]+[(theta+np.pi)%(2*np.pi), 5, 0]).reshape(5, 1) for theta in np.linspace(-np.pi, np.pi, num_goal_states, endpoint=False)]
region = Rectangle([-5, -5], 20, 20)


# initialize obstacles
rects = []
# arguments: rectangle bottom left corner, width, height, angle from horizontal (deg)
rects.append(Rectangle([2, 3], 3, 1.5, angle=120.0))
rects.append(Rectangle([7, 4], 2.5, 1.5, angle=30.0))

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
tree = RRT_Dirtrel(start_state, goal_states, sys, scene, collision_function)

# run RRT_Dirtrel
run_options = {
    'epsilon':          1,                              # min dist to goal
    'max_iter':         500,                            # iterations
    'plot_freq':        None,                           # how often to plot tree expansion (num iterations)
    'plot_size':        (10, 10),                       # plot size
    'direction':        'backward',                    # determine whether to propogate tree forwards or backwards
    'goal_sample_rate': 0.5,                            # favor tree expansion towards goal
    'input_max':        5,                              # max magnitude of input in any one dimension
    'input_samples':    20,                             # number of random inputs to sample in steering method
    'nx':               sys_opts['nx'],
    'nu':               sys_opts['nu'],
    'nw':               sys_opts['nw'],
    'D':                0.05*np.eye(sys_opts['nw']),
    'E0':               0.05*np.eye(sys_opts['nx']),
    'Ql':               np.eye(sys_opts['nx']),         # use if robust cost from DIRTREL paper added
    'Rl':               np.eye(sys_opts['nu']),         # ...
    'QlN':              np.eye(sys_opts['nx']),         # ...
    'Q':                np.diag((5, 5, 0, 0, 0)),
    'R':                np.eye(sys_opts['nu'])}

tree.ellipseTreeExpansion(run_options)
final_path = tree.final_path()
tree.draw_sceneandtree(size=(15, 15))
tree.draw_path(final_path)
# hlfmtxpts drawing currently is slow
tree.drawEllipsoids(final_path, hlfmtxpts=True)
print(' Finished')
plt.show()
