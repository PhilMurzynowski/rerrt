"""
RRT-Dirtrel Implementation
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
start = [10, 10]
start_state = np.array(start + [-np.pi*2/3, 5, 0]).reshape(5, 1)
goal = np.array([[0], [0]])
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
tree = RRT_Dirtrel(start_state, scene.goal, sys, scene, collision_function)

# run RRT_Dirtrel
run_options = {
    'epsilon':          1,                              # min dist to goal
    'max_iter':         100,                            # iterations
    'plot_freq':        100,                            # how often to plot tree expansion (num iterations)
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

