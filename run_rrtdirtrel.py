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
rects.append(Rectangle([7, 11], 3, 1.5, angle=120.0))
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

# select input type
# when cleaning up, convert to class for easier setup
input_type = 'deterministic'
u0max, u1max = 1, 1 
if input_type == 'deterministic':
    input_actions = [np.array([[acc], [ang_vel]]) for acc in np.linspace(-u0max, u0max, 2) for ang_vel in np.linspace(-u1max, u1max, 3)]
    numinput_samples = len(input_actions)
elif input_type == 'random':
    numinput_samples = 15

col = CollisionDetection()
collision_function = col.selectCollisionChecker('erHalfMtxPts')

# initialize RRT_Dirtrel
tree = RRT_Dirtrel(start_state, goal_states, sys, scene, collision_function)

# run RRT_Dirtrel
run_options = {
    'epsilon':          1,                              # :float:                       min dist to goal
    'max_iter':         800,                            # :int:                         iterations
    'plot_freq':        None,                           # :int:                         plot tree expansion freq. (num iterations), Update
    'plot_size':        (10, 10),                       # :(int, int):                  plot size
    'direction':        'backward',                     # :'backward'/'forward':        determine tree growth direction
    'goal_sample_rate': 0.15,                           # :float:                       goal sample freq. (out of 1)
    'input_type':       input_type,                     # :'random'/'deterministic':    control sampling method 
    'input_max':        (u0max, u1max),                 # :(float,): (dim(input) x 1)   if input type random, max magnitude of each input
    'numinput_samples': numinput_samples,               # :int:                         if input_type random, num random samples, otherwise num actions
    'input_actions':    input_actions,                  # :list(inputs):                if input_type deterministic, possible inputs
    'extend_by':        1,                              # :int:                         num timesteps to simulate in steer function with each extension
    'nx':               sys_opts['nx'],                 # :int:                         dim of state
    'nu':               sys_opts['nu'],                 # :int:                         dim of input
    'nw':               sys_opts['nw'],                 # :int:                         dim of uncertainty
    'D':                0.05*np.eye(sys_opts['nw']),    # :nparray: (nw x nw)           ellipse describing uncertainty
    'E0':               0.05*np.eye(sys_opts['nx']),    # :nparray: (nx x nx)           initial state uncertainty
    'Ql':               np.eye(sys_opts['nx']),         # :nparray: (nx x nx)           use if robust cost from DIRTREL paper added
    'Rl':               np.eye(sys_opts['nu']),         # :nparray: (nu x nu)           see above
    'QlN':              np.eye(sys_opts['nx']),         # :nparray: (nx x nx)           see above
    'Q':                np.diag((5, 5, 0, 0, 0)),       # :nparray: (nx x nx)           TVLQR Q
    'R':                np.eye(sys_opts['nu'])}         # :nparray: (nu x nu)           TVLQR R

tree.ellipseTreeExpansion(run_options)
final_path = tree.final_path()
tree.draw_sceneandtree(size=(15, 15))
tree.draw_path(final_path)
# hlfmtxpts drawing currently is slow
tree.drawEllipsoids(final_path, hlfmtxpts=True)
print(' Finished')
plt.show()
