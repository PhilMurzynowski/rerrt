"""
Simulating both Furuta RERRT and Furta RRT to compare robustness.
Code in examples provide more documentation to understand layout.
"""

import numpy as np
import matplotlib.pyplot as plt

from trees.rrt import RRT
from trees.rerrt import RERRT
from utils.shapes import Rectangle, Ellipse
from utils.metrics import l2norm2D, furutaDistanceMetric
from utils.collision import CollisionDetection
from systems.primitives import Input
from systems.examples import Furuta
from visuals.helper import pickRandomColor
from visuals.plotting import (Scene, drawScene, drawTree, drawPath,
                              drawReachable, drawEllipsoids, drawEllipsoidTree)

# general setup for both rrt and rerrt
start = [np.pi, 0]
goal = [0, np.pi]
region = Rectangle([-2*np.pi, -2*np.pi], 4*np.pi, 4*np.pi)
start_state = np.array(start + [0, 0]).reshape(4, 1)
goal_state = np.array(goal + [0, 0]).reshape(4, 1)
obstacles = []
scene = Scene(start, goal, region, obstacles)

# system setup
sys_opts = {
    'dt': 0.005,
    'nx': 4,
    'nu': 2,
    'nw': 2,
    'm1': 1.300,
    'm2': 0.075,
    'l1': 0.150,
    'l2': 0.148,
    'L1': 0.278,
    'L2': 0.300,
    'b1': 1e-4,
    'b2': 2.8e-4,
    'J1': 2.48e-2,
    'J2': 3.86e-3
    }
sys = Furuta(sys_opts)

# pick metric and collision detection
dist_metric = furutaDistanceMetric
col = CollisionDetection()
collision_function = col.selectCollisionChecker('erHalfMtxPts')

# rrt setup
rrt_input = Input(dim=sys_opts['nu'], type_='random')
rrt_input.setLimits(np.array([10, 0]))
rrt_input.determinePossibleActions(resolutions=np.array([10, 1]))
rrt_input.setNumSamples(3)
rrt_tree = RRT(start=start_state,
           goal=goal_state,
           system=sys,
           input_=rrt_input,
           scene=scene,
           dist_func=dist_metric)
rrt_options = {
    'min_dist':         1e-1,
    'max_iter':         50,
    'direction':        'backward',
    'track_children':   True,
    'extend_by':        20,
    'goal_sample_rate': 0.20,
    'sample_dim':       2,
}

# rerrt setup
rerrt_input = Input(dim=sys_opts['nu'], type_='deterministic')
rerrt_input.setLimits(np.array([10, 0]))
rerrt_input.determinePossibleActions(resolutions=np.array([3, 1]))
rerrt_tree = RERRT(start=start_state,
             goal=goal_state,
             system=sys,
             input_=rerrt_input,
             scene=scene,
             dist_func=dist_metric,
             collision_func=collision_function)
rerrt_options = {
    'min_dist':         1e-1,                             # :float:                       min dist to goal
    'max_iter':         50,                            # :int:                         iterations
    'direction':        'backward',                     # :'backward'/'forward':        determine tree growth direction
    'track_children':   True,                           # :bool:                        keep record of children of node
    'extend_by':        20,                             # :int:                         num timesteps to simulate in steer function with each extension
    'goal_sample_rate': 0.20,                           # :float:                       goal sample freq. (out of 1)
    'sample_dim':       2,                              # :int:                         Determine how many dimensions to sample in, e.g. 2 for 2D
    'D':                0.10*np.eye(sys_opts['nw']),    # :nparray: (nw x nw)           ellipse describing uncertainty
    'E0':               0.10*np.eye(sys_opts['nx']),    # :nparray: (nx x nx)           initial state uncertainty
    'Q':                np.diag((5, 5, 0, 0)),       # :nparray: (nx x nx)           TVLQR Q
    'R':                np.eye(sys_opts['nu']),         # :nparray: (nu x nu)           TVLQR R
}

# run rrt
print('\nRRT Expanding...')
rrt_tree.treeExpansion(rrt_options)
print('\nPlotting...')
rrt_final_path = rrt_tree.finalPath()
drawScene(scene, size=(15, 15))
plt.xlabel('Theta1 (Radians)', fontsize=20)
plt.ylabel('Theta2 (Radians)', fontsize=20)
plt.title('Note: Positions are modulo 2pi',fontsize=16)
plt.suptitle('Furuta RRT',fontsize=25, y=0.925)
drawTree(rrt_tree, color='blue')
drawPath(rrt_final_path, color='red')
print('Finished\n')
plt.draw()
plt.pause(0.001)    # hack to show plots realtime

# run rerrt
print('RERRT Expanding...')
rerrt_tree.treeExpansion(rerrt_options)
print('\nPlotting...')
rerrt_final_path = rerrt_tree.finalPath()
drawScene(scene, size=(15, 15))
plt.xlabel('Theta1 (Radians)', fontsize=20)
plt.ylabel('Theta2 (Radians)', fontsize=20)
plt.title('Note: Positions are modulo 2pi',fontsize=16)
plt.suptitle('Furuta RERRT',fontsize=25, y=0.925)
drawTree(rerrt_tree, color='blue')
drawPath(rerrt_final_path, color='red')
drawEllipsoidTree(rerrt_tree, rerrt_options)
print('Finished\n')
plt.draw()
plt.pause(0.001)

print('Comparing Robustness...')
# this only valid if both trees grown backwards
# can organize this into class methods
rrt_tree.start.setSi(np.zeros((sys.nx, sys.nx)))
rrt_tips = rrt_tree.getTipNodes()
for startnode in rrt_tips:
    # calc TVLQR
    # using same Q and R as rerrt
    # likely slow due to getPath
    path = rrt_tree.getPath(startnode, reverse=False)
    N = len(path)
    # x, u already provided from tree growth
    for i in range(N-1):
        path[i].getJacobians(sys)
    for i in range(N-1, 0, -1):
        path[i-1].calcSi(rerrt_options['Q'], rerrt_options['R'], path[i])
    for i in range(N-1):
        path[i].calcKi(rerrt_options['R'], path[i+1])

#rerrt_tips = rerrt_tree.getTipNodes()


print('Finished\n')
plt.show()
