"""
File created to test sampleEllipsoid from RRTSimulator in simulation.simulators
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from trees.rrt import RRT
from utils.shapes import Rectangle, Ellipse
from utils.metrics import l2norm2D, furutaDistanceMetric
from systems.primitives import Input
from systems.examples import Furuta
from simulation.simulators import RRTSimulator
from visuals.helper import pickRandomColor
from visuals.plotting import (Scene, drawScene, drawTree, drawPath)

start = [np.pi, 0]
goal = [0, np.pi]
region = Rectangle([-2*np.pi, -2*np.pi], 4*np.pi, 4*np.pi)
start_state = np.array(start + [0, 0]).reshape(4, 1)
goal_state = np.array(goal + [0, 0]).reshape(4, 1)
obstacles = []
scene = Scene(start, goal, region, obstacles)

sys_opts = {
    'dt': 0.005,
    'nx': 4,
    'nu': 2,
    'nw': 3,
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

dist_metric = furutaDistanceMetric

rrt_input = Input(dim=sys_opts['nu'], type_='random')
rrt_input.setLimits(np.array([[4, 0]]).T)
rrt_input.determinePossibleActions(range_=0.25, resolutions=np.array([10, 1]))
rrt_input.setNumSamples(3)
rrt_tree = RRT(start=start_state,
           goal=goal_state,
           system=sys,
           input_=rrt_input,
           scene=scene,
           dist_func=dist_metric)

options = {
    'min_dist':         1e-3,                           # :float:                       min dist to goal
    'max_iter':         10,                             # :int:                         iterations
    'direction':        'backward',                     # :'backward'/'forward':        determine tree growth direction
    'track_children':   True,                           # :bool:                        keep record of children of node
    'extend_by':        20,                             # :int:                         num timesteps to simulate in steer function with each extension
    'goal_sample_rate': 0.20,                           # :float:                       goal sample freq. (out of 1)
    'sample_dim':       2,                              # :int:                         Determine how many dimensions to sample in, e.g. 2 for 2D
    'D':                0.10*np.eye(sys_opts['nw']),    # :nparray: (nw x nw)           ellipse describing uncertainty
    'E0':               0.10*np.eye(sys_opts['nx']),    # :nparray: (nx x nx)           initial state uncertainty
    'max_dims':         np.array([5, 5]),               # :nparray: (2,)                maximum axis length of ellipse in each dimension
                                                        #                               currently only 2D supported
    'Q':                10*np.diag((1, 1, 0.5, 0.5)),   # :nparray: (nx x nx)           TVLQR Q
    'R':                np.eye(sys_opts['nu']),         # :nparray: (nu x nu)           TVLQR R
}

# run rrt
print('\nRRT Expanding...')
rrt_tree.treeExpansion(options)
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


sim1 = RRTSimulator(tree=rrt_tree,
                    opts=options)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(n):
    s = sim1.sampleEllipsoid()
    ax.scatter(s[0, 0], s[1, 0], s[2, 0])
plt.draw()

#num_simulations=1
#vis_rrt = True
#goal_epsilon = 1e-2
#print(f"Simulating RRT with{'' if vis_rrt else 'out'} visualization...")
#if vis_rrt: drawScene(scene, size=(15, 15))
#sim1.assessTree(num_simulations, goal_epsilon, vis_rrt)
#if vis_rrt:
#    plt.xlabel('Theta1 (Radians)', fontsize=20)
#    plt.ylabel('Theta2 (Radians)', fontsize=20)
#    plt.title('Note: Positions are modulo 2pi',fontsize=16)
#    plt.suptitle('Furuta RRT Simulation',fontsize=25, y=0.925)
#    plt.draw()
#    plt.pause(0.001)

plt.show()
