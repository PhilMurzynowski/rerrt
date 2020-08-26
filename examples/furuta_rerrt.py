"""
Furuta RERRT Example
"""

# python libraries
import numpy as np
import matplotlib.pyplot as plt

# custom classes
from trees.rrt import RRT
from trees.rerrt import RERRT
from utils.shapes import Rectangle, Ellipse
from utils.metrics import l2norm2D, l2normND, furutaDistanceMetric
from utils.collision import CollisionDetection
from systems.primitives import Input
from systems.examples import Furuta
from visuals.helper import pickRandomColor
from visuals.plotting import (Scene, drawScene, drawTree, drawPath,
                              drawReachable, drawEllipsoids, drawEllipsoidTree)


# Initialize start, goal, bounds on area
start = [np.pi, 0]
goal = [0, np.pi]
region = Rectangle([-2*np.pi, -2*np.pi], 6*np.pi, 6*np.pi)

# start_state used for forward expansion as start
# goal_state used for backward expansion as start
# currently RRT does not require both starting state and ending state to belong to final tree
# one must be the root of tree and returns solution closest to other desired state
start_state = np.array(start + [0, 0]).reshape(4, 1)
goal_state = np.array(goal + [0, 0]).reshape(4, 1)

# initialize obstacles
# no obstacles currently
obstacles = []

# Scene describes the start, goal, obstacles, mostly for plotting
scene = Scene(start, goal, region, obstacles)

# Furuta pendulum system
# Derivation and parameters provided by:
#     https://www.hindawi.com/journals/jcse/2011/528341/
# assigning values to physical parameters
# look more into 5. Simplifications to ensure parameters chose below
# allow for the applied simplications, also 2.2 Assumptions
sys_opts = {
    'dt': 0.005,
    'nx': 4,
    'nu': 2,
    'nw': 2,
    'm1':  0.300,
    'm2':  0.075,
    'l1':  0.150,
    'l2':  0.148,
    'L1':  0.278,
    'L2':  0.300,
    'b1':  1e-4,
    'b2':  2.8e-4,
    'J1':  2.48e-2,
    'J2':  3.86e-3
    }
sys = Furuta(sys_opts)

#   'dt'   :float:  timestep
#   'nx'   :int:    dim of state
#   'nu'   :int:    dim of input
#   'nw'   :int:    dim of uncertainty
#   rest of parameters furuta specific, see systems/examples.py
#   where the Furuta subclass is defined

# named input_ to avoid conflict with python keyword input
input_ = Input(dim=sys_opts['nu'], type_='deterministic')
# no motor at second joint, making underactuated
input_.setLimits(np.array([10, 0]))
input_.determinePossibleActions(resolutions=np.array([3, 1]))

#   setLimits          args     :nparray: (dim(input),)         max magnitude of each input
#   setType            args     :'random'/'deterministic':      input sampling type, often abbrv. input type 
#   setNumSamples      args     :int:                           if input type random, num random samples
#   determineActions   args     :nparray: (dim(input),)         if input type deterministic, calculates deterministic actions  


# pick desired distance metric
# custom distance metric described for furuta pendulum
#dist_metric = l2norm2D
dist_metric = furutaDistanceMetric

col = CollisionDetection()
collision_function = col.selectCollisionChecker('erHalfMtxPts')

# initialize RERRT
tree = RERRT(start=start_state,
             goal=goal_state,
             system=sys,
             input_=input_,
             scene=scene,
             dist_func=dist_metric,
             collision_func=collision_function)

# options to configure RERRT initialization and expansion
run_options = {
    'min_dist':         1e-1,                             # :float:                       min dist to goal
    'max_iter':         400,                            # :int:                         iterations
    'direction':        'backward',                     # :'backward'/'forward':        determine tree growth direction
    'track_children':   True,                           # :bool:                        keep record of children of node
    'extend_by':        20,                             # :int:                         num timesteps to simulate in steer function with each extension
    'goal_sample_rate': 0.20,                           # :float:                       goal sample freq. (out of 1)
    'sample_dim':       2,                              # :int:                         Determine how many dimensions to sample in, e.g. 2 for 2D
    'D':                0.10*np.eye(sys_opts['nw']),    # :nparray: (nw x nw)           ellipse describing uncertainty
    'E0':               0.10*np.eye(sys_opts['nx']),    # :nparray: (nx x nx)           initial state uncertainty
    'max_dims':         np.array([5, 5]),               # :nparray: (2,)                maximum axis length of ellipse in each dimension
                                                        #                               currently only 2D supported
    'Q':                np.diag((10, 10, 5, 5)),        # :nparray: (nx x nx)           TVLQR Q
    'R':                np.eye(sys_opts['nu']),         # :nparray: (nu x nu)           TVLQR R
}

print('\nTree Expanding...')
tree.treeExpansion(run_options)
print('\nPlotting...')
final_path = tree.finalPath()
# order determines what gets occluded in figure
drawScene(scene, size=(15, 15))
plt.xlabel('Theta1 (Radians)', fontsize=20)
plt.ylabel('Theta2 (Radians)', fontsize=20)
plt.title('Note: Positions are modulo 2pi',fontsize=16)        # hack to get subtitle
plt.suptitle('Furuta RERRT',fontsize=25, y=0.925)              # position with y for no overlap
#plt.title('Furuta RERRT', fontsize=25)
#plt.text(0.9, 0.9, 'hello', transform=ax.transAxes)
# drawScene is called first as it creates the figure then shows region+obstacles
#drawReachable(tree.node_list, fraction=1.00)
# fractional plotting is used as dataset becomes huge
drawTree(tree, color='blue')
drawPath(final_path, color='red')
# hlfmtxpts drawing currently is not optimized
#drawEllipsoids(final_path, hlfmtxpts=False, fraction=1.00)
drawEllipsoidTree(tree, run_options)
print('Finished\n')
plt.show()


