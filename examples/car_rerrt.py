"""
Car RERRT Example
"""

# python libraries
import numpy as np
import matplotlib.pyplot as plt

# custom classes
from trees.rrt import RRT
from trees.rerrt import RERRT
from utils.shapes import Rectangle, Ellipse
from utils.metrics import l2norm2D
from utils.collision import CollisionDetection
from systems.primitives import Input
from systems.examples import Car
from visuals.helper import pickRandomColor
from visuals.plotting import (Scene, drawScene, drawTree, drawReachable,
                              drawEllipsoids, drawEllipsoidTree)


# Initialize start, goal, bounds on area
start = [12.5, 12.5]
goal = [0, 0]
region = Rectangle([-5, -5], 20, 20)

# start_state used for forward expansion as start
# goal_state used for backward expansion as start
# currently RRT does not require both starting state and ending state to belong to final tree
# one must be the root of tree and returns solution closest to other desired state
start_state = np.array(start + [-np.pi*2/3, 5, 0]).reshape(5, 1)
goal_state = np.array(goal + [-np.pi*2/3, 5, 0]).reshape(5, 1)

# initialize obstacles
obstacles = []
#   Rectangle
#   args     bottomLeftCornerXY, width, height, angle (degrees rotated about bottomLeftCornerXY)
obstacles.append(Rectangle([7, 11], 3, 1.5, angle=120.0))
obstacles.append(Rectangle([7, 4], 2.5, 1.5, angle=30.0))

# Scene describes the start, goal, obstacles, mostly for plotting
scene = Scene(start, goal, region, obstacles)

# could automate system config
sys_opts = {
    'dt': 0.005,
    'nx': 5,
    'nu': 2,
    'nw': 2
    }
sys = Car(sys_opts)

#   'dt'   :float:  timestep
#   'nx'   :int:    dim of state
#   'nu'   :int:    dim of input
#   'nw'   :int:    dim of uncertainty

# named input_ to avoid conflict with python keyword input
input_ = Input(dim=sys_opts['nu'], type_='deterministic')
input_.setLimits(np.array([10, 10]))
input_.determinePossibleActions(resolutions=np.array([2, 3]))

#   setLimits          args     :nparray: (dim(input),)         max magnitude of each input
#   setType            args     :'random'/'deterministic':      input sampling type, often abbrv. input type 
#   setNumSamples      args     :int:                           if input type random, num random samples
#   determineActions   args     :nparray: (dim(input),)         if input type deterministic, calculates deterministic actions  

# pick desired distance metric
# some examples provided from where l2norm2D is imported
dist_metric = l2norm2D

# pick collision detection method
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
    'max_iter':         50,                            # :int:                         iterations
    'direction':        'backward',                     # :'backward'/'forward':        determine tree growth direction
    'track_children':   True,                           # :bool:                        keep record of children of node
    'extend_by':        20,                             # :int:                         num timesteps to simulate in steer function with each extension
    'goal_sample_rate': 0.20,                           # :float:                       goal sample freq. (out of 1)
    'sample_dim':       2,                              # :int:                         Determine how many dimensions to sample in, e.g. 2 for 2D
    'D':                0.10*np.eye(sys_opts['nw']),    # :nparray: (nw x nw)           ellipse describing uncertainty
    'E0':               0.10*np.eye(sys_opts['nx']),    # :nparray: (nx x nx)           initial state uncertainty
    'Q':                np.diag((5, 5, 0, 0, 0)),       # :nparray: (nx x nx)           TVLQR Q
    'R':                np.eye(sys_opts['nu']),         # :nparray: (nu x nu)           TVLQR R
    'Ql':               np.eye(sys_opts['nx']),         # :nparray: (nx x nx)           use if robust cost from DIRTREL paper added
    'Rl':               np.eye(sys_opts['nu']),         # :nparray: (nu x nu)           see above
    'QlN':              np.eye(sys_opts['nx']),         # :nparray: (nx x nx)           see above
}

print('\nTree Expanding...')
tree.treeExpansion(run_options)
print('\nPlotting...')
final_path = tree.finalPath()
# order determines what gets occluded in figure
drawScene(scene, size=(15, 15))
plt.xlabel('X (position)', fontsize=20)
plt.ylabel('Y (position)', fontsize=20)
plt.title('Car RERRT', fontsize=25)
# drawScene is called first as it creates the figure then shows region+obstacles
#drawReachable(tree.node_list, fraction=1.00)
# fractional plotting is used as dataset becomes huge
drawTree(tree, color='blue')
#drawPath(final_path, color='red')
# hlfmtxpts drawing currently is not optimized
#drawEllipsoids(final_path, hlfmtxpts=False, fraction=1.00)
drawEllipsoidTree(tree, run_options)
print('Finished\n')
plt.show()


