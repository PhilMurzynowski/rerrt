"""
Car RRT Example
"""

# python libraries
import numpy as np
import matplotlib.pyplot as plt

# custom classes
from trees.rrt import RRT
from utils.shapes import Rectangle, Ellipse
from utils.metrics import l2norm2D
from systems.primitives import Input
from systems.examples import Car
from visuals.helper import pickRandomColor
from visuals.plotting import (Scene, drawScene, drawTree, drawPath)

# Initialize start, goal, bounds on area
start = [12.5, 12.5]
goal = [0, 0]
region = Rectangle([-5, -5], 20, 20)

# start_state used for forward expansion as start
# goal_state used for backward expansion as start
# currently RRT does not require both starting state and ending state to belong to final tree
# one must be the root of tree and returns solution closest to other desired state
start_state = np.array(start + [-np.pi*2/3, 5, 0]).reshape(5, 1)
goal_state = np.array(goal + [-np.pi*2/3, 5, 0])

# initialize obstacles
obstacles = []
#   Rectangle          args     bottomLeftCornerXY, width, height, angle (degrees rotated about bottomLeftCornerXY)
obstacles.append(Rectangle([7, 11], 3, 1.5, angle=120.0))
obstacles.append(Rectangle([7, 4], 2.5, 1.5, angle=30.0))

# Scene describes the start, goal, obstacles, mostly for plotting
scene = Scene(start, goal, region, obstacles)

# could automate system config
sys_opts = {
    'dt': 0.02,
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
# using low resolution deterministic actions for Car dynamics
# tends to result in spiraling paths
# using random and sampling a small portion also descreases chances that
# will attempt to add exact same trajectory as already exists
#   can really see the usefulness of RGRRT trying to tune this
input_ = Input(dim=sys_opts['nu'], type_='random')
input_.setLimits(np.array(np.array([[10, 10]]).T))
input_.determinePossibleActions(range_=0.5, resolutions=np.array([5, 5]))
input_.setNumSamples(10)

#   setLimits          args     :nparray: (dim(input),)         max magnitude of each input
#   setType            args     :'random'/'deterministic':      input sampling type, often abbrv. input type 
#   setNumSamples      args     :int:                           if input type random, num random samples
#   determineActions   args     :nparray: (dim(input),)         if input type deterministic, calculates deterministic actions  

# pick desired distance metric
# some examples provided from where l2norm2D is imported
dist_metric = l2norm2D

# initialize RERRT
tree = RRT(start=start_state,
             goal=goal_state,
             system=sys,
             input_=input_,
             scene=scene,
             dist_func=dist_metric)

# options to configure RRT initialization and expansion
run_options = {
    'min_dist':          1,                             # :float:                       min dist to goal
    'max_iter':         1e3,                            # :int:                         iterations
    'direction':        'backward',                     # :'backward'/'forward':        determine tree growth direction
    'track_children':   True,                           # :bool:                        keep record of children of node
    'extend_by':        10,                             # :int:                         num timesteps to simulate in steer function with each extension
    'goal_sample_rate': 0.20,                           # :float:                       goal sample freq. (out of 1)
    'sample_dim':       2,                              # :int:                         Determine how many dimensions to sample in, e.g. 2 for 2D
}

print('\nTree Expanding...')
tree.treeExpansion(run_options)
print('\nPlotting...')
final_path = tree.finalPath()
#for n in final_path:
#    print(f'num children: {len(n.children)}')
#    print([(child.x, child.u) for child in n.children])
#print(f'max children: {max((len(n.children) for n in final_path))}')
# order determines what gets occluded in figure
drawScene(scene, size=(15, 15))
plt.xlabel('X (position)', fontsize=20)
plt.ylabel('Y (position)', fontsize=20)
plt.title('Car RRT', fontsize=25)
# drawScene is called first as it creates the figure then shows region+obstacles
drawTree(tree, color='blue')
drawPath(final_path, color='red')
print('Finished\n')
plt.show()


