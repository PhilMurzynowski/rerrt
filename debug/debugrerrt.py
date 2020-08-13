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
from trees.rrt import RRT
from trees.rerrt import RERRT
from utils.shapes import Rectangle, Ellipse
from utils.collision import CollisionDetection
from utils.systems import System, Car
from utils.visual import Scene
from utils.general import pickRandomColor


# Initialize start, goal, bounds on area
start = [12.5, 12.5]
goal = [0, 0]
region = Rectangle([-5, -5], 20, 20)

# start_state used for forward expansion as start
# goal_states used for backward expansion as start
#   multiple goals states allow for better tree growth using backwards RRT with nonlinear systems
# currently RRT does not require both starting state
# and ending state to be satisfied
# one must be satisfied and returns solution closest to other desired state
start_state = np.array(start + [-np.pi*2/3, 5, 0]).reshape(5, 1)
num_goal_states = 10
eps = 1e-4
goal_speed = 1
goal_states = [np.array([goal[0]+eps*np.cos(theta)]+[goal[1]+eps*np.sin(theta)]+[(theta+np.pi)%(2*np.pi), goal_speed, 0]).reshape(5, 1) for theta in np.linspace(-np.pi, np.pi, num_goal_states, endpoint=False)]


# initialize obstacles
obstacles = []
# arguments: rectangle bottom left corner, width, height, angle from horizontal (deg)
obstacles.append(Rectangle([7, 11], 3, 1.5, angle=120.0))
obstacles.append(Rectangle([7, 4], 2.5, 1.5, angle=30.0))

# Scene describes the start, goal, obstacles, mostly for plotting
scene = Scene(start, goal, region, obstacles)

sys_opts = {
    'dt': 0.1,
    'nx': 5,
    'nu': 2,
    'nw': 2
    }
sys = Car(sys_opts)

# select input type
# when cleaning up, convert to class for easier setup
input_type = 'deterministic'
u0max, u1max = 10, 10
if input_type == 'deterministic':
    input_actions = [np.array([[acc], [ang_vel]]) for acc in np.linspace(-u0max, u0max, 2) for ang_vel in np.linspace(-u1max, u1max, 3)]
    numinput_samples = len(input_actions)
elif input_type == 'random':
    numinput_samples = 15

col = CollisionDetection()
collision_function = col.selectCollisionChecker('erHalfMtxPts')

# initialize RRT_Dirtrel
tree = RERRT(start_state, goal_states, sys, scene, collision_function)

# run RRT_Dirtrel
run_options = {
    'epsilon':          1,                              # :float:                       min dist to goal
    'max_iter':         100,                            # :int:                         iterations
    'plot_freq':        None,                           # :int:                         plot tree expansion freq. (num iterations), Update
    'plot_size':        (10, 10),                       # :(int, int):                  plot size
    'direction':        'backward',                     # :'backward'/'forward':        determine tree growth direction
    'track_children':   True,                           # :bool:                        keep record of children of node
    'goal_sample_rate': 0.20,                           # :float:                       goal sample freq. (out of 1)
    'input_type':       input_type,                     # :'random'/'deterministic':    control sampling method 
    'input_max':        (u0max, u1max),                 # :(float,): (dim(input) x 1)   if input type random, max magnitude of each input
    'numinput_samples': numinput_samples,               # :int:                         if input_type random, num random samples, otherwise num actions
    'input_actions':    input_actions,                  # :list(inputs):                if input_type deterministic, possible inputs
    'extend_by':        1,                             # :int:                         num timesteps to simulate in steer function with each extension
    'nx':               sys_opts['nx'],                 # :int:                         dim of state
    'nu':               sys_opts['nu'],                 # :int:                         dim of input
    'nw':               sys_opts['nw'],                 # :int:                         dim of uncertainty
    'D':                0.01*np.eye(sys_opts['nw']),    # :nparray: (nw x nw)           ellipse describing uncertainty
    'E0':               0.01*np.eye(sys_opts['nx']),    # :nparray: (nx x nx)           initial state uncertainty
    'Ql':               np.eye(sys_opts['nx']),         # :nparray: (nx x nx)           use if robust cost from DIRTREL paper added
    'Rl':               np.eye(sys_opts['nu']),         # :nparray: (nu x nu)           see above
    'QlN':              np.eye(sys_opts['nx']),         # :nparray: (nx x nx)           see above
    'Q':                np.diag((5, 5, 0, 0, 0)),       # :nparray: (nx x nx)           TVLQR Q
    'R':                np.eye(sys_opts['nu'])}         # :nparray: (nu x nu)           TVLQR R

tree.ellipseTreeExpansion(run_options)
print('\nPlotting...')
final_path = tree.final_path()
# order determines what gets occluded in figure
# fractional plotting is used as dataset becomes huge
tree.draw_scene(size=(15, 15))
#tree.drawReachable(tree.node_list, fraction=1.00)
tree.draw_tree(color='blue')
#tree.draw_path(final_path, color='red')
# hlfmtxpts drawing currently is slow
#tree.drawEllipsoids(final_path, hlfmtxpts=False, fraction=1.00)
tree.drawEllipsoidTree(run_options)

# cleanup
# ellipse debugging

def traceChildren(node, genleft, color, plotted):
    #node.ellipse.convertFromMatrix()
    node.plotNode(new_figure=False, color=color)
    plotted.add(node)
    print(f'gen: {genleft}')
    print(f'x, h, w:\n {node.x, node.ellipse.h, node.ellipse.w}')
    #print(f'node.ellipse.mtx:\n {node.ellipse.mtx}')
    print(f'node.E:\n {node.E}')
    if genleft > 0:
        new_color = (color[0]+0.03, color[1]+0.03, color[2]+0.03)
        print(f'{len(node.children)} children: \n {[child.x for child in node.children]}')
        # follow path of larger child
        largest_area = 0
        largest_child = None
        for child in node.children:
            area = child.ellipse.getArea()
            if area > largest_area:
                largest_area = area
                largest_child = child
        if largest_child is not None:
            print(f'Control input of child: {largest_child.u}')
            print(f'Child G: {largest_child.G}')
            traceChildren(largest_child, genleft-1, new_color, plotted)

def findNodeLargestEllipse(tree, opts):
    # finds the startnode that ends up having the largest ellipses
    largest_area = 0
    largestnode = None
    corresponding_startnode = None
    if not opts['track_children']:
        raise RuntimeError('Enable track_children')
    if opts['direction'] == 'backward':
        startnodes = (n for n in tree.node_list if len(n.children)==0)
        for startnode in startnodes:
            valid_propagation = tree.repropagateEllipses(startnode, opts)
            assert valid_propagation, 'BUG'
            path = tree.getPath(startnode, reverse=False)
            for node in path:
                if node.ellipse is None: continue
                node.ellipse.convertFromMatrix()
                area = node.ellipse.getArea()
                if area > largest_area:
                    largest_area = area
                    largestnode = node
                    corresponding_startnode = startnode
    elif opts['direction'] == 'forward':
        raise NotImplementedError('Not implemented yet for forward RRT')
    return corresponding_startnode, largestnode

def debugLargestEllipse(tree, opts):
    generations = 3
    startnode, largestnode = findNodeLargestEllipse(tree, opts)
    assert tree.repropagateEllipses(startnode, opts)
    path = tree.getPath(startnode, reverse=False)
    tree.drawEllipsoids(path)
    #snippet = path[:min(generations, len(path))]
    snippet = path[:min(generations, path.index(largestnode)+1)]
    for node in snippet:
        if node == largestnode:
            plt.scatter(node.x[0], node.x[1], color='red')
        print(f'x, h, w:\n {node.x, node.ellipse.h, node.ellipse.w}')
        print(f'node.E:\n {node.E}')
        print(f'A:\n {node.A}')
        print(f'B:\n {node.B}')
        print(f'K:\n {node.K}')
        print(f'BK:\n {node.B@node.K}')
        print(f'A-BK: {node.A-node.B@node.K}')
        #print(f'u:\n {node.u}')

tree.draw_scene(size=(15, 15))
tree.draw_tree(color='blue')
debugLargestEllipse(tree, run_options)
print('Finished')
plt.show()





