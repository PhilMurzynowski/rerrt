"""
Plotting tools
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visuals.helper import pickRandomColor

class Scene:
    """
    Object to facilitate plotting. After instatiation,
    drawScene always used first to create a figure and then tree, reachable states, ellipsoids drawn on top.
    """


    def __init__(self, start, goal, region, obstacles):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.region = region
        self.obstacles = obstacles

    def plotScene(self):
        for r in self.obstacles:
            drawRectangle(r)
        plt.axis([self.region.v1[0]-0.5, self.region.v4[0]+0.5, self.region.v1[1]-0.5, self.region.v4[1]+0.5])
        plt.plot(self.start[0], self.start[1], "xr", markersize=10)
        plt.plot(self.goal[0], self.goal[1], "xb", markersize=10)
        plt.legend(('start', 'goal'), loc='upper left')
        plt.gca().set_aspect('equal')


def drawScene(scene, size=(5, 5)):
    """Draw Scene object with desired figure size. Draws start, stop and obstacles in allowable region.
    Almost always used first to create a figure and then tree, reachable states, ellipsoids drawn on top.
    """
    plt.figure(figsize=size)
    scene.plotScene()

def drawRectangle(rectangle, color='cyan', fill=True):
    """Draw given Rectangle object with color and fill options.
    """
    ax = plt.gca()
    rect = patches.Rectangle(rectangle.v1, rectangle.w, rectangle.h, rectangle.angle, color='cyan')
    ax.add_artist(rect)

def drawEllipsoid(ellipse, color="gray", fill=False):
    """Draw given Ellipse object with color and fill options.
    """
    ax = plt.gca()
    ellip = patches.Ellipse(ellipse.c, ellipse.w, ellipse.h, ellipse.angle, color=color, fill=fill)
    ax.add_artist(ellip)

def plotNode(node, new_figure=False, color=None, reachable=False, lines=False, ellipse=False):
    """Plot a node.
    node            :RERRTNode:         node to plot
    new_figure      :bool:              create a new figure
    color           :string:, (r,g,b)
    reachable       :bool:              draw reachable states
    lines           :bool:              draw lines to reachable states
    ellipse         :bool:              draw ellipsoid
    """
    if new_figure:
        print('new figure')
        plt.figure()
    if reachable:
        if lines:
            for key, reach in node.reachable.items():
                # using plot to get lines
                plt.plot([node.x[0], reach[0]], [self.x[1], reach[1]], color=color)
        else:
            reach_xs = [reach[0] for key, reach in node.reachable.items()]
            reach_ys = [reach[1] for key, reach in node.reachable.items()]
        plt.scatter(reach_xs, reach_ys)
    plt.scatter(node.x[0], node.x[1], color=color)
    if ellipse: drawEllipsoid(node.ellipse, color=color)
    if new_figure:
        plt.show()

def drawPath(path, color='blue'):
    """Given a path, list,tuple,set, etc. of nodes draws connections between nodes.
    nodes       :RRTNode: list,set,etc.     nodes to draw
    color       :string:, (r, g, b)
    """
    for node in path:
        if node.parent:
            plt.plot([node.x[0], node.parent.x[0]], [node.x[1], node.parent.x[1]], color=color)

def drawTree(tree, color='blue'):
    """Draws full tree.
    tree        :RERRT:                     tree to draw
    color       :string:, :(r, g, b):
    """
    for node in tree.node_list:
        if node.parent is not None:
            plt.plot([node.x[0], node.parent.x[0]], [node.x[1], node.parent.x[1]], color=color)

def drawReachable(nodes, color='limegreen', fraction=1.00):
    """Given list/set,etc. of nodes, draws corresponding reachable states.
    nodes       :RERRTNode: list,set,etc.   nodes to draw
    color       :string:, (r, g, b)
    fraction    :float:                     out of 1, fraction to draw
    """
    freq = 1/fraction
    plotnum = 0
    for node in nodes:
        for key, reach in node.reachable.items():
            plotnum += 1
            if plotnum%freq==0:
                plt.plot([node.x[0], reach[0]], [node.x[1], reach[1]], color=color)

def drawEllipsoids(nodes, hlfmtxpts=False, color='gray', fraction=1.00):
    """Given list/set,etc. of nodes, draws corresponding ellipsoids.
    nodes       :RERRTNode: list,set,etc.   nodes to draw ellipsoids for
    hlfmtxpts   :bool:                      whether to draw hlfmtxpts
    color       :string:, :(r, g, b):       desired color
    fraction    :float:                     out of 1, what percentage to draw can
                                            become computationally expensive
    """
    freq = 1/fraction
    for i, n in enumerate(nodes):
        if i%freq==0:
            if n.ellipse is None:
                # if a goalstate was never propogated from
                # will not have an ellipse set
                continue
            # inefficient if has already been converted
            n.ellipse.convertFromMatrix()
            drawEllipsoid(n.ellipse, color=color)
            if hlfmtxpts:
                halfmtx_pts = n.ellipse.getHalfMtxPts()
                plt.scatter(halfmtx_pts[0, :], halfmtx_pts[1, :])

def drawEllipsoidTree(tree, opts):
    """Function to draw all ellipses in RERRT tree. Ellipses at each node
    currently only keep the last propogated ellipse, otherwise RERRT would be
    extremely memory intensive. So if the path branches only the last propogated
    value will be kept. This basic way to draw all ellipses with backwards RRT
    must first find all valid start nodes and for each start node reprogate from
    that start node and draw ellipses.
    Note: Currently only supports backward RERRT.
    tree    :RERRT:     tree that has been grown with tree.ellipseTreeExpansion
    opts    :dict:      options that tree was expanded with
    """
    if not opts['track_children']:
        raise RuntimeError('Enable track_children')
    if opts['direction'] == 'backward':
        startnodes = (n for n in tree.node_list if len(n.children)==0)
        for startnode in startnodes:
            valid_propagation = tree.repropagateEllipses(startnode, opts)
            assert valid_propagation, 'BUG'
            path = tree.getPath(startnode, reverse=False)
            drawEllipsoids(path, color=pickRandomColor())
    elif opts['direction'] == 'forward':
        raise NotImplementedError('Not implemented yet for forward RRT')
