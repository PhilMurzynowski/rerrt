"""
Plotting tools
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visuals.helper import pickRandomColor

class Scene:


    def __init__(self, start, goal, region, obstacles):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.region = region
        self.obstacles = obstacles

    def plotScene(self):
        ax = plt.gca()
        for r in self.obstacles:
            rect = patches.Rectangle(r.v1, r.w, r.h, r.angle, color='cyan')
            ax.add_artist(rect)
        plt.axis([self.region.v1[0]-0.5, self.region.v4[0]+0.5, self.region.v1[1]-0.5, self.region.v4[1]+0.5])
        plt.plot(self.start[0], self.start[1], "xr", markersize=10)
        plt.plot(self.goal[0], self.goal[1], "xb", markersize=10)
        plt.legend(('start', 'goal'), loc='upper left')
        plt.gca().set_aspect('equal')


def drawScene(scene, size=(5, 5)):
    plt.figure(figsize=size)
    scene.plotScene()

def plotNode(node, new_figure=False, color=None):
    # debug tool
    if new_figure:
        print('new figure')
        plt.figure()
    #for key, reach in node.reachable.items():
    #    # using plot to get lines
    #    plt.plot([node.x[0], reach[0]], [self.x[1], reach[1]], color=color)
    #reach_xs = [reach[0] for key, reach in node.reachable.items()]
    #reach_ys = [reach[1] for key, reach in node.reachable.items()]
    #plt.scatter(reach_xs, reach_ys)
    plt.scatter(node.x[0], node.x[1], color=color)
    node.ellipse.drawEllipse(color=color)
    if new_figure:
        plt.show()

def drawPath(path, color='blue'):
    for node in path:
        if node.parent:
            plt.plot([node.x[0], node.parent.x[0]], [node.x[1], node.parent.x[1]], color=color)

def drawTree(tree, color='blue'):
    for node in tree.node_list:
        if node.parent:
            plt.plot([node.x[0], node.parent.x[0]], [node.x[1], node.parent.x[1]], color=color)

def drawReachable(nodes, color='limegreen', fraction=1.00):
    freq = 1/fraction
    plotnum = 0
    for node in nodes:
        for key, reach in node.reachable.items():
            plotnum += 1
            if plotnum%freq==0:
                plt.plot([node.x[0], reach[0]], [node.x[1], reach[1]], color=color)

def drawEllipsoids(nodes, hlfmtxpts=False, color='gray', fraction=1.00):
    freq = 1/fraction
    for i, n in enumerate(nodes):
        if i%freq==0:
            if n.ellipse is None:
                # if a goalstate was never propogated from will not have an ellipse set
                continue
            n.ellipse.convertFromMatrix()
            n.ellipse.drawEllipse(color=color)
            if hlfmtxpts:
                halfmtx_pts = n.ellipse.getHalfMtxPts()
                plt.scatter(halfmtx_pts[0, :], halfmtx_pts[1, :])

def drawEllipsoidTree(tree, opts):
    # ellipses at each node currently only keep the last propogated ellipse
    # otherwise would be extremely memory intensive
    # so if the path branches only the last propogated value will be kept
    # basic way to draw all ellipses with backwards RRT must:
    # find all valid start nodes and for each start node:
    # reprogate from that start node and draw ellipses
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
