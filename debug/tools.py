"""
File under constant changes, currently not adding doc
"""


from visuals.plotting import plotNode, drawEllipsoids
import matplotlib.pyplot as plt

#print(f'cond: {np.linalg.cond(abk)}')
#e = np.linalg.eigvals(abk)
#print(f'e_max: {max(e)} e_min: {min(e)}')


def traceChildren(node, genleft, color, plotted):
    #node.ellipse.convertFromMatrix()
    plotNode(node, new_figure=False, color=color)
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
    drawEllipsoids(path)
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
