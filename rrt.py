"""
Base RRT Implementation
"""

import numpy as np
import matplotlib.pyplot as plt



class RRT:


    class Node:

        def __init__(self, x, parent=None):
            self.x = x.reshape(x.size, 1)
            self.parent = parent
            if self.parent is not None:
                self.n = self.parent.n+1    # time sample
            else:
                self.n = 0


    def __init__(self, start, goal, system, scene):
        self.start = start
        self.goal = goal
        self.system = system
        self.scene = scene
        self.region = scene.region
        self.obstacles = scene.obstacles
        self.poly = [self.region] + self.obstacles


    def inObstacle(self, point):
        """Method to check if point within obstacle
           Currently not optimized"""
        for o in self.obstacles:
            if o.inPoly(point):
                return True
        return False


    def inRegion(self, point):
        return self.region.inPoly(point)


    def sample(self, options):
        """Sample region outside of obstacles or sample goal point"""
        if np.random.rand() > options['goal_sample_rate']:
            # sample region outside of obstacles
            rnd = None
            while rnd is None or self.inObstacle(rnd):
                rnd = np.random.uniform(self.region.v1, self.region.v4, (2, 1))
        else:
            # sample goal
            rnd = self.goal[:2, :]
        return rnd


    def distance_metric(self, p1, p2):
        """Can implement different distance metrics for nonlinear systems"""
        if p1.shape != (2, 1):
            p1_xy = p1[:2].reshape(2, 1)
        else:
            p1_xy = p1
        if p2.shape != (2, 1):
            p2_xy = p2[:2].reshape(2, 1)
        else:
            p2_xy = p2
        return np.linalg.norm(p1_xy - p2_xy)


    def dist_to_goal(self, pt):
        """Distance from p to goal"""
        return self.distance_metric(pt, self.goal)


    def nearest_node(self, new_location, get_dist=False):
        """Find node in tree nearest to new_node
           Can swap in more efficient implementation with kd Tree perhaps
           For nonlinear dynamics euclidean distance doesn't really make sense
           looking into Kinodynamic planner, OMPL, control-based planners or http://www.mit.edu/~dahleh/pubs/2.Real-Time%20Motion%20Planning%20for%20Agile%20Autonomous%20Vehicles.pdf"""
        # if system does not move on first timestep, will encounter error
        # will keep returning start node, as first in list
        L = len(self.node_list)
        closest_distance = np.Inf
        closest_node = None
        for i in range(L):
            distance = self.distance_metric(new_location, self.node_list[L-1-i].x[:2])
            if distance < closest_distance:
                closest_distance = distance
                closest_node = self.node_list[L-1-i]
        if get_dist:
            return closest_node, closest_distance
        return closest_node

    def pickAction(self, sel, opts):
        # sel, select only useful for deterministic action types
        if opts['input_type'] == 'random':
            u = np.zeros(opts['nu'], 1)
            for i in range(opts['nu']):
                u[i] = np.random.uniform(-opts['input_max'][i], opts['input_max'][i])
        elif opts['input_type'] == 'deterministic':
            u = opts['input_actions'][sel]
        return u

    def steer(self, from_node, to_location, opts):
        """samples multiple controls and checks for best extension
           cheap in low input dimension
           does not return covariance matrix as they are precomputed"""
        best_u = None
        best_proximity = np.inf
        for k in range(opts['numinput_samples']):
            u = self.pickAction(k, opts)
            if opts['direction'] == 'forward':
                x_samp = self.system.nextState(from_node.x, u)
            elif opts['direction'] == 'backward':
                x_samp = self.system.prevState(from_node.x, u)
            proximity = self.distance_metric(to_location, x_samp[0:2])
            if best_u is None or proximity < best_proximity:
                best_u, best_proximitiy, best_x = u, proximity, x_samp
        return best_x, best_u

    def steerMultiTimeStep(self, from_node, to_location, opts):
        """samples multiple controls and checks for best extension
           cheap in low input dimension
           does not return covariance matrix as they are precomputed"""
        best_u = None
        best_proximity = np.inf
        for k in range(opts['numinput_samples']):
            u = self.pickAction(k, opts)
            # extend by desired number of timesteps
            extension = [from_node.x]
            for i in range(opts['extend_by']):
                if opts['direction'] == 'forward':
                    extension.append(self.system.nextState(extension[-1], u))
                elif opts['direction'] == 'backward':
                    extension.append(self.system.prevState(extension[-1], u))
            proximity = self.distance_metric(to_location, extension[-1][0:2])
            if best_u is None or proximity < best_proximity:
                best_u, best_proximitiy, best_extension = u, proximity, extension
        # cut out from_node.x
        return best_extension[1:], best_u

    #def near(self, location, mu=3):
    #    """Eqn 45, may want to read paper for probalistic
    #        optimality guarantees, how to pick mu, gamma, etc
    #        also could use kd trees for efficiency?"""
    #    # implement eqn 46
    #    r_n = min(np.inf, mu)
    #    dlist = [self.distance_metric(new_location, n.x[0:2]) for n in self.node_list]
    #    near_indices = [dlist.index(x) for x in dlist if x <= r_n]
    #    return self.node_list[near_indices]


    def tree_expansion(self, options):
        iter_step = 0
        self.node_list = [Node(self.start)]
        best_dist_to_goal = self.dist_to_goal(self.start.x[0:2])

        while best_dist_to_goal > options['epsilon'] and iter_step <= options['max_iter']:
            samp = self.sample(options)
            closest = self.nearest_node(samp)
            x_hat, _ = self.steer(closest, samp, options)
            n_min = self.Node(x_hat, closest)

            if not self.inObstacle(x_hat[0:2]):
                # n_min only added to tree if not in obstacle
                self.node_list.append(n_min)
                dist = self.dist_to_goal(n_min.x[0:2])
                if dist < best_dist_to_goal:
                    best_dist_to_goal = dist
            if iter_step%options['plot_freq']==0 or best_dist_to_goal <= options['epsilon']:
                self.draw_sceneandtree()
                plt.title(f'Iteration: {iter_step}\nDistance to goal: {best_dist_to_goal}')
                plt.show()
            iter_step+=1

        #assert best_dist_to_goal <= epsilon


    def getPath(self, endnode, reverse=True):
        path = [endnode]
        node = endnode
        while node.parent is not None:
            node = node.parent
            path.append(node)
        if reverse: path = path[::-1]
        return path


    def final_path(self):
        # modify if run for multiple paths
        final = self.nearest_node(self.goal)
        return self.getPath(final)


    def draw_path(self, path):
        for node in path:
            if node.parent:
                plt.plot([node.x[0], node.parent.x[0]], [node.x[1], node.parent.x[1]], "-r")


    def draw_tree(self, color='blue'):
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x[0], node.parent.x[0]], [node.x[1], node.parent.x[1]], color=color)


    def draw_scene(self, size=(5, 5)):
        plt.figure(figsize=size)
        self.scene.plot_scene()

