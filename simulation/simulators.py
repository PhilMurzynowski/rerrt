"""
Simulators.
Note: May be a better idea to generalize simulator for different tree types.
      Currently have different types, RRTSimulator and RERRTSimulator.
Note: Simulators may overwrite data for RRT, i.e. assigning S, K values to nodes.
"""

import numpy as np
import matplotlib.pyplot as plt

from visuals.helper import printProgressBar


class RRTSimulator():
    """Note: Create a separate simulator object for each tree.
    """


    def __init__(self, tree, opts):
        self.tree = tree
        self.system = tree.system
        self.opts = opts

    def simulateTrajectory(self, trajectory, quick_check=False):
        """Given trajectory of nodes with computed us (inputs) and Ks (feedback),
        if quick_check is enabled, simply runs the simulation and returns a bool
        to determine whether the trajectory is valid. If quick_check is disabled,
        returns a new trajectory (a sequence of states) determined by applying
        said us, Ks, and sampling unceratinty. Only starting node is
        guaranteed to have same state as trajectory.
        Defining the origin as the time-varying trajectory in order to simplify
        computation, i.e. control input for timestep i will be
        u_i + K_i*dx_i, where dx_i is the deviation from ith point in the
        trajectory at timestep i.
        Note: Can pass in section of trajectory (subtrajectory) if desired.
        trajectory      [:RRTNode:, ]       trajectory, list/tuple of nodes
                                            each with u and K
        quick_check     :bool:              determine whether to quickly check
                                            simulated trajectory or return copy
        """
        N = len(trajectory)
        if quick_check:
            valid = True
            x = trajectory[0].x
            for i in range(0, N-1):
                #print(f'0?: {x-trajectory[i].x}')
                u = trajectory[i].u+trajectory[i].K@(trajectory[i].x-x)
                w = self.sampleUncertainty()
                x = self.system.nextState(x, u, w)
                if not self.tree.validState(x):
                    valid = False
                    return valid
            return valid
        else:
            simulated = np.zeros((self.system.nx, N))
            simulated[:, 0:1] = trajectory[0].x
            for i in range(0, N-1):
                print(f'0?: {trajectory[i].x-simulated[:, i:i+1]}')
                u = trajectory[i].u+trajectory[i].K@(trajectory[i].x-simulated[:,i:i+1])
                w = self.sampleUncertainty()
                simulated[:, i+1:i+2] = self.system.nextState(simulated[:, i:i+1], u, w)
            return simulated

    def sampleUncertainty(self):
        # initially will just call sampleEllipsoid
        # but framework here to generalize and for readability
        # for example may want to sample from surface of ellipsoid
        # for maximum uncertainty
        return self.sampleEllipsoid()

    def sampleEllipsoid(self):
        # must implement
        # will use D from opts
        return np.zeros((self.system.nw, 1))

    def assessTrajectory(self, trajectory, num_simulations, percentage=True):
        """Assess the robustness of the trajectory via Monte Carlo, no longer
        assuming uncertainty values of 0, sampling and simulating forward.
        After simulating the trajectory, robustness is determined as a percentage
        of trajectories with all valid states i.e. in valid region, no obstacle
        collisions, etc.
        trajectory      [:RRTNode:, ]       trajectory, list/tuple of nodes
                                            each with u and K
        num_simulations :int:               number of simulations to run
        percentage      :bool:              whether to return percentage valid
                                            or simply number of valid
        Returns :float: percentage of valid trajectories or number of valid
        trajectories.
        """
        num_valid = 0
        for i in range(num_simulations):
            valid = self.simulateTrajectory(trajectory, quick_check=True)
            if valid: num_valid += 1
        if percentage: return num_valid/num_simulations
        return num_valid

    def assessTree(self, traj_resolution):
        """Note: currently only for backward RRT.
        traj_resolution     :int:       num of simulations per trajectory
        """
        num_valid = 0
        n = 0
        self.tree.start.setSi(np.zeros((self.system.nx, self.system.nx)))
        rrt_tips = self.tree.getTipNodes(gen=False)
        num_traj = len(rrt_tips)
        num_sim = num_traj*traj_resolution
        plt.figure() # debugging
        for startnode in rrt_tips:
            # calc TVLQR, should put into method
            # using same Q and R as rerrt
            # likely slow due to getPath
            path = self.tree.getPath(startnode, reverse=False)
            N = len(path)
            # x, u already provided from tree growth
            print('getting Jacobians')
            for i in range(N-1):
                path[i].getJacobians(self.system)
            print('calc Si')
            for i in range(N-1, 0, -1):
                path[i-1].calcSi(self.opts['Q'], self.opts['R'], path[i])
            print('calc K')
            for i in range(N-1):
                path[i].calcKi(self.opts['R'], path[i+1])
            print('assess traj')
            num_valid += self.assessTrajectory(path, traj_resolution, False)
            n+=traj_resolution

            # debugging
            print('simulating')
            simulated = self.simulateTrajectory(path, quick_check=False)
            print('comparing')
            #print([n.x for n in path])
            #print(simulated)
            plt.plot(simulated[0, :], simulated[1, :])
            printProgressBar('Simulations complete', n, num_sim)
            printProgressBar('| Current % valid', num_valid, n, writeover=False)

        plt.draw()

    # may be unnecessary
    #def trajectoryStats(self, trajectory, num_simulations=1):
    #    """Statics on trajectory. Document more.
    #    trajectory      [:RRTNode:, ]       trajectory, list/tuple of nodes
    #                                        each with u and K
    #    num_simulations :int:               number of simulations to run
    #    Returns information on avg deviation from nominal trajectory, max
    #    deviation, avg and max final deviation from target, and obstacle
    #    collision.
    #    """
    #    avg_dev = None
    #    max_dev = None
    #    avg_final_dev = None
    #    max_final_dev = None
    #    simulations = [None]*num_simulations
    #    deviations = [None]*num_simulations
    #    N = len(trajectory)
    #    nominal = np.zeros((self.system.nx, N))
    #    for i in range(N):
    #        nominal[:, i] = trajectory[i].x
    #    for i in range(num_simulations):
    #        simulations[i] = self.simulateTrajectory(trajectory)
    #        deviations[i] = simulations[i] - nominal

    #    # continue writing here

    #    statistics = {
    #            'avg_dev': avg_dev,
    #            'max_dev': max_dev,
    #            'avg_final_dev': avg_final_dev,
    #            'max_final_dev': max_final_dev
    #            }
    #    return statistics


class RERRTSimulator(RRTSimulator):


    def assessEllipsoids(self):
        """Assess whether trajectory staying within ellipsoids
        """
        pass

    def assessTree(self, num_simulations):
        """Note: currently only for backward RRT.
        """
        for startnode in self.tree.getTipNodes():
           pass
