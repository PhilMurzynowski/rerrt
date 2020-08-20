"""
Simulators.
"""

import numpy as np


class RRTSimulator():


    def __init__(self, tree, system):
        self.tree = tree
        self.system = system

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
            for i in range(0, N):
                u = trajectory[i].u+trajectory[i].K@(x-trajectory[:,i])
                w = self.sampleUncertainty()
                x = self.system.nextState(x, u, w)
                if not self.tree.validState(x):
                    valid = False
                    return valid
            return valid
        else:
            simulated = np.zeros((self.system.nx, N))
            simulated[:, 0] = trajectory[0].x
            for i in range(0, N-1):
                u = trajectory[i].u+trajectory[i].K@(simulated[:,i]-trajectory[:,i])
                w = self.sampleUncertainty()
                simulated[:, i+1] = self.system.nextState(simulated[:, i], u, w)
            return simulated

    def sampleUncertainty(self):
        # initially will just call sampleEllipsoid
        # but framework here to generalize and for readability
        # for example may want to sample from surface of ellipsoid
        # for maximum uncertainty
        return self.sampleEllipsoid()

    def sampleEllipsoid(self):
        # must implement
        return np.array([[0], [0]])

    def assessTrajectory(self, trajectory, num_simulations):
        """Assess the robustness of the trajectory via Monte Carlo, no longer
        assuming uncertainty values of 0, sampling and simulating forward.
        After simulating the trajectory, robustness is determined as a percentage
        of trajectories with all valid states i.e. in valid region, no obstacle
        collisions, etc.
        trajectory      [:RRTNode:, ]       trajectory, list/tuple of nodes
                                            each with u and K
        num_simulations :int:               number of simulations to run
        Returns :float: percentage of valid trajectories.
        """
        num_valid = 0
        for i in range(num_simulations):
            valid = self.simulateTrajectory(trajectory, quick_check=True)
            if valid: num_valid += 1
        return num_valid/num_simulations

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
