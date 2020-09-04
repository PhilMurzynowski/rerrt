"""
Simulators.
Note: May be a better idea to generalize simulator for different tree types.
      Currently have different types, RRTSimulator and RERRTSimulator.
Note: Simulators may overwrite data for RRT, i.e. assigning S, K values to nodes.
"""

import numpy as np
import matplotlib.pyplot as plt

from visuals.helper import printProgressBar, pickRandomColor


class RRTSimulator():
    """Note: Create a separate simulator object for each tree.
    """


    def __init__(self, tree, opts):
        self.tree = tree
        self.system = tree.system
        self.input = tree.input
        self.opts = opts
        # vars used for sampleUncertainty function
        self.Dinv = None
        self.Dinv_sampbounds = None
        # vars used for sampleInitUncertainty function
        self.Einv = None
        self.Einv_sampbounds = None

    def withinGoalEpsilon(self, state, goal_epsilon):
        """Checks whether state is within epsilon of the goal according to
        distanceMetric specified for the tree. Reminder that for backward
        RRT the goal is in fact the 'start', as it is where the tree begins
        growth from.
        Note: Currently only for backward RRT.
        """
        if self.tree.distanceMetric(state, self.tree.start.x, self.system) <= goal_epsilon:
            return True
        return False

    def simulateTrajectory(self, trajectory, goal_epsilon, quick_check=False):
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
        goal_epsilon    :float:             epsilon to use
        quick_check     :bool:              determine whether to quickly check
                                            simulated trajectory or return copy
        """
        N = len(trajectory)
        valid = True
        reach_dest = False
        if quick_check:
            x = trajectory[0].x + self.sampleInitUncertainty()
            for i in range(0, N-1):
                #print(f'0?: {x-trajectory[i].x}')
                u = trajectory[i].u+trajectory[i].K@(trajectory[i].x-x)
                u = np.clip(u, -self.input.limits, self.input.limits)
                w = self.sampleUncertainty()
                x = self.system.nextState(x, u, w)
                if not self.tree.validState(x):
                    valid = False
                    return valid, reach_dest
                # check if reaching goal within final two extensions
                if i >= N-2*self.opts['extend_by']:
                    if self.withinGoalEpsilon(x, goal_epsilon):
                        reach_dest = True
            return valid, reach_dest
        else:
            simulated = np.zeros((self.system.nx, N))
            simulated[:, 0:1] = trajectory[0].x + self.sampleInitUncertainty()
            for i in range(0, N-1):
                #print(f'0?: {trajectory[i].x-simulated[:, i:i+1]}')
                u = trajectory[i].u+trajectory[i].K@(trajectory[i].x-simulated[:,i:i+1])
                u = np.clip(u, -self.input.limits, self.input.limits)
                w = self.sampleUncertainty()
                simulated[:, i+1:i+2] = self.system.nextState(simulated[:, i:i+1], u, w)
                if not self.tree.validState(simulated[:, i+1:i+2]):
                    valid = False
                # check if reaching goal within final two extensions
                if i >= N-2*self.opts['extend_by']:
                    if self.withinGoalEpsilon(simulated[:, i+1:i+2], goal_epsilon):
                        reach_dest = True
            return simulated, valid, reach_dest

    def sampleUncertainty(self):
        # initially will just call sampleEllipsoid
        # but framework here to generalize and for readability
        # for example may want to sample from surface of ellipsoid
        # for maximum uncertainty
        #   can do so by sampling in spherical coordinates and projecting?
        if self.Dinv is None or self.Dinv_sampbounds is None:
            self.Dinv = np.linalg.inv(self.opts['D'])
            self.Dinv_sampbounds = 1/np.sqrt(np.linalg.eigvals(self.Dinv))
        return self.sampleEllipsoid(self.Dinv, self.Dinv_sampbounds, self.system.nw)

    def sampleInitUncertainty(self):
        if self.Einv is None or self.Einv_sampbounds is None:
            self.Einv = np.linalg.inv(self.opts['E0'])
            self.Einv_sampbounds = 1/np.sqrt(np.linalg.eigvals(self.Einv))
        return self.sampleEllipsoid(self.Einv, self.Einv_sampbounds, self.system.nx)

    def sampleEllipsoid(self, invertedmtx, sampbounds, dim):
        # partially optimized, can get few thousand samples in second
        # can use spherical coordinates to optimize, project distribution
        # using CDF
        # sampling periphery of ellipse, so not uniform from within ellipse
        # using below method to sample within an ellipse, ie val < 1, val > 0
        # is fine, but would be much better to sample from surface directly
        # if want to check boundaries
        #return np.zeros((dim, 1))
        # needs to be updated for speed
        val = np.Inf
        while val > 1 or val < 0.9:
            sample = np.random.uniform(-sampbounds, sampbounds, (dim, 1))
            val = sample.T@invertedmtx@sample
        return sample

    def assessTrajectory(self, trajectory, num_simulations,
                         goal_epsilon, visualize, percentage=True):
        """Assess the robustness of the trajectory via Monte Carlo, no longer
        assuming uncertainty values of 0, sampling and simulating forward.
        After simulating the trajectory, robustness is determined as a percentage
        of trajectories with all valid states i.e. in valid region, no obstacle
        collisions, etc.
        trajectory      [:RRTNode:, ]  trajectory, list/tuple of nodes
                                       each with u and K
        num_simulations :int:          number of simulations to run
        goal_epsilon    :float:        epsilon to use
        percentage      :bool:         whether to return percentage valid
                                       or simply number of valid
        visualize       :bool:         whether to plot the simulated trajectories
        Note: Ideally visualizations would be moved out to separate functions
        Returns :float: percentage of valid trajectories or number of valid
        trajectories.
        """
        num_valid = 0
        num_reach_dest = 0
        for i in range(num_simulations):
            if not visualize:
                valid, reach_dest = self.simulateTrajectory(trajectory, goal_epsilon, quick_check=True)
            else:
                simulated, valid, reach_dest = self.simulateTrajectory(trajectory, goal_epsilon, quick_check=False)
                # plotting all of them, could do fractional plotting?
                # especially for smaller timesteps
                plt.plot(simulated[0, :], simulated[1, :], color=pickRandomColor())
            if valid: num_valid += 1
            if reach_dest: num_reach_dest+=1
        if percentage: return num_valid/num_simulations, num_reach_dest/num_simulations
        return num_valid, num_reach_dest


    def assessTree(self, traj_resolution, goal_epsilon, visualize=False):
        """Assess the whether the trajectories in the tree remain in valid
        areas of the state space and whether they reach the goal.
        Reaching the goal is currently defined as being with goal_epsilon of the
        final state within the last two extensions.
        This metric was chosen with highly sensitive systems like the furuta
        pendulum in mind, in which a system may come exceedingly close and then
        rapidly accelerate away. Check for reaching the goal state can be
        changed in simulateTrajectory() and withinGoalEpsilon().
        traj_resolution   :int:        num of simulations per trajectory
        goal_epsilon      :float:      epsilon to use
        visualize         :bool:       whether to plot the simulated trajectories
        Note: Ideally visualizations would be moved out to separate functions
        Note: currently only for backward RRT.
        """
        num_valid = 0
        num_reach_dest = 0
        n = 0
        self.tree.start.setSi(np.zeros((self.system.nx, self.system.nx)))
        rrt_tips = self.tree.getTipNodes(gen=False)
        num_traj = len(rrt_tips)
        num_sim = num_traj*traj_resolution
        print(f'{num_traj} trajectory(s) in tree.')
        for startnode in rrt_tips:
            # calc TVLQR, should put into method
            # using same Q and R as rerrt
            # likely slow due to getPath
            path = self.tree.getPath(startnode, reverse=False)
            N = len(path)
            # x, u already provided from tree growth
            for i in range(N-1):
                path[i].getJacobians(self.system)
            for i in range(N-1, 0, -1):
                path[i-1].calcSi(self.opts['Q'], self.opts['R'], path[i])
            for i in range(N-1):
                path[i].calcKi(self.opts['R'], path[i+1])
            # plotting in here, messy
            new_valid, new_reached = self.assessTrajectory(path, traj_resolution,
                                                           goal_epsilon, visualize, False)
            num_valid+=new_valid
            num_reach_dest+=new_reached
            n+=traj_resolution

            printProgressBar('Simulations complete', n, num_sim)
            printProgressBar('| % valid', num_valid, n, writeover=False)
            printProgressBar('| % reached goal', num_reach_dest, n, writeover=False)

    # Note: could be very helpful in diagnosing accuracy of integration schemes
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


    def __init__(self, tree, opts):
        super().__init__(tree, opts)

    def assessEllipsoids(self):
        """Assess whether trajectory staying within ellipsoids
        """
        pass

    def assessTree(self, traj_resolution, goal_epsilon, visualize=False):
        """Same layout as for RRTSimulator except do not need to compute Ks as
        those have already been computed.
        traj_resolution   :int:        num of simulations per trajectory
        visualize         :bool:       whether to plot the simulated trajectories
        figsize           :(int, int): size for figure to be plotted
        Note: Ideally visualizations would be moved out to separate functions
        Note: currently only for backward RRT.
        """
        num_valid = 0
        num_reach_dest = 0
        n = 0
        rrt_tips = self.tree.getTipNodes(gen=False)
        num_traj = len(rrt_tips)
        num_sim = num_traj*traj_resolution
        print(f'{num_traj} trajectory(s) in tree.')
        # messy to add visualization in here, refactor if necessary
        for startnode in rrt_tips:
            # likely slow due to getPath
            path = self.tree.getPath(startnode, reverse=False, gen=True)
            # plotting in here, messy
            new_valid, new_reached = self.assessTrajectory(path, traj_resolution,
                                                           goal_epsilon, visualize, False)
            num_valid+=new_valid
            num_reach_dest+=new_reached
            n+=traj_resolution

            printProgressBar('Simulations complete', n, num_sim)
            printProgressBar('| % valid', num_valid, n, writeover=False)
            printProgressBar('| % reached goal', num_reach_dest, n, writeover=False)


























