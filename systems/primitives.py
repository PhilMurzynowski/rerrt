"""
Dynamics
Using simplified custom system classes
"""
import numpy as np
from pydrake.all import (AutoDiffXd, autoDiffToGradientMatrix, initializeAutoDiff)


class System():
    """Class used to describe a discrete system for use with RERRT. Provide configuration
    dt      :float:     timstep to be used in simulation
    nx      :int:       dimension of state
    nu      :int:       dimension of input
    nw      :int:       dimension of uncertainty
    Can provide dynamics by creating a subclass with required dynamics.
    Car Class example provided in examples.
    """

    def __init__(self, sys_config):
        self.dt = sys_config['dt']
        self.nx = sys_config['nx']
        self.nu = sys_config['nu']
        self.nw = sys_config['nw']

    def dynamics(self, x, u, w=None):
        """Implement in subclass. If w not specified assumed to be 0s.
        x       :nparray: (nx x 1)          state
        u       :nparray: (nu x 1)          input
        w       :nparray: (nw x 1), None    uncertainty
        """
        raise NotImplementedError('Specify a sublcass with dynamics')

    def nextState(self, x, u, w=None):
        """Wrapper for forwards integration
        x       :nparray: (nx x 1)          state
        u       :nparray: (nu x 1)          input
        w       :nparray: (nw x 1), None    uncertainty
        """
        if np.sign(self.dt) == -1:
            self.dt = -self.dt
        return self.dynamics(x, u, w)

    def prevState(self, x, u, w=None):
        """Wrapper for backwards integration
        x       :nparray: (nx x 1)          state
        u       :nparray: (nu x 1)          input
        w       :nparray: (nw x 1), None    uncertainty
        """
        if np.sign(self.dt) == 1:
            self.dt = -self.dt
        return self.dynamics(x, u, w)

    def simulate(self, x_start, u, num_timesteps, direction, w=None):
        """Simulate system in direction holding u for num_timesteps
        x_start     :nparray: (nx x 1)          starting state
        u           :nparray: (nu x 1)          input to simulate with (single input, held constant for num_timsteps)
        n           :int:                       number of timesteps to simulate
        direction   'forward','backward'        forward or backward simulation
        w           :nparray: (nw x 1), None    uncertainty
        """
        x = x_start
        for i in range(num_timesteps):
            if direction == 'forward':
                x = self.nextState(x, u)
            elif direction == 'backward':
                x = self.prevState(x, u)
        return x

    def getJacobians(self, x, u, w=None):
        """Calculate linearized matrices (around provided x, u, w).
        Makes use of drakes autodiff. If w not specified assumed to be 0s.
        x       :nparray: (nx x 1)      state
        u       :nparray: (nu x 1)      input
        w       :nparray: (nw x 1)      uncertainty
        """
        if w is None:
            w = np.zeros((self.nw, 1))
        # make sure getting forward in time
        if np.sign(self.dt) == -1:
            self.dt = -self.dt
        # format for autodiff
        xuw = np.vstack((x, u, w))
        xuw_autodiff = initializeAutoDiff(xuw)
        # name and split here for readability
        x_autodiff = xuw_autodiff[:self.nx, :]
        u_autodiff = xuw_autodiff[self.nx:self.nx+self.nu, :]
        w_autodiff = xuw_autodiff[self.nx+self.nu:, :]
        x_next_autodiff = self.dynamics(x_autodiff, u_autodiff, w_autodiff)
        # nice function organizes and return gradient matrix
        x_next_gradient = autoDiffToGradientMatrix(x_next_autodiff)
        # split into Ai, Bi, Gi
        Ai = x_next_gradient[:, 0:self.nx]
        Bi = x_next_gradient[:, self.nx:self.nx+self.nu]
        Gi = x_next_gradient[:, self.nx+self.nu:]
        return Ai, Bi, Gi


class Input():
    """
    Object designed to provide RERRT with different input options
    when calculating reachable states from a node. Currently supports
    two types, 'deterministic' and 'random' as they are defined below.
    'deterministic' :   Use all provided possible actions
    'random'        :   Sample random subset of possible actions
    If 'deterministic' is desired:
        Two options:
            1. Pass into setActions all the actions to be used and call it a day
            2. Or have a set of actions generated by calling setLimits and determinePossibleActions
    For 'random':
        Exactly the same as above followed by setting the desired number of samples to be
        tried from these possible actions with setNumSamples
    """


    def __init__(self, dim, type_=None):
        """Initialize, has example default configuration
        dim         integer, dimension of the input excluding the one i.e. nu from (nu x 1)
        """
        self.dim = dim
        if type_ is not None:
            self.setType(type_)
        # default configuration
        elif type_ is None:
            self.setType('deterministic')
            self.setLimits(np.ones(dim))
            self.determinePossibleActions(2*np.ones(dim))

    def setLimits(self, limits):
        """Set limits of input, i.e. values for which input must not exceed.
        Limits set as - and + for each value provided e.g.
        limits = np.array([[2, 1]]).T limits will be -2, 2 and -1, 1
        Enforced in simulation by clipping.
        limits      :nparray: (nu x 1)      values for limits
        """
        assert limits.shape == (self.dim, 1)
        self.limits = limits

    def setType(self, type_):
        """Sets type of input.
        type_       :'deterministic'/'random':
        """
        assert type_ == 'deterministic' or type_ == 'random'
        self.type = type_

    def setNumSamples(self, n):
        """Once determinePossibleActions has been called to set all possible actions,
        if input type is random can choose how many actions to sample when calculating
        reachable states. Input type deterministic will always use all provided actions
        n       :int:       number of sample to use
        """
        assert self.type == 'random'
        assert n<= len(self.actions), 'Increase number of actions'
        self.numsamples = n

    def setActions(self, actions):
        """Set provided actions to possible actions.
        actions     :nparray:   (num_different_actions, dim, 1)
        """
        self.actions = actions

    def determinePossibleActions(self, range_, resolutions):
        """Once limits have been set, pass in resolution at which to generate
        possible actions. Higher resolutions will result in more action
        combinations. Range is a fraction for close to approach limits.
        e.g.
            For a 1d input, passing in resolutions=np.array([5])
            will result in 5 different actions between -range_*limit and
            range_*limit.
            For a 2d input, passing in resolutions=np.array([3, 2])
            will create 6 different action combinations, again sampling
            between -range_*limit and range_*limit for each dimension
            limits are described in self.limits
        range           :float:     how close to approach limit
                                    i.e. 0.9 is 90 percent of limit
        resolutions     :nparray:   (nu,)
        """
        assert resolutions.size == self.dim
        num_combinations = int(np.prod(resolutions))
        actions = np.zeros((num_combinations, self.dim, 1))
        for idx, r in enumerate(resolutions):
            repeat = int(num_combinations/r)
            tmp = np.tile(np.linspace(range_*-self.limits[idx, 0], range_*self.limits[idx, 0], r), repeat).reshape(num_combinations, 1)
            actions[:, idx, :] = tmp
        self.setActions(actions)
        if self.type == 'deterministic': self.numsamples = num_combinations

    def getAction(self, idx):
        """Obtain one action from self.actions. If deterministic, uses provided idx.
        If random, generates random idx to get action. Returns idx, action pair for
        bookkeeping necessary if using in RERRT reachable state functions.
        idx         :int:       idx into Input.actions
        Note: slight abuse indexing/keys, using this number as both a key for a dict
        in node.reachable and an index into Input.actions which is a list
        """
        if self.type == 'deterministic': return idx, self.actions[idx]
        rand_idx = np.random.randint(low=0, high=len(self.actions))
        return rand_idx, self.actions[rand_idx]


