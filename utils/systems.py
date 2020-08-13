"""
Dynamics
Using simplified custom system classes
"""
import numpy as np
from pydrake.all import (AutoDiffXd, autoDiffToGradientMatrix, initializeAutoDiff)


class System():

    def __init__(self, sys_opts):
        self.dt = sys_opts['dt']
        self.nx = sys_opts['nx']
        self.nu = sys_opts['nu']
        self.nw = sys_opts['nw']
        self.dir = 1

    def dynamics(self, state, inputs, uncertainty=None):
        raise NotImplementedError('Specify a sublcass with dynamics')

    def nextState(self, state, inputs):
        # wrapper for forwards integration
        if self.dir == -1:
            self.dir = 1
        return self.dynamics(state, inputs)

    def prevState(self, state, inputs):
        # backwards integration
        if self.dir == 1:
            self.dir = -1
        return self.dynamics(state, inputs)

    def simulate(self, state, inputs, num_steps, direction):
        # simulate num_steps with single input
        x = state
        for i in range(num_steps):
            if direction == 'forward':
                x = self.nextState(x, inputs)
            elif direction == 'backward':
                x = self.prevState(x, inputs)
        return x

    def getJacobians(self, x, u, w=None):
        if w is None:
            w = np.zeros((self.nw, 1))
        # format for autodiff
        xuw = np.vstack((x, u, w))
        xuw_autodiff = initializeAutoDiff(xuw)
        # name and split here for readability
        x_autodiff = xuw_autodiff[:self.nx, :]
        u_autodiff = xuw_autodiff[self.nx:self.nx+self.nu, :]
        w_autodiff = xuw_autodiff[self.nx+self.nu:, :]
        x_next_autodiff = self.dynamics(x_autodiff, u_autodiff, w_autodiff)
        # nice function organize for us and return gradient matrix
        x_next_gradient = autoDiffToGradientMatrix(x_next_autodiff)
        # split into Ai, Bi, Gi
        Ai = x_next_gradient[:, 0:self.nx]
        Bi = x_next_gradient[:, self.nx:self.nx+self.nu]
        Gi = x_next_gradient[:, self.nx+self.nu:]
        return Ai, Bi, Gi


class Car(System):


    def dynamics(self, state, inputs, uncertainty=None):
        if uncertainty is None:
            uncertainty = np.zeros((self.nw, 1))
        #["x_pos", "y_pos", "heading", "speed", "steer_angle"]
        x_next = np.array([
            state[0] + self.dir*self.dt*(state[3]*np.cos(state[2])),
            state[1] + self.dir*self.dt*(state[3]*np.sin(state[2])),
            state[2] + self.dir*self.dt*(state[3]*np.tan(state[4] )),#+ uncertainty[0])),
            state[3] + self.dir*self.dt*(inputs[0]),
            state[4] + self.dir*self.dt*(inputs[1] )])#+ uncertainty[1])])
        return x_next


