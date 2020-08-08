"""
File including common classes and functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydrake.all import (AutoDiffXd, autoDiffToGradientMatrix, initializeAutoDiff)
from pydrake.systems.framework import (BasicVector_, LeafSystem_)
from pydrake.systems.scalar_conversion import TemplateSystem



def printProgressBar(text, current_value, max_value, writeover=True):
    start = '\r' if writeover else ' '
    print("{start}{text}: {prog:>5}%".format(start=start, text=text, prog=round(100 * current_value / max_value, 3)), end='')


def getRotationMtx(angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


class Scene:

    def __init__(self, start, goal, region, obstacles):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.region = region
        self.obstacles = obstacles

    def plot_scene(self):
        ax = plt.gca()
        for r in self.obstacles:
            rect = patches.Rectangle(r.v1, r.w, r.h, r.angle, color='cyan')
            ax.add_artist(rect)
        plt.axis([self.region.v1[0]-0.5, self.region.v4[0]+0.5, self.region.v1[1]-0.5, self.region.v4[1]+0.5])
        plt.plot(self.start[0], self.start[1], "xr", markersize=10)
        plt.plot(self.goal[0], self.goal[1], "xb", markersize=10)
        plt.legend(('start', 'goal'), loc='upper left')
        plt.gca().set_aspect('equal')

    def show_scene(self):
        plt.figure()
        plot_scene()
        plt.tight_layout()
        plt.show()


"""Dynamics
Using simplified custom system class
"""


class MySystem():

    def __init__(self, sys_opts):
        self.dt = sys_opts['dt']
        self.nx = sys_opts['nx']
        self.nu = sys_opts['nu']
        self.nw = sys_opts['nw']
        self.dir = 1

    def dynamics(self, state, inputs, uncertainty=None):

        if uncertainty is None:
            uncertainty = np.zeros((self.nw, 1))

        #["x_pos", "y_pos", "heading", "speed", "steer_angle"]
        x_next = np.array([
            state[0] + self.dir*self.dt*(state[3]*np.cos(state[2])),
            state[1] + self.dir*self.dt*(state[3]*np.sin(state[2])),
            state[2] + self.dir*self.dt*(state[3]*np.tan(state[4] + uncertainty[0])),
            state[3] + self.dir*self.dt*(inputs[0]),
            state[4] + self.dir*self.dt*(inputs[1] + uncertainty[1])])
        return x_next

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
