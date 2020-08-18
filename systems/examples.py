
import numpy as np

from systems.primitives import System


class Car(System):
    """Example System subclass for system dynamics.
    Car which can move in a 2D plane. If w not specified assumed to be 0s.
    state: np.array([[x, y, theta, speed, steer_angle]]).T
    """

    def dynamics(self, x, u, w=None):
        if w is None:
            w = np.zeros((self.nw, 1))
        x_next = np.array([
            x[0] + self.dt*(x[3]*np.cos(x[2])),
            x[1] + self.dt*(x[3]*np.sin(x[2])),
            x[2] + self.dt*(x[3]*np.tan(x[4] + w[0])),
            x[3] + self.dt*(u[0]),
            x[4] + self.dt*(u[1] + w[1])])
        return x_next


class Furuta(System):
    """Example System subclass for system dynamics.
    Furuta penulum with horizontal arm attached to motor, vertical arm attached
    to end of horizontal arm. If w not specified assumed to be 0s.
    state: np.array([[theta, phi, theta_dot, phi_dot]]).T
    """
    pass
