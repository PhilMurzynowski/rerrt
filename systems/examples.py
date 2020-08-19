
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
    Derivation and parameters provided by:
        https://www.hindawi.com/journals/jcse/2011/528341/
    """
    def dynamics(self, x, u, w=None):
        if w is None:
            w = np.zeros((self.nw, 1))

        theta1 = x[0]
        theta2 = x[1]
        theta1_dot = x[2]
        theta2_dot = x[3]
        # assigning values to physical parameters
        # look more into 5. Simplifications to ensure parameters chose below
        # allow for the applied simplications, also 2.2 Assumptions
        g = 9.8
        m1 = 0.300
        m2 = 0.075
        l1 = 0.150
        l2 = 0.148
        L1 = 0.278
        L2 = 0.300
        # b1 damping motor bearings
        # b2 damping between pin coupling between arm 1 and arm 2
        #   also damping of motor if choose to use a motor at joint
        b1 = 1e-4
        b2 = 2.8e-4
        # J terms, related to inertia tensors or entries in inertia tensors
        # after Simplifications (Section 5)
        J1 = 2.48e-2
        J2 = 3.86e-3
        # applying simplifications
        J1_hat = J1 + m1*l1**2
        J2_hat = J2 + m2*l2**2
        J0_hat = J1 + m1*l1**2+m2*L1**2

        # split into parts and added to gether for readability
        # could be rewritten to reduced number of redundant operations if needed
        part1 = np.array([[-J2_hat*b1],
                          [m2*L1*l2*np.cos(theta2)*b2],
                          [-J2_jat**2*np.sin(2*theta2)],
                          [-1/2*J2_hat*m2*L1*l2*np.cos(theta2)*np.sin(2*theta2)],
                          [J2_hat*m2*L1*l2*np.sin(theta2)]])
        basis = np.array([[theta1_dot],
                          [theta2_dot],
                          [theta1_dot*theta2_dot],
                          [theta1_dot**2]
                          [theta2_dot**2]])
        part2 = np.array([[J2_hat],
                          [-m2*L1*l2*np.cos(theta2)],
                          [1/2*m2**2*l2**2*L1*np.sin(2*theta2)]])
        forces = np.array([[u[0]],
                           [u[1]],
                           [g]])
        denominator = (J0_hat*J2_hat+J2_hat**2*np.sin(theta2)**2-m2**2*L1**2*l2**2*np.cos(theta2)**2)
        part3 = np.array([[m2*L1*l2*np.cos(theta2)*b1],
                          [-b2*(J0_hat+J2_hat*np.sin(theta2)**2)],
                          [m2*L1*l2*J2_hat*np.cos(theta2)*np.sin(2*theta2)],
                          [-1/2*m2*np.sin(2*theta2)*(J0_hat*J2_hat+J2_hat**2*np.sin(theta2)**2)],
                          [-1/2*m2**2*L1**2*l2**2*np.sin(2*theta2)]])
        part4 = np.array([[-m2*L1*l2*np.cos(theta2)],
                          [J0_hat+J2_hat*np.sin(theta2)**2],
                          [-m2*l2*np.sin(theta2)*(J0_hat+J2_hat*np.sin(theta2)**2)]])

        theta1_ddot = part1.T.dot(basis)
        theta1_ddot += part2.T.dot(forces)
        theta1_ddot /= denominator
        theta2_ddot = part3.T.dot(basis)
        theta2_ddot += part4.T.dot(forces)
        theta2_ddot /= denominator

        x_next = np.array([
            x[0] + self.dt*theta1_dot,
            x[1] + self.dt*theta2_dot,
            x[2] + self.dt*theta1_ddot,
            x[3] + self.dt*theta2_ddot
            ])
        return x_next
