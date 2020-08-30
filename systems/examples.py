
import numpy as np

from systems.primitives import System


class Car(System):
    """Example System subclass for system dynamics.
    Car which can move in a 2D plane. If w not specified assumed to be 0s.
    state: np.array([[x, y, theta, speed, steer_angle]]).T
    Required sys options:
        dt      :float:     timstep to be used in simulation
        nx      :int:       dimension of state
        nu      :int:       dimension of input
        nw      :int:       dimension of uncertainty
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
    Required sys options:
        dt      :float:     timstep to be used in simulation
        nx      :int:       dimension of state
        nu      :int:       dimension of input
        nw      :int:       dimension of uncertainty
        m1      :float:     mass on first arm near elbow
        m2      :float:     mass on second arm
        l1      :float:     distance to m1 (mass not necessarily on end of arm)
        l2      :float:     distance to m2 (mass not necessarily on end of arm)
        L1      :float:     length of first arm
        L2      :float:     length of second arm
        b1      :float:     b1 damping motor bearings
        b2      :float:     b2 damping between pin coupling between arm 1 and arm 2
                                also damping of motor if choose to use a motor at joint
        J1      :float:     J terms, related to inertia tensors or entries in inertia tensors
        J2      :float:         after Simplifications (Section 5)
    """
    def __init__(self, sys_config):
        super().__init__(sys_config)
        self.m1 = sys_config['m1']
        self.m2 = sys_config['m2']
        self.l1 = sys_config['l1']
        self.l2 = sys_config['l2']
        self.L1 = sys_config['L1']
        self.L2 = sys_config['L2']
        self.b1 = sys_config['b1']
        self.b2 = sys_config['b2']
        self.J1 = sys_config['J1']
        self.J2 = sys_config['J2']
        # saving sysconfig incase want to add uncertainty to parameters
        self.sys_config = sys_config

    def dynamics(self, x, u, w=None):
        if w is None:
            w = np.zeros((self.nw, 1))

        # uncertainty for two massess
        self.m1 = self.sys_config['m1'] + w[0, 0]
        self.m2 = self.sys_config['m2'] + w[1, 0]

        theta1 = x[0, 0]
        theta2 = x[1, 0]
        theta1_dot = x[2, 0]
        theta2_dot = x[3, 0]
        # could make torque inputs uncertain with below two lines
        #tau1 = u[0, 0] + w[0, 0]    # temporarily added uncertainty to actuation
        #tau2 = u[1, 0] + w[1, 0]    # usually no actuation here, essentially disturbance
        tau1 = u[0, 0]
        tau2 = u[1, 0]

        g = 9.8
        # applying simplifications
        # not moved to __init__ as may want to add uncertainty in inertia
        J1_hat = self.J1 + self.m1*self.l1**2
        J2_hat = self.J2 + self.m2*self.l2**2
        J0_hat = self.J1 + self.m1*self.l1**2+self.m2*self.L1**2
        # split into parts and added to gether for readability
        # could be rewritten to reduced number of redundant operations if needed
        part1 = np.array([[-J2_hat*self.b1],
                          [self.m2*self.L1*self.l2*np.cos(theta2)*self.b2],
                          [-J2_hat**2*np.sin(2*theta2)],
                          [-1/2*J2_hat*self.m2*self.L1*self.l2*np.cos(theta2)*np.sin(2*theta2)],
                          [J2_hat*self.m2*self.L1*self.l2*np.sin(theta2)]])
        basis = np.array([[theta1_dot],
                          [theta2_dot],
                          [theta1_dot*theta2_dot],
                          [theta1_dot**2],
                          [theta2_dot**2]])
        part2 = np.array([[J2_hat],
                          [-self.m2*self.L1*self.l2*np.cos(theta2)],
                          [1/2*self.m2**2*self.l2**2*self.L1*np.sin(2*theta2)]])
        forces = np.array([[tau1],
                           [tau2],
                           [g]])
        denominator = (J0_hat*J2_hat+J2_hat**2*np.sin(theta2)**2-self.m2**2*self.L1**2*self.l2**2*np.cos(theta2)**2)
        part3 = np.array([[self.m2*self.L1*self.l2*np.cos(theta2)*self.b1],
                          [-self.b2*(J0_hat+J2_hat*np.sin(theta2)**2)],
                          [self.m2*self.L1*self.l2*J2_hat*np.cos(theta2)*np.sin(2*theta2)],
                          [-1/2*self.m2*np.sin(2*theta2)*(J0_hat*J2_hat+J2_hat**2*np.sin(theta2)**2)],
                          [-1/2*self.m2**2*self.L1**2*self.l2**2*np.sin(2*theta2)]])
        part4 = np.array([[-self.m2*self.L1*self.l2*np.cos(theta2)],
                          [J0_hat+J2_hat*np.sin(theta2)**2],
                          [-self.m2*self.l2*np.sin(theta2)*(J0_hat+J2_hat*np.sin(theta2)**2)]])

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
