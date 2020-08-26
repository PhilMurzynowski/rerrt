"""
Metrics used in RRT and RERRT
"""

import numpy as np

"""
Distance Metrics
Different possible metrics for computing how 'close' two states are.
"""

# should clean up 2D and ND with decorator to pick dimension
def l2norm2D(state1, state2, system=None):
    """Calculate 2 dimensional l2 norm between two states. Uses first
    two states if dimension of state is greater than 2.
    Typically used if first two states are in physical rectangular coordinates,
    e.g. x and y.
    state1      :nparray: (nx x 1)      state one
    state2      :nparray: (nx x 1)      state two
    Not dependent on system
    """
    p1, p2 = state1, state2
    #p1_xy = p1[:2].reshape(2, 1) if p1.shape != (2, 1) else p2
    #p2_xy = p2[:2].reshape(2, 1) if p2.shape != (2, 1) else p2
    #return np.linalg.norm(p1_xy - p2_xy)
    p1r = p1.reshape(-1, 1) if (p1.ndim == 1 or p1.shape != (-1, 1)) else p1
    p2r = p2.reshape(-1, 1) if (p2.ndim == 1 or p2.shape[1] != 1) else p2
    return np.sqrt((p1r[0, 0]-p2r[0, 0])**2+(p1r[1, 0]-p2r[1, 0])**2)

def l2normND(state1, state2, system=None):
    """Calculate N dimensional l2 norm between two states.
    state1      :nparray: (nx x 1)      state one
    state2      :nparray: (nx x 1)      state two
    Not dependent on system
    """
    return np.linalg.norm(state1 - state2)

def furutaDistanceMetric(state1, state2, system):
    """Basic alternative distance metric designed for the furuta pendulum.
    Favors maneuvers with fewer rotations.
    Returns the sum of the distances between hands and elbows.
    Elbow defined as the perpedicular joint between the first and second arm.
    Hand defined as the end point of the second arm.
    state1      :nparray: (nx x 1)      first state
    state2      :nparray: (nx x 1)      second state
    system      :Furuta:                instance of Furuta, used for params
    """

    def convertElbowCartesian(state, system):
        """Function that given the state of furuta pendulum, returns cartesian
        coordiantes of the elbow joint.
        state   :nparray: (nx x 1)      state of furuta pendulum
        """
        theta1 = state[0, 0]
        x = system.L1*np.cos(theta1)
        y = system.L1*np.sin(theta1)
        z = 0
        return np.hstack((x, y, z))

    def convertHandCartesian(state, system):
        """Function that given the state of furuta pendulum, returns cartesian
        coordinates of end of second arm (end of 'hand' aka not the elbow joint or
        location where torque is being applied).
        state   :nparray: (nx x 1)      state of furuta pendulum
        """
        theta1, theta2 = state[0, 0], state[1, 0]
        x1 = system.L1*np.cos(theta1)
        y1 = system.L1*np.sin(theta1)
        x2 = x1 - y1*np.sin(theta2)*system.L2/system.L1
        y2 = y1 - x1*np.sin(theta2)*system.L2/system.L1
        z2 = -np.cos(theta2)*system.L2
        return np.hstack((x2, y2, z2))

    elb1 = convertElbowCartesian(state1, system)
    elb2 = convertElbowCartesian(state2, system)
    hand1 = convertHandCartesian(state1, system)
    hand2 = convertHandCartesian(state2, system)

    return np.linalg.norm(elb1-elb2)+np.linalg.norm(hand1-hand2)

"""
Other metrics?
"""
