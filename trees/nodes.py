import numpy as np
import scipy
import matplotlib.pyplot as plt

from utils.shapes import Ellipse
from utils.math import isPSD, getNearPD


class RRTNode:
    """
    Basic structure used to denote point in trajectory with RRT.
    attributes:
        x           State.
        parent      Node that was extended from by a single timestep.
                    Forward in time if forward RRT, backward in time for backward RRT.
        u           Control input used at node.
                    Forward RRT simply chooses u by associated best reachable state to extend from.
                    Backward RRT uses backward integration, simulating the parent node state in reverse
                    to obtain reachable states and then given best reachable state setting respective u
                    for the child node (so that it can be used forward in time).
        n           Time sample, not necessary, mostly debug information.
        children    Children of node, added if tree grows and track_children enabled.
    """


    def __init__(self, x, parent=None, opts=None):
        if x.shape != (x.size, 1): x = x.reshape(x.size, 1)
        self.x = x
        self.parent = parent
        self.u = None
        self.n = self.parent.n+1 if self.parent is not None else 0
        if opts['track_children']:
            self.children = []

    def setU(self, u):
        """ Set control input for node.
        u   :nparray: (nu x 1)
        """
        self.u = u

    def addChild(self, child):
        """Adds child to node. Currently implemented as a list.
        child       :RRTNode:             node to add as child
        """
        self.children.append(child)


class RERRTNode(RRTNode):
    """
    Structure used to denote point in trajectory with RERRT, modified from RRTNode.
    attributes:
        inherits from RRTNode
        A           df/dx linearized dynamics wrt state.
        B           df/du linearized dyanmics wrt input.
        G           df/dw linearized dyanmics wrt uncertainty.
        S           TVLQR Cost matrix.
        K           TVLQR Control matrix.
        H           Effect of past uncertainty on the ellipsoid. Note: need a more accurate description?
        E           Ellipsoidal set capturing the deviation from nominal trajectory due to uncertainty.
        ellipse     Object made with projection of E into 2D (or3D?) for visualization and collision checking.
        reachable   Set of states that are reachable from that node by simulating the system forward extend_by
                    number of times with given system.dt.
    """


    def __init__(self, x, parent=None, u=None, opts=None):
        super().__init__(x, parent, opts)
        self.A = None
        self.B = None
        self.G = None
        self.S = None
        self.K = None
        self.H = None
        self.E = None
        self.ellipse = None
        self.reachable = {}

    def createEllipse(self):
        """Create Ellipse object for node.
        """
        # let EE bet E^1/2
        # take first 2 dimensions and project. Note: update to support higher dimensions?
        EE = scipy.linalg.sqrtm(self.E)
        A = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        EEw = np.linalg.pinv(A@EE)
        # last inversion below is to keep it in the xE^-1x < 1 format
        ellipse_projection = np.linalg.inv(EEw.T@EEw)
        self.ellipse = Ellipse(self.x[:2], ellipse_projection)

    def setEi(self, Ei):
        """Set function to set initial uncertainty in state or called during propagation.
        Automatically creates an associated Ellipse object for visualization and collision checking.
        Ei  :nparray: (nx x nx)
        """
        self.E = Ei
        self.createEllipse()

    def setHi(self, Hi):
        """Set function to set initial uncertainty due to past uncertainty or called during propagation.
        Initial values will always 0 as at the first timestep there is no past uncertainty besides the
        uncertainty captured by E0.
        Hi  :nparray: (nx x nw)
        """
        self.H = Hi

    def setSi(self, Si):
        """Set function to set initial cost or called during propagation.
        For TVLQR, the initial cost would be 0 on the final node and then propagated backwards.
        Si  :nparray: (nx x nx)
        """
        self.S = Si

    def getJacobians(self, system):
        """Obtain linearized matrices for nonlinear system.
        Wrapper function for the system.getJacobians function.
        """
        self.A, self.B, self.G = system.getJacobians(self.x, self.u)

    def calcSi(self, Q, R, nextNode):
        """Ricatti recursion for TVLQR.
        Calculated on the node level to abstract from tree growth direction.
        Q           :nparray: (nx x nx)     TVLQR state penalty matrix
        R           :nparray: (nu x nu)     TVLQR input penalty matrix
        nextNode    :RERRTNode:             node at future timestep
        """
        nextN = nextNode
        self.S = Q + self.A.T@nextN.S@self.A - self.A.T@nextN.S@self.B@np.linalg.inv(R + self.B.T@nextN.S@self.B)@self.B.T@nextN.S@self.A

    def calcKi(self, R, nextNode):
        """Calculate control matrices K for TVLQR.
        the next node must have a valid cost (S) either through initialization or calcSi
        R           :nparray: (nu x nu)     TVLQR input penalty matrix
        nextNode    :RERRTNode:             node at future timestep
        """
        nextN = nextNode
        self.K = np.linalg.inv(R + self.B.T@nextN.S@self.B)@self.B.T@nextN.S@self.A

    def propogateEllipse(self, D, nextNode):
        """Propogate ellipse characerized by E to nextNode at future timestep.
        Ai-BiKi stored as abk for numerical stability to better preserve symmetry
        if Ai-BiKi is poorly conditioned, as may happen with larger timesteps, x
        may get negative eigen values on the order of machine precision.
        This is adjusted with getNearPD
        D           :nparray: (nw x nw)     Matrix describing ellipsoidal uncertainty set
        nextNode    :RERRTNode:             node at future timestep
        """
        abk = self.A-self.B@self.K
        En = abk@self.E@abk.T
        En += abk@self.H@self.G.T + self.G@self.H.T@abk.T
        En += self.G@D@self.G.T
        Hn = abk@self.H + self.G@D
        if not isPSD(En):
            En = getNearPD(En)
        nextNode.setHi(Hn)
        nextNode.setEi(En)

    def calcReachableMultiTimeStep(self, system, input_, opts):
        """Determine the set of states reachable by holding each possible action
        for multiple timesteps. Number of timesteps determined by extend_by.
        If input type is determinstic, calculates reachable state for each determined input.
        If input type is random, Note: not yet implemented.
        Collision checking not included here as it is cheaper to check when trying to add node
        if there is a larger amount  of reachable states. Efficiency determined by likelihood
        of being in obstacle vs likelihood of being selected.
        """
        for i in range(input_.numsamples):
            key, action = input_.getAction(i)
            self.reachable[i] = system.simulate(self.x, action, opts['extend_by'], opts['direction'])

    def popReachable(self, key):
        """Removes reachable state. Meant to be used after creation of node so that the state is not
        double counted as both a node and a reachable state.
        """
        return self.reachable.pop(key)

    def calcCost(self):
        raise NotImplementedError


