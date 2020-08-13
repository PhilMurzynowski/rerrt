import numpy as np
import scipy
import matplotlib.pyplot as plt

from utils.shapes import Ellipse
from utils.math import isPSD, getNearPD


class RRTNode:


    def __init__(self, x, parent=None):
        self.x = x.reshape(x.size, 1)
        self.parent = parent
        if self.parent is not None:
            self.n = self.parent.n+1    # time sample
        else:
            self.n = 0


class RERRTNode(RRTNode):


    def __init__(self, x, parent=None, u=None, opts=None):
        super().__init__(x, parent)
        # can linearize system at parent since control input is known
        # and propogate Ei
        self.u = None   # control input used at node
        self.A = None   # memory intensive to store matrices
        self.B = None   # but can examine values, variation easily
        self.G = None   # as well as save computation
        self.S = None
        self.K = None
        self.H = None
        self.E = None
        self.ellipse = None
        self.reachable = {}
        if opts['track_children']:
            self.children = []

    def createEllipse(self):
        # let EE bet E^1/2
        EE = scipy.linalg.sqrtm(self.E)
        # take first 2 dimensions and project
        A = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        EEw = np.linalg.pinv(A@EE)
        # last inversion below is to keep it in the xE^-1x < 1 format
        ellipse_projection = np.linalg.inv(EEw.T@EEw)
        self.ellipse = Ellipse(self.x[:2], ellipse_projection)

    def setEi(self, Ei):
        self.E = Ei
        self.createEllipse()

    def setHi(self, Hi):
        self.H = Hi

    def setSi(self, Si):
        self.S = Si

    def set_u(self, u):
        self.u = u

    def getJacobians(self, system):
        # wrapper function
        self.A, self.B, self.G = system.getJacobians(self.x, self.u)

    def calcSi(self, Q, R, nextNode):
        p = nextNode
        self.S = Q + self.A.T@p.S@self.A - self.A.T@p.S@self.B@np.linalg.inv(R + self.B.T@p.S@self.B)@self.B.T@p.S@self.A

    def calcKi(self, R, nextNode):
         p = nextNode
         self.K = np.linalg.inv(R + self.B.T@p.S@self.B)@self.B.T@p.S@self.A

    def propogateEllipse(self, D, nextNode):
        abk = self.A-self.B@self.K
        #print(f'cond: {np.linalg.cond(abk)}')
        #e = np.linalg.eigvals(abk)
        #print(f'e_max: {max(e)} e_min: {min(e)}')
        En = abk@self.E@abk.T
        En += abk@self.H@self.G.T + self.G@self.H.T@abk.T
        En += self.G@D@self.G.T
        Hn = abk@self.H + self.G@D
        #print(np.linalg.eigvals(En))
        if not isPSD(En):
            #print('not PSD')
            #print(np.linalg.eigvals(En))
            #print(f'En:\n {En}')
            En = getNearPD(En)
        # debug ellipses blowingg up
        #if En[0, 0] > 100 or En[1, 1] > 100:
        #    print(f'self.E:\n {self.E}')
        #    print(f'En:\n {En}')
        #    import pdb; pdb.set_trace()
        nextNode.setHi(Hn)
        nextNode.setEi(En)

    def calcReachable(self, system, opts):
        # not sure yet whether including collision checking here
        # will be effecient or detrimental, depends on 
        # likelihood of being in obstacle vs likelihood of being selected
        for i, action in enumerate(opts['input_actions']):
            if opts['direction'] == 'forward':
                self.reachable[i] = system.nextState(self.x, action)
            elif opts['direction'] == 'backward':
                self.reachable[i] = system.prevState(self.x, action)

    def calcReachableMultiTimeStep(self, system, inputConfig, opts):
        # not sure yet whether including collision checking here
        # will be effecient or detrimental, depends on 
        # likelihood of being in obstacle vs likelihood of being selected
        for i, action in enumerate(inputConfig.actions):
            self.reachable[i] = system.simulate(self.x, action, opts['extend_by'], opts['direction'])

    def popReachable(self, idx):
        return self.reachable.pop(idx)

    def addChild(self, child):
        self.children.append(child)

    def calcCost(self):
        pass


