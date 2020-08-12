"""
Shape Classes
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class Rectangle:


    def __init__(self, bottomleft_corner, width, height, Pc=None, angle=0):
        """cjt is the center of rectangle, thought of as translation.
           may be more efficient to incorporate time into object"""
        self.Pc = Pc    # covariance, add fnc to rotate appropriately
        self.angle = angle
        self.angle_rad = angle*np.pi/180
        # bottomleft_corner is defined with regards to angle=0 and rotates appropriately
        self.v1 = np.array(bottomleft_corner).reshape(2, 1)
        self.v2 = self.v1 + np.array([width*np.cos(self.angle_rad), width*np.sin(self.angle_rad)]).reshape(2, 1)
        self.v3 = self.v1 + np.array([-height*np.sin(self.angle_rad), height*np.cos(self.angle_rad)]).reshape(2, 1)
        self.v4 = self.v2 + self.v3 - self.v1
        self.vertices = [self.v1, self.v2, self.v4, self.v3]
        self.num = len(self.vertices)
        self.w = width
        self.h = height

        self.get_polyhedron()


    def get_lineqns(self, two_points):
        # want ax + by = c from two pts
        # returns a, b, c
        # (y1-y2) * x + (x2-x1) * y + (x1-x2)*y1 + (y2-y1)*x1 = 0
        x1, y1 = two_points[0]
        x2, y2 = two_points[1]
        return y1-y2, x2-x1, (x1-x2)*-y1 - (y2-y1)*x1


    def get_Ab(self, adj_pairs):
        self.A = np.zeros((4, 2))
        self.b = np.zeros((4, 1))
        for i, pair in enumerate(adj_pairs):
            self.A[i][0], self.A[i][1], self.b[i] = self.get_lineqns(pair)
        self.A = -self.A
        self.b = -self.b    # flip sign to match literature for inequality
        # written as inequalities <= the conjunction of half spaces gives obstacle
        # can add graphing utility to parse and verify functionality


    def get_c(self):
        # self.c = np.linalg.solve(self.A, self.b)
        # self.c = np.linalg.pinv(self.A).dot(self.b)
        # c_ijt is a point nominally (i.e. cj = 0) on the ith constraint at time step t
        # self.c = np.zeros((4, 2, 1))    # formatting useful to get column vectors later
        # for i in range(4):
        #     # solve for other coordinate if one is 0, warning may divide by 0 if origin
        #     self.c[i][0] = self.b[i]/self.A[i][0]
        self.c = np.zeros((4, 2, 1))
        self.c[0] = self.v1.reshape(2, 1)
        self.c[1] = self.v4.reshape(2, 1)
        self.c[2] = self.v4.reshape(2, 1)
        self.c[3] = self.v1.reshape(2, 1)


    def get_polyhedron(self):
        adj_pairs = []
        for i in range(self.num-1):
            adj_pairs.append(self.vertices[i:(i+2)])
        adj_pairs.append([self.vertices[-1], self.vertices[0]])

        self.get_Ab(adj_pairs)
        self.get_c()


    def inPoly(self, point):
        # reshape is slow, fix later
        #print(self.b)
        halfspaces = self.A.dot(point).reshape(4, 1) > self.b
        #halfspaces = self.A.dot(point) > self.b
        if np.any(halfspaces):
            return False
        return True


    def print_eqns_Ab(self):
        """To see paste into https://www.desmos.com/calculator
           rounding as desmos interprets 'e' in terms of exponential
           not scientific notation"""
        A = np.around(self.A, 2)
        b = np.around(self.b, 2)
        for i in range(np.shape(self.A)[0]):
            print(f'{A[i][0]}x + {A[i][1]}y <= {b[i][0]}')


    def print_eqns_Ac(self):
        """"Verification, should print same as above, disregarding rounding error"""
        A = np.around(self.A, 2)
        c = np.around(self.c, 2)
        for i in range(np.shape(self.A)[0]):
            print(f'{A[i][0]}(x - {c[i][0][0]}) + {A[i][1]}(y - {c[i][1][0]}) <= 0')



class Ellipse():


    def __init__(self, center, matrix):
        self.mtx = matrix
        self.c = center
        # self.c3D = np.append(center, [[0]], axis=0)
        self.hlfmtxpts = None
        self.area = None


    def convertFromMatrix(self, mtx=None):
        # designed for 2x2 input
        # generalize to n dim
        if mtx is None:
            mtx = self.mtx
        e, v, = np.linalg.eig(np.linalg.inv(mtx))
        # this is concerning that I have to do this
        # again, numerical error
        e, v = np.real(e), np.real(v)
        self.w = 2/np.sqrt(e[0])
        self.h = 2/np.sqrt(e[1])
        #print((v[1][0], v[0][0]))
        self.angle = np.degrees(np.arctan2(v[1][0], v[0][0]))
        #if mtx is None:
        #    mtx = self.mtx
        #a = self.mtx[0, 0]
        #b = self.mtx[0, 1]
        #assert np.isclose(b, self.mtx[1, 0])
        #c = self.mtx[1, 1]
        #e1 = (a+c)/2 + np.sqrt(((a-c)/2)**2+b**2)
        #e2 = (a+c)/2 - np.sqrt(((a-c)/2)**2+b**2)
        #if b == 0 and a >= c:
        #    theta = 0
        #elif b == 0 and a < c:
        #    theta = np.pi/2
        #else:
        #    theta = np.arctan2(e1 - a, b)
        #self.w = 2*np.sqrt(e1)
        #self.h = 2*np.sqrt(e2)
        #self.angle = np.degrees(theta)


    def convertToMatrix(self, angle=None, w=None, h=None):
        # designed for 2x2 input
        # generalize to n dim
        if angle is None:
            angle = self.angle
        if w is None:
            w = self.w
        if h is None:
            h = self.h
        rotate = get_rotation_mtx(angle)
        #self.mtx = np.linalg.inv(rotate@np.diag(((2/w)**2, (2/h)**2))@rotate.T)
        self.mtx = np.linalg.inv(rotate@np.diag(((2/w)**2, (2/h)**2))@rotate.T)


    def getHalfMtxPts(self):
        if self.hlfmtxpts is None:
            mtx_half = scipy.linalg.sqrtm(self.mtx)
            halfmtx_pts = np.zeros((2, 2*mtx_half.shape[0]))
            for i in range(mtx_half.shape[0]):
                halfmtx_pts[:, i] = mtx_half[:, i]
                halfmtx_pts[:, i + mtx_half.shape[0]] = -mtx_half[:, i]
            self.halfmtxpts = halfmtx_pts + self.c
        return self.halfmtxpts


    def support(self, dir, dim='3D', exact=True):
        # https://juliareach.github.io/LazySets.jl/latest/lib/sets/Ellipsoid/
        #B = np.linalg.cholesky(np.linalg.inv(self.mtx))
        dir2 = dir[:2]
        if dim == '3D':
            return self.c3D + np.append(self.mtx@dir2/(np.sqrt(dir2.T@self.mtx@dir2)), np.reshape(dir[2, :], (1, 1)), axis=0)
            #return self.c3D + np.append(B@dir2/(np.sqrt(dir2.T@self.mtx@dir2)), np.reshape(dir[2, :], (1, 1)), axis=0)
            #return self.c3D + np.append(dir2*np.linalg.norm(B@dir2, axis=0), np.reshape(dir[2, :], (1, 1)), axis=0)
        elif dim == '2D':
            return self.c + np.array([[self.w/2*dir2[0, 0]], [self.h/2*dir2[1, 0]]])
            #return self.c + self.mtx@dir2/(np.sqrt(dir2.T@self.mtx@dir2))
            #return self.c + B@dir2/(np.sqrt(dir2.T@self.mtx@dir2))
            #return self.c + dir2*np.linalg.norm(B@dir2, axis=0)

    def getArea(self):
        if self.area is None:
            self.area = np.pi*self.w/2*self.h/2
        return self.area

    def volume(self):
        pass


    def drawEllipse(self, color="gray", fill=False):
        ax = plt.gca()
        ellip = patches.Ellipse(self.c, self.w, self.h, self.angle, color=color, fill=fill)
        ax.add_artist(ellip)


