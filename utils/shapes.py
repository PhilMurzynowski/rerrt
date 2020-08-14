"""
Shape Classes
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Rectangle:
    """
    Class used to describe a rectangle in 2D. In the context of RERRT, used to
    obtain linear equations for obstacle and collision checking.
    attributes:
        angle       angle in degrees rectangle is rotated by, rotated around bottomleft_corner
        vertices    4 vertices of the rectangle, first is bottomleft_corner, proceeding counterclockwise
        w           width of rectangle
        h           height of rectangle
    Note: remaining attributes redundant, can be cleaned
    Note: Can be updated to inherit from a more general ConvexPolygon Class in n-dim
    """


    def __init__(self, bottomleft_corner, width, height, angle=0):
        """Given bottomleft_corner width, height and angle creates rectangle object by calculating four
        vertices and corresponding linear equations.
        Note: remove redundant attributes
        """
        self.angle = angle
        self.angle_rad = angle*np.pi/180
        self.v1 = np.array(bottomleft_corner).reshape(2, 1)
        self.v2 = self.v1 + np.array([width*np.cos(self.angle_rad), width*np.sin(self.angle_rad)]).reshape(2, 1)
        self.v3 = self.v1 + np.array([-height*np.sin(self.angle_rad), height*np.cos(self.angle_rad)]).reshape(2, 1)
        self.v4 = self.v2 + self.v3 - self.v1
        self.vertices = [self.v1, self.v2, self.v4, self.v3]
        self.num = len(self.vertices)
        self.w = width
        self.h = height
        self.getPolyhedron()

    def getLinearEqns(self, two_points):
        """Given two points obtains linear equation which can be used to describe them.
        Can describe two poits with ax + by = c, returns a, b, c.
        Uses (y1-y2) * x + (x2-x1) * y + (x1-x2)*y1 + (y2-y1)*x1 = 0
        """
        x1, y1 = two_points[0]
        x2, y2 = two_points[1]
        return y1-y2, x2-x1, (x1-x2)*-y1 - (y2-y1)*x1

    def getAb(self, adj_pairs):
        """Given list of pairs of adjacent points, determines A and b to describe rectangle.
        From Ax = b where A is a matrix, x is a vector, b is a vector.
        """
        self.A = np.zeros((4, 2))
        self.b = np.zeros((4, 1))
        for i, pair in enumerate(adj_pairs):
            self.A[i][0], self.A[i][1], self.b[i] = self.getLinearEqns(pair)
        self.A = -self.A
        self.b = -self.b    # flip sign to match literature for inequality
        # written as inequalities <= the conjunction of half spaces gives obstacle

    def getC(self):
        """Used to describe a polyhedron with linear inequalites in terms of its vertices.
        A matter of moving around terms. Equivalent to using Ax=b.
        """
        self.c = np.zeros((4, 2, 1))
        self.c[0] = self.v1.reshape(2, 1)
        self.c[1] = self.v4.reshape(2, 1)
        self.c[2] = self.v4.reshape(2, 1)
        self.c[3] = self.v1.reshape(2, 1)

    def getPolyhedron(self):
        """Using the vertices that have been calculated for the polyhedron,
        determines the linear inequalities which can also be used to describe it.
        """
        adj_pairs = []
        for i in range(self.num-1):
            adj_pairs.append(self.vertices[i:(i+2)])
        adj_pairs.append([self.vertices[-1], self.vertices[0]])
        self.getAb(adj_pairs)
        self.getC()

    def inPoly(self, point):
        """Check whether a point is within this object.
        Note: reshape should simply create a view object, not copy, confirm
        """
        halfspaces = self.A.dot(point).reshape(4, 1) > self.b
        if np.any(halfspaces):
            return False
        return True


class Ellipse():
    """
    Note: Can be updated to inherit from a more general Ellipsoid class in n-dim
    Class used to describe an ellipse in 2D. In the context of RERRT, used for
    uncertainty visualization and collision checking.
    attributes:
        mtx         :nparray: (2x2)     matrix describing the ellipse
        c           :nparray: (2x1)     centerpoint of the ellipse
        w           :float:             width
        h           :float:             height
        angle       :float:             angle ellipse rotated from horizontal, degrees
        hlfmtxpts   :nparray: (2x4)     4 pts on the edge of ellipse, obtained by taking c+-col(mtx^(1/2))
        area        :float:             area of the ellipse
    """


    def __init__(self, center, matrix):
        self.mtx = matrix
        self.c = center
        self.hlfmtxpts = None
        self.area = None

    def convertFromMatrix(self, mtx=None):
        """Obtain information from mtx describing ellipse that is necessary for plotting.
        Including, width, height, angle.
        Note: generalize to higher dim
        Note: verify correctness
        """
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
        """Function to convert angle, width, height into matrix describing desired ellipse.
        Largely used to verify functionality of convertFromMatrix is it is its inverse operation.
        Note: generalize to higher dim
        Note: verify correctness
        """
        if angle is None:
            angle = self.angle
        if w is None:
            w = self.w
        if h is None:
            h = self.h
        rotate = getRotationMtx2D(angle)
        #self.mtx = np.linalg.inv(rotate@np.diag(((2/w)**2, (2/h)**2))@rotate.T)
        self.mtx = np.linalg.inv(rotate@np.diag(((2/w)**2, (2/h)**2))@rotate.T)

    def getHalfMtxPts(self):
        """Obtain 4 pts on the edge of ellipse, can get by taking c+-col(mtx^(1/2))
        """
        if self.hlfmtxpts is None:
            mtx_half = scipy.linalg.sqrtm(self.mtx)
            halfmtx_pts = np.zeros((2, 2*mtx_half.shape[0]))
            for i in range(mtx_half.shape[0]):
                halfmtx_pts[:, i] = mtx_half[:, i]
                halfmtx_pts[:, i + mtx_half.shape[0]] = -mtx_half[:, i]
            self.halfmtxpts = halfmtx_pts + self.c
        return self.halfmtxpts

    def support(self, dir, dim='3D', exact=True):
        """Unfinished/buggy support function for GJK algorithm.
        Currently does not return farthest exactly in desired direction.
        """
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
        """Returns area of ellipse. Would generalize to volume in higher dim."""
        if self.area is None:
            self.area = np.pi*self.w/2*self.h/2
        return self.area



