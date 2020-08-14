"""
Helpful math functions
"""

import numpy as np


def getRotationMtx2D(angle_deg):
    """Given an angle in degrees returns 2x2 rotation matrix.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def isPSD(x, symmetric=False):
    """Checks if matrix x is positive semidefnite.
    """
    if symmetric:
        # can use eigh only if matrix is symmetric
        return np.all(np.linalg.eigvalsh(x) >= 0)
    return np.all(np.linalg.eigvals(x) >= 0)

def isSymmetric(x, rtol=1e-05, atol=1e-08):
    """Checks if matrix x is symmetric.
    """
    return np.allclose(x, x.T, rtol=rtol, atol=atol)

def getNearPSD(x):
    """Ensures matrix x is symmetric, sets negative eigenvalues
    to zero, and resynthesizes.
    """
    S = (x + x.T)/2
    e, v = np.linalg.eig(S)
    e[e < 0] = 0
    return v.dot(np.diag(e)).dot(v.T)

def getNearPD(X, ep=1e-10):
    """Ensures matrix x is symmetric, sets negative eigenvalues
    to small positive value ep, and resynthesizes.
    """
    S = (X + X.T)/2
    e, v = np.linalg.eig(S)
    e[e < 0] = ep
    return v.dot(np.diag(e)).dot(v.T)
