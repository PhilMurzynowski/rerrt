import numpy as np


def getRotationMtx(angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def isPSD(x, symmetric=False):
    if symmetric:
        # can use eigh only if matrix is symmetric
        return np.all(np.linalg.eigvalsh(x) >= 0)
    return np.all(np.linalg.eigvals(x) >= 0)

def isSymmetric(x, rtol=1e-05, atol=1e-08):
    return np.allclose(x, x.T, rtol=rtol, atol=atol)

def getNearPSD(X):
    S = (X + X.T)/2
    e, v = np.linalg.eig(S)
    e[e < 0] = 0
    return v.dot(np.diag(e)).dot(v.T)

def getNearPD(X, ep=1e-10):
    S = (X + X.T)/2
    e, v = np.linalg.eig(S)
    e[e < 0] = ep
    return v.dot(np.diag(e)).dot(v.T)
