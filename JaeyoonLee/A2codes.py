# COMP 3105A
# Assignment 1
# 
# Name: Jaeyoon Lee
# Seed: 105


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import generateData, linearKernel, polyKernel, gaussKernel


# Question 1: Binary Classification (Primal Form)

# a)
def minSqHinge(X, y, lamb):
    """
    w,w0 = argmin_{w,w0} sum(max(0, 1-m_i)^2) + lambda/2 * ||w||_2^2
    Args:
        X: (n, d)
        y: (n, 1)
        lamb: scalalr
    Returns:
        w: (d, 1)
        w0: scalar
    """
    n,d = X.shape

    def obj_func(params):
        """
        Args:
            params: (d+1, 1)
                 = w:[0:d] + w0:[d]
        Returns:
            scalar
        """
        w_curr = params[:d].reshape(d,1)
        w0_curr = params[d]

        margin = y * (X @ w_curr + w0_curr)

        loss = np.maximum(0, 1 - margin)**2

        reg = (lamb/2) * np.sum(w_curr**2)

        return np.sum(loss) + reg

    init_params = np.zeros(d+1)

    res = minimize(obj_func, init_params)
    best_params = res.x

    w = best_params[:d].reshape(d,1)
    w0 = best_params[d]
    return w, w0


# b)
def minHinge(X, y, lamb, stabilizer=1e-5):
    """
    w*,w0* = argmin_{w,w0} sum(max(0, 1-m_i)) + lambda/2 * ||w||_2^2
    Args:
        X: (n, d)
        y: (n, 1)
        lamb: scalar
        stabilizer: scalar
    Returns:
        w: (d, 1)
        w0: scalar
    """
    # first d variables are w, second n variables are delta
    n, d = X.shape

    # q^T = [0_{d+1} 1_n] : (d+n, 1)
    q = np.concatenate([np.zeros(d+1), np.ones(n)])

    # G = [G_1 = [0_(n*d)       0_(n,1)         -I_n]
    #      G_2 = [-diag(y)@X    -diag(y)@1_n    -I_n]]
    # G : (2n, d+1+n)
    G = np.concatenate([np.concatenate([np.zeros((n,d)), np.zeros((n,1)), -np.eye(n)], axis=1), # -xi ⪯ 0_n
                        np.concatenate([-np.diag(y.flatten()) @ X, -y, -np.eye(n)], axis=1) # -d(y)(X@w + w0@1_n)-Xi ⪯ -1_n
                       ])

    # h = [h^(1) = 0_n
    #      h^(2) = 1_n]
    # h : (2n, 1)
    h = np.concatenate([np.zeros((n,1)), -np.ones((n,1))])

    # P = [[lamb@I_d, 0, 0],
    #      [0, 0, 0],
    #      [0, 0, 0]]
    # P : (d+1+n, d+1+n)
    P = np.zeros((d+1+n, d+1+n))
    P[:d, :d] = lamb * np.eye(d)
    P = P + stabilizer * np.eye(n+d+1)

    solvers.options["show_progress"] = False

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    res = solvers.qp(P, q, G, h)
    w = res['x'][:d]
    w0 = res['x'][d]
    return np.array(w), float(w0)

def classify(Xtest, w, w0):
    """
    yhat = sign(x^T @ w + w0)
    Args:
        Xtest: (m, d)
        w    : (d, 1)
        w0   : scalar
    Returns:
        yhat : (m, 1)
    """
    yhat = np.sign(Xtest @ w + w0)
    yhat[yhat == 0] = 1.0
    return yhat
