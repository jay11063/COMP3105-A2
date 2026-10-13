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
    w,w0 = argmin_{w,w0} sum(max(0, 1-m_i)^2 + lambda/2 * ||w||_2^2)
    Args:
        X: (n, d)
        y: (n, 1)
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
    pass

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
