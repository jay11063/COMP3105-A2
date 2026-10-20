# COMP 3105A
# Assignment 1
# 
# Name: Jaeyoon Lee
# Group ID: 105


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import generateData, linearKernel, polyKernel, gaussKernel

GROUP_ID = 105

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

# c)
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


# d)
def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    train_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    # TODO: Change the following random seed to your GROUP ID
    #       (the XX in A2-XX on Brightspace)
    np.random.seed(GROUP_ID)
    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                w, w0 = minSqHinge(Xtrain, ytrain, lamb)
                train_acc_sq_hinge[i, j, r] = np.mean(classify(Xtrain, w,w0)==ytrain) # TODO: compute accuracy on training set
                test_acc_sq_hinge[i, j, r] = np.mean(classify(Xtest, w,w0)==ytest) # TODO: compute accuracy on test set
                
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = np.mean(classify(Xtrain, w,w0)==ytrain) # TODO: compute accuracy on training set
                test_acc_hinge[i, j, r] = np.mean(classify(Xtest, w,w0)==ytest) # TODO: compute accuracy on test set

    # TODO: compute the average accuracies over runs
    avg_train_sq_hinge = np.mean(train_acc_sq_hinge, axis=2)
    avg_test_sq_hinge = np.mean(test_acc_sq_hinge, axis=2)
    avg_train_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_hinge = np.mean(test_acc_hinge, axis=2)

    # TODO: combine accuracies (sq_hinge and hinge)
    train_acc = np.concatenate([avg_train_sq_hinge, avg_train_hinge], axis=1)
    test_acc = np.concatenate([avg_test_sq_hinge, avg_test_hinge], axis=1)

    # TODO: return 4-by-6 train accuracy and 4-by-6 test accuracy
    return train_acc, test_acc


# Question 2: Binary Classification (Adjoint Form)

# a)
def adjSqHinge(X, y, lamb, kernel_func):
    """
    a,a0 = argmin_{a,a0} sum(max(0, 1-m_i)^2) + lambda/2 * a^T @ K @ a
    Args:
        X: (n, d)
        y: (n, 1)
        lamb: scalalr
        kernel_func: function(X,X) -> K
    Returns:
        a: (n, 1)
        a0: scalar
    """
    n, d = X.shape
    K = kernel_func(X, X)

    def obj_func(params):
        """
        Args:
            params: (n+1, 1)
                 = a:[0:n] + a0:[n]
        Returns:
            scalar
        """
        a = params[:n]
        a0 = params[n]

        margin = y.flatten() * (K @ a + a0)

        loss = np.maximum(0, 1 - margin)**2

        reg = (lamb/2) * (a @ K @ a)

        return np.sum(loss) + reg

    init_params = np.zeros(n+1)

    res = minimize(obj_func, init_params)
    best_params = res.x

    alpha = best_params[:n].reshape(n,1)
    alpha0 = best_params[n]

    return alpha, alpha0

# b)
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    """
    a*,a0* = argmin_{w,w0} sum(max(0, 1-m_i)) + lambda/2 * a^T @ K @ a
    Args:
        X: (n, d)
        y: (n, 1)
        lamb: scalar
        stabilizer: scalar
    Returns:
        w: (d, 1)
        w0: scalar
    """
    n, d = X.shape
    K = kernel_func(X,X)

    # q^T = [0_{n+1} 1_n] : (2n, 1)
    q = np.concatenate([np.zeros(n+1), np.ones(n)])

    # G = [G_1 = [0_(n*n)       0_(n,1)         -I_n]
    #      G_2 = [-diag(y)@K    -diag(y)@1_n    -I_n]]
    # G : (2n, 2n+1)
    G = np.concatenate([np.concatenate([np.zeros((n,n)), np.zeros((n,1)), -np.eye(n)], axis=1), # -xi ⪯ 0_n
                        np.concatenate([-np.diag(y.flatten()) @ K, -y, -np.eye(n)], axis=1) # -d(y)(Ka + a0*1_n)-Xi ⪯ -1_n
                       ])

    # h = [h^(1) = 0_n
    #      h^(2) = 1_n]
    # h : (2n, 1)
    h = np.concatenate([np.zeros((n,1)), -np.ones((n,1))])

    # P = [[lamb@K, 0, 0],
    #      [0, 0, 0],
    #      [0, 0, 0]]
    # P : (2n+1, 2n+1)
    P = np.zeros((2*n+1, 2*n+1))
    P[:n, :n] = lamb * K
    P = P + stabilizer * np.eye(2*n+1)

    solvers.options["show_progress"] = False

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    res = solvers.qp(P, q, G, h)
    a = res['x'][:n]
    a0 = res['x'][n]
    return np.array(a), float(a0)


# c) 
def adjClassify(Xtest, a, a0, X, kernel_func):
    """
    yhat = sign(kernel(x^T, X) @ alpha + alpha0)
    Args:
        Xtest: (m, d)
        a    : (n, 1)
        a0   : scalar
    Returns:
        yhat : (m, 1)
    """
    K_test = kernel_func(Xtest, X)
    yhat = np.sign(K_test @ a + a0)
    yhat[yhat == 0] = 1.0
    return yhat


# d)
def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
                   lambda X1, X2: polyKernel(X1, X2, 2),
                   lambda X1, X2: polyKernel(X1, X2, 3),
                   lambda X1, X2: gaussKernel(X1, X2, 1.0),
                   lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])

    # TODO: Change the following random seed to your GROUP ID
    #       (the XX in A2-XX on Brightspace)
    np.random.seed(GROUP_ID)
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                a, a0 = adjSqHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_sq_hinge[i, j, r] = np.mean(adjClassify(Xtrain, a, a0, Xtrain, kernel)==ytrain) # TODO: compute accuracy on training set
                test_acc_sq_hinge[i, j, r] = np.mean(adjClassify(Xtest, a, a0, Xtrain, kernel)==ytest) # TODO: compute accuracy on test set

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = np.mean(adjClassify(Xtrain, a, a0, Xtrain, kernel)==ytrain) # TODO: compute accuracy on training set
                test_acc_hinge[i, j, r] = np.mean(adjClassify(Xtest, a, a0, Xtrain, kernel)==ytest) # TODO: compute accuracy on test set

    # TODO: compute the average accuracies over runs
    avg_train_sq_hinge = np.mean(train_acc_sq_hinge, axis=2)
    avg_test_sq_hinge = np.mean(test_acc_sq_hinge, axis=2)
    avg_train_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_hinge = np.mean(test_acc_hinge, axis=2)

    # TODO: combine accuracies (sq_hinge and hinge)
    train_acc = np.concatenate([avg_train_sq_hinge, avg_train_hinge], axis=1)
    test_acc = np.concatenate([avg_test_sq_hinge, avg_test_hinge], axis=1)

    # TODO: return 5-by-6 train accuracy and 5-by-6 test accuracy
    return train_acc, test_acc
