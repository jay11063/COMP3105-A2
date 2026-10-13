# COMP 3105 Assignment 2
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your code. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your code. 
import YourName.A2codes as A2codes
from A2helpers import plotModel, plotAdjModel, plotDualModel, polyKernel, generateData


def _plotCls():

	n = 100
	lamb = 0.01
	gen_model = 1
	kernel_func = lambda X1, X2: polyKernel(X1, X2, 3)

	# Generate data
	Xtrain, ytrain = generateData(n=n, gen_model=gen_model)

	# Learn and plot results
	# Primal
	w, w0 = A2codes.minHinge(Xtrain, ytrain, lamb)
	plotModel(Xtrain, ytrain, w, w0, A2codes.classify)
	# Adjoint
	a, a0 = A2codes.adjHinge(Xtrain, ytrain, lamb, kernel_func)
	plotAdjModel(Xtrain, ytrain, a, a0, kernel_func, A2codes.adjClassify)
	# Dual
	a, b = A2codes.dualHinge(Xtrain, ytrain, lamb, kernel_func)
	plotDualModel(Xtrain, ytrain, a, b, lamb, kernel_func, A2codes.dualClassify)


if __name__ == "__main__":

	_plotCls()
