from math import *

def f(mu, sigma2, x):
	"""
	Computes the output of a gaussian evaluated in 'x' with 'mu' mean and 'sigma2' (co)variance.
	"""

    return 1/sqrt(2.*pi*sigma2) * exp(-.5*(x-mu)**2 / sigma2)

print f(10.,4.,8.) 

# Note: To maximize the output, we should use 'x = mu' as the evaluation point.
