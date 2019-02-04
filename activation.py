import numpy as np

def sigmoid(x,deriv=False):
	m = x.max()
	out = 1/(1+np.exp(-x))
	if deriv==True:
		return out*(1-out)
	return out

def tanh(x,deriv=False):
	out = np.tanh(x)
	if deriv==True:
		return 1-out**2
	return out