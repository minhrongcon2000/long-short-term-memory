import numpy as np

def int2binary(x,binary_bit):
	return '0'*(binary_bit-len(np.binary_repr(x)))+np.binary_repr(x)

def binary2int(x):
	out = 0
	bit = len(x)
	for i in range(bit):
		out += x[i]*2**(bit-i-1)
	return out