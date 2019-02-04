import numpy as np

def datagen(num_ex,binary_bit):
	m = []
	n = []
	p = []
	largest_num = 2**binary_bit - 1
	for _ in range(num_ex):
		a = np.random.randint(largest_num/2)
		b = np.random.randint(largest_num/2)
		c = a + b
		if c < largest_num:
			m.append([a])
			n.append([b])
			p.append([c])

	return np.array(m), np.array(n), np.array(p)