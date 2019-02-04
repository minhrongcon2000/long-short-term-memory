import numpy as np 
import matplotlib.pyplot as plt 
from data import *
from activation import *
from loss import *
from utils import *

np.random.seed(0) # deterministic randomization

# hyperparameters
input_unit = 2 # number of input units
hidden_unit = 8 # number of hidden units
output_unit = 1 # number of output units
binary_bit = 4 # number of bits 
display_step = 20 # when model displays results
largest_num = 2**binary_bit-1 #the largest number with given number of bits
num_ex=50 # number of training examples we want to generate
alpha = 0.001 # learning rate
maxVal = 10
minVal = -10
# generate data
m,n,p = datagen(num_ex,binary_bit)

m_seq = np.zeros((m.shape[0],binary_bit))
n_seq = np.zeros((n.shape[0],binary_bit))
p_seq = np.zeros((p.shape[0],binary_bit))

for ex_index in range(m.shape[0]):
	m_seq[ex_index] = [float(bits) for bits in int2binary(m[ex_index][0],binary_bit)]
	n_seq[ex_index] = [float(bits) for bits in int2binary(n[ex_index][0],binary_bit)]
	p_seq[ex_index] = [float(bits) for bits in int2binary(p[ex_index][0],binary_bit)]

# parameters
## memory cell parameters
wcx = 2*np.random.random((hidden_unit,input_unit)) - 1
wca = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bc = 2*np.random.random((hidden_unit,1)) - 1
## update-gate parameters
wux = 2*np.random.random((hidden_unit,input_unit)) - 1
wua = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bu = 2*np.random.random((hidden_unit,1)) - 1
## forget-gate parameters
wfx = 2*np.random.random((hidden_unit,input_unit)) - 1
wfa = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bf = 2*np.random.random((hidden_unit,1)) - 1
## output-gate parameters
wox = 2*np.random.random((hidden_unit,input_unit)) - 1
woa = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bo = 2*np.random.random((hidden_unit,1)) - 1
## predict parameters
wya = 2*np.random.random((output_unit,hidden_unit)) - 1
by = 2*np.random.random((output_unit,1)) - 1

a = {0: np.zeros((hidden_unit,num_ex))}
c_tol = {}
c = {0: np.zeros((hidden_unit,num_ex))}
pred = {}

da = {}
dc = {}
dc_tol = {}

# derivative
dwcx = np.zeros_like(wcx)
dwca = np.zeros_like(wca)
dbc = np.zeros_like(bc)

dwux = np.zeros_like(wux)
dwua = np.zeros_like(wua)
dbu = np.zeros_like(bu)

dwfx = np.zeros_like(wfx)
dwfa = np.zeros_like(wfa)
dbf = np.zeros_like(bf)

dwox = np.zeros_like(wox)
dwoa = np.zeros_like(woa)
dbo = np.zeros_like(bo)

dwya = np.zeros_like(wya)
dby = np.zeros_like(by)

j = 0
err = []

for j in range(500):
	overall = 0.
	for time in range(1,binary_bit+1):
		# Forward pass
		x = np.array([m_seq[:,binary_bit-time],n_seq[:,binary_bit-time]])
		y = np.expand_dims(p_seq[:,binary_bit-time],axis=0)

		c_tol[time] = tanh(np.dot(wca,a[time-1]) + np.dot(wcx,x) + bc)

		u_gate = sigmoid(np.dot(wua,a[time-1]) + np.dot(wux,x) + bu) 
		f_gate = sigmoid(np.dot(wfa,a[time-1]) + np.dot(wfx,x) + bf) 
		o_gate = sigmoid(np.dot(woa,a[time-1]) + np.dot(wox,x) + bo) 

		c[time] = u_gate*c_tol[time] + f_gate*c[time-1]

		a[time] = o_gate*tanh(c[time])

		pred[binary_bit-time] = sigmoid(np.dot(wya,a[time])+by)

		overall += crossentropy(pred[binary_bit-time],y)

		# Backpropagation
		error = pred[binary_bit-time] - y
		dwya_update = error.dot(a[time].T)
		dwya += dwya_update
		dby_update = np.sum(error,axis=1,keepdims=True)
		dby += dby_update

		da[time] = wya.T.dot(error)
		do_gate = da[time]*tanh(c[time])*sigmoid(np.dot(woa,a[time-1]) + np.dot(wox,x) + bo, deriv=True)
		dwoa_update = do_gate.dot(a[time-1].T)
		dwoa += dwoa_update
		dwox_update = do_gate.dot(x.T)
		dwox += dwox_update
		dbo_update = np.sum(do_gate,axis=1,keepdims=True)
		dbo += dbo_update

		dc[time] = da[time]*o_gate
		df_gate = dc[time]*c[time-1]*sigmoid(np.dot(wfa,a[time-1]) + np.dot(wfx,x) + bf, deriv=True)
		dwfa_update = df_gate.dot(a[time-1].T)
		dwfa += dwfa_update
		dwfx_update = df_gate.dot(x.T)
		dwfx += dwfx_update
		dbf_update = np.sum(df_gate,axis=1,keepdims=True)
		dbf += dbf_update

		du_gate = dc[time]*c_tol[time]*sigmoid(np.dot(wua,a[time-1]) + np.dot(wux,x) + bu, deriv=True)
		dwua_update = du_gate.dot(a[time-1].T)
		dwua += dwua_update
		dwux_update = du_gate.dot(x.T)
		dwux += dwux_update
		dbu_update = np.sum(du_gate,axis=1,keepdims=True)
		dbu += dbu_update

		dc_tol[time] = dc[time]*u_gate*tanh(np.dot(wca,a[time-1]) + np.dot(wcx,x) + bc, deriv=True)
		dwca_update = dc_tol[time].dot(a[time-1].T)
		dwca += dwca_update
		dwcx_update = dc_tol[time].dot(x.T)
		dwcx += dwcx_update
		dbc_update = np.sum(dc_tol[time],axis=1,keepdims=True)
		dbc += dbu_update

	wcx -= alpha*dwcx 
	wca -= alpha*dwca 
	bc -= alpha*dbc

	wux -= alpha*dwux 
	wua -= alpha*dwua 
	bu -= alpha*dbu

	wfx -= alpha*dwfx 
	wfa -= alpha*dwfa 
	bf -= alpha*dbf

	wox -= alpha*dwox 
	woa -= alpha*dwoa 
	bo -= alpha*dbo

	wya -= alpha*dwya
	by -= alpha*dby

	err.append(overall)

	if j%display_step==0:
			print('--------------------------')
			print('Iteration %d'%j)
			print('Loss %s'%overall)
			test = np.random.randint(m.shape[0])
			prediction = [int(pred[i][0,test] >= 0.5) for i in range(binary_bit)]
			print('%d + %d = %d'%(int(m[test]),int(n[test]),binary2int(prediction)))
			print('--------------------------')
		
	
plt.figure(figsize=(20,10))
plt.title('loss through each iteration')
plt.plot(np.arange(1,1+len(err)),err,label='loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig('without_gradient_clipping.png',dpi=300)
plt.show()






