import sys
import numpy as np
from numpy import array as arr
import math
import random
import time

def read_file(file_train,out):
	
	file = open(file_train)

	s = file.read().split("\n")
	# print len(s)
	Samples = []
	for x in xrange(len(s)):
		p = s[x].split(",")
		Samples.append(p)


	Samples = arr(Samples).astype(np.int)	
	# print len(Samples)
	# print Samples[0]

	V_TYPES =[]

	for v in xrange(10):
		v_types = []
		for x in xrange(len(Samples)):
			if (Samples[x][v] not in v_types):
				v_types.append(Samples[x][v])
			
		V_TYPES.append(v_types)

	# for x in xrange(len(V_TYPES)):
		# print V_TYPES[x]


	NEW_SAMPLES = [[0 for x in range(85)] for x in range(len(Samples))]
	# print NEW_SAMPLES[0]


	for i in xrange(len(Samples)):
		for j in xrange(10):
			if (j%2==0):
				NEW_SAMPLES[i][j/2*17+ (Samples[i][j] -1) ] = 1
			else :
				NEW_SAMPLES[i][j/2*17 + 4 + (Samples[i][j] -1)] = 1


	Y =  [[0.0 for j in range(out)]]
	Y = []
	for sample in range(len(Samples)):
		l = [0.0 for j in range(out)]
		l[Samples[sample][-1]] = 1.
		Y.append(l) 

	return NEW_SAMPLES , Y




def read_data(file_train,out):
	file = open(file_train)

	s = file.read().split("\n")
	# print len(s)
	Samples = []
	# print len(s)
	for x in xrange(len(s)-1):
		p = s[x].split(",")
		# p = arr(p)
		# p.astype(np.int)

		Samples.append(p)

	# print Samples[0]
	Samples = arr(Samples).astype(np.int)

	NEW_SAMPLES = []
	# Y = []
	for i in range(len(Samples)):	
		NEW_SAMPLES.append(Samples[i][:-1])
		# Y.append(Samples[i][-1])

	# Y =  [[0.0 for j in range(out)]]
	Y = []
	for sample in range(len(Samples)):
		l = [0.0 for j in range(10)]
		l[Samples[sample][-1]] = 1.
		Y.append(l) 

	return NEW_SAMPLES , Y 


# print NEW_SAMPLES[0]
# NEW_SAMPLES = arr(NEW_SAMPLES).astype(np.str)

#///////////////////////#///////////////////////#///////////////////////#///////////////////////#///////////////////////
def sigmoid(net):
	return 1/(np.exp(net*(-1)) + 1)

def relu(net):
	if (net < 0):
		return 0
	else :
		return net

def activate(activation_function , net):
	if (activation_function=="sigmoid"):
		return sigmoid(net)
	else:
		return relu(net)

def min(a,b):
	if (a<b):
		return a
	else:
		return b

def relu_derivartive(z):
	z[z<=0] = 0
	z[z>0] = 1

	return z


config = open(sys.argv[1])
config  = config.read().split("\n")


inp = int(config[0])
out = int(config[1])
r = int(config[2])
hidden_layers  = int(config[3])
hu = config[4].split(" ")
nrns_in_layer =[]

for units in range(len(hu)):
	nrns_in_layer.append(int(hu[units]))

non_lnrty = config[5]
learning_rate = config[6]
n = 0.1

print nrns_in_layer

file_train = sys.argv[2]

input,Y = read_data(file_train,out)




# parameters = [[0.0 for i in range([])]]
def initialize(nrns_in_layer,inp,out):
	net_parameters  = []
	bias_parameters = []
	for l in range(len(nrns_in_layer)):
		if (l==0):
			layer_parameters = [[0.0 for i in range(inp)] for j in range(nrns_in_layer[l])]
			bias = [[0.0] for j in range(nrns_in_layer[l])]
		else:
			layer_parameters =  [[0.0 for i in range(nrns_in_layer[l-1])] for j in range(nrns_in_layer[l])]
			bias =  [[0.0] for j in range(nrns_in_layer[l])]

		net_parameters.append(layer_parameters)
		bias_parameters.append(bias)


	layer_parameters = []
	layer_parameters = [[0.0 for i in range(nrns_in_layer[len(nrns_in_layer)-1])] for j in range(out)]
	bias = [[0.0] for j in range(out)]
	net_parameters.append(layer_parameters)
	return net_parameters , bias_parameters
	W,b



net_parameters,bp = initialize(nrns_in_layer,inp,out)
input  = arr(input)

# INITIALIZE THE PARAMETERS
network_parameters = []
del_jtheeta = []
for l in xrange(len(net_parameters)):
			# for p in xrange(len(net_parameters[l])):
			# 	for par in range(len(net_parameters[l][p])):
			# 		net_parameters[l][p][par] = random.uniform(-1.0, 1.0) #random.uniform(0,0.01)#random.uniform(-l_size[l+1],l_size[l])*np.sqrt(2./l_size[l])
					# print net_parameters[l][p][par]

			# np.random.rand(net_parameters[l],net_parameters[l][p])
			p = len(net_parameters)-1
			# network_parameters.append(np.random.rand(len(net_parameters[l]),len(net_parameters[l][p])))
			network_parameters.append(np.random.uniform(-1,1,len(net_parameters[l])*len(net_parameters[l][p])).reshape((len(net_parameters[l]),len(net_parameters[l][p]))) )
			del_jtheeta.append(arr(np.zeros((len(net_parameters[l]),len(net_parameters[l][p]))))) # no bias in W
			# bias_parameters.append(np.random.rand(len(net_parameters[l]),1))


#//// net_parameters has all the parameters of the network
# r = 100

m = r# len(input)/r
# Get the current output w.r.t current parameters
net_output = [arr([0.0 for i in range(nrns_in_layer[j]*m )]) for j in range(len(nrns_in_layer)) ]
net_output.append(arr([0.0 for i in range(out*m)]))


bias_parameters = [arr(np.ndarray.tolist(np.random.uniform(-1,1,nrns_in_layer[j]*m))) for j in range(len(nrns_in_layer)) ]
bias_parameters.append(arr(np.ndarray.tolist(np.random.uniform(-1,1,out*m))))


del_jtheeta_net = [arr([0.0 for i in range(nrns_in_layer[j]*m )]) for j in range(len(nrns_in_layer)) ]
del_jtheeta_net.append(arr([0.0 for i in range(out*m)]))

bias_del_jtheeta = [arr([0.0 for i in range(nrns_in_layer[j])]) for j in range(len(nrns_in_layer)) ]
bias_del_jtheeta.append(arr([0.0 for i in range(out)]))

# net_output = arr(net_output)
for i in range(len(net_output)-1):
	net_output[i] =   net_output[i].reshape(-1,m)
	bias_parameters[i] =   bias_parameters[i].reshape(-1,m)
	del_jtheeta_net[i] =   del_jtheeta_net[i].reshape(-1,m)
	bias_del_jtheeta[i] = bias_del_jtheeta[i].reshape(-1,1)

net_output[-1] = net_output[-1].reshape(-1,m)
bias_parameters[-1] = bias_parameters[-1].reshape(-1,m)
del_jtheeta_net[-1] = del_jtheeta_net[-1].reshape(-1,m)
bias_del_jtheeta[-1] = bias_del_jtheeta[-1].reshape(-1,1)



#TRAIN TIME
#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////

m = len(input)
error =0.0
erroe_prev = 0.0
epoch_prev_prev = 0.0
EPOCHS = 10000

t0 = time.time()

for epoch in range(EPOCHS):
	error = 0.0
	for b in range(0,m/r):

			
		for  l in range(len(del_jtheeta)):
			del_jtheeta[l] = 0.0 * del_jtheeta[l]

	
		inp = input[b*r:(b+1)*r]
		Y_label = Y[b*r:(b+1)*r]
		for i in xrange(len(net_parameters)):
			# for each layer do
			if (i==0):
				# print network_parameters[i].shape , np.transpose(inp).shape ,bias_parameters[i].shape 
				net_output[i]  = np.dot(network_parameters[i],np.transpose(inp)) + bias_parameters[i]
				net_output[i] = np.maximum(net_output[i],0)#1/(np.exp(net_output[i]*(-1)) + 1)
			else:
				
				# new_input = np.insert(net_output[i-1], 0, np.array(np.ones((1,r))), 0)  
				net_output[i] = np.dot(network_parameters[i],(net_output[i-1])) + bias_parameters[i]
				if (i<len(net_parameters)-1):
					net_output[i] = np.maximum(net_output[i],0)
				else:
					net_output[i] = 1/(np.exp(net_output[i]*(-1)) + 1)


			
		# // WE HAVE DONE  A FORWARD PROPAGATION

			



		# LETS DO BACK PROPAGATION
		
		
		
		# REINITIALIZE del_jtheeta_net
		for l in range(len(del_jtheeta_net)):
			del_jtheeta_net[l] = 0.0*del_jtheeta_net[l]



		# // UPDATE the del_jtheeta_net matrix

		# for each layer l
		for l in range(len(del_jtheeta_net)-1,-1,-1):
				
			if (l==len(del_jtheeta_net)-1):
				del_jtheeta_net[l] = -1 * np.subtract(np.transpose(Y_label),net_output[l])*net_output[l]*(1-net_output[l])
			else:
				del_jtheeta_net[l] = np.dot(np.transpose(network_parameters[l+1]),del_jtheeta_net[l+1]) *relu_derivartive(net_output[l])



		# UPDATE J_THEETA
		for l in xrange(len(del_jtheeta)):
			
			if (l==0):
				del_jtheeta[l] = np.dot(del_jtheeta_net[l],inp)
				bias_del_jtheeta[l] = np.sum(del_jtheeta_net[l],axis = 1).reshape(-1,1)
			else:
				# print "DJN",del_jtheeta_net[l].shape
				del_jtheeta[l] = np.dot(del_jtheeta_net[l],np.transpose(net_output[l-1]))
				bias_del_jtheeta[l] = np.sum(del_jtheeta_net[l],axis = 1).reshape(-1,1)

		# for d in range(out):
			# error += (Y[sample][d]-net_output[-1][d])*(Y[sample][d]-net_output[-1][d])*1./2
		ERROR =  Y_label - np.transpose(net_output[-1])
		ERROR = np.dot(np.transpose(ERROR),ERROR)
		
		for error_sum in range(len(ERROR[0])):
			error += 1./2 *ERROR[error_sum][error_sum]

		# error = np.sum(np.sum(ERROR,axis=1),axis=0)
		#//UPDATE AFTER ONE BATCH
		for l in xrange(len(del_jtheeta)):
			# for p in xrange(len(del_jtheeta[l])):
			# 	for par in range(len(del_jtheeta[l][p])):
			# 		# print "Before: ",net_parameters[l][p][par],del_jtheeta[l][p][par]	
			# 		network_parameters[l][p][par] += -n*del_jtheeta[l][p][par]	
			# 		# print "After: ",net_parameters[l][p][par]	
			
			network_parameters[l] = network_parameters[l] + -n*del_jtheeta[l]
			bias_parameters[l] = bias_parameters[l] + -n*arr([[bias_del_jtheeta[l][row][0] for col in range(r)] for row in range(len(bias_del_jtheeta[l]))])






	# print "bias parameters", bias_parameters[0]
	print "Epoch ",epoch," completed :) error",error
	print "con",error - erroe_prev ,abs(float(error-erroe_prev)/m)
	if (learning_rate != "fixed"):	
		if ((abs(float(error-erroe_prev)/m) < 0.0001 ) and (abs(float(erroe_prev_prev-erroe_prev)/m) < 0.0001 )):
			n = n/5
			print "			CHANGED ETA"

	if (epoch ==  350 or (abs(float(error-erroe_prev)/m) < 0.00001 )):
		break
	erroe_prev_prev = erroe_prev
	erroe_prev = error

# if (sys.argv[4]=="f")
t1 = time.time()

print "Train Time: ", t1-t0 	
#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////


net_output = [[0.0 for i in range(nrns_in_layer[j])] for j in range(len(nrns_in_layer)) ]
net_output.append([0.0 for i in range(out)])


# TEST TIME
#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////
TEST_SAMPLES = input
TEST_OUTPUT  = Y
confusion_mat = [[0 for j in range(10)] for i in range(10)]

correct  = 0
# r = 100

def predict_relu(TEST_SAMPLES,TEST_OUTPUT,net_output,net_parameters):
	
	correct = 0
	for b in range(0,len(TEST_OUTPUT)/r):
		
		inp = TEST_SAMPLES[b*r:(b+1)*r]
		Y_label = TEST_OUTPUT[b*r:(b+1)*r]
		for i in xrange(len(net_parameters)):
				# for each layer do
				if (i==0):
					# print network_parameters[i].shape , np.transpose(inp).shape ,bias_parameters[i].shape 
					net_output[i]  = np.dot(network_parameters[i],np.transpose(inp)) + bias_parameters[i]
					net_output[i] = np.maximum(net_output[i],0)#1/(np.exp(net_output[i]*(-1)) + 1)
				else:
					
					# new_input = np.insert(net_output[i-1], 0, np.array(np.ones((1,r))), 0)  
					net_output[i] = np.dot(network_parameters[i],(net_output[i-1])) + bias_parameters[i]
					if (i==len(net_parameters)-1):
						net_output[i] = 1/(np.exp(net_output[i]*(-1)) + 1)
					else:
						net_output[i] = np.maximum(net_output[i],0)

		for i in xrange(r):
			max_ind  = 0
			curr_max = -1000
			
			

			for j in range(len(net_output[-1])):
				# net_output[-1][j][i]
				if (net_output[-1][j][i] > curr_max):
					curr_max = net_output[-1][j][i]
					max_ind = j

			test_ind = 0
			for y in range(len(Y_label[i])):
				if (Y_label[i][y]==1):
					test_ind = y

			confusion_mat[max_ind][test_ind] +=1
			if (max_ind == test_ind):
				correct += 1
	print "Accuracy : ", float(correct)/len(TEST_OUTPUT)
	for i in range(len(confusion_mat)):
		print confusion_mat[i]	

t0 = time.time()
predict_relu(TEST_SAMPLES,TEST_OUTPUT,net_output,network_parameters)
t1 = time.time()
print "Train Test time", t1-t0



file_test = sys.argv[3]
TEST_SAMPLES , TEST_OUTPUT = read_data(file_test,out)
t0 = time.time()
predict_relu(TEST_SAMPLES,TEST_OUTPUT,net_output,network_parameters)
# print len(TE)
t1 = time.time()
print "Test Test time", t1-t0


#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////























