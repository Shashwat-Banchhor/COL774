import sys
import numpy as np
from numpy import array as arr
import math
import random

file = open(sys.argv[1])
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


# print NEW_SAMPLES[0]
# NEW_SAMPLES = arr(NEW_SAMPLES).astype(np.str)

#///////////////////////#///////////////////////#///////////////////////#///////////////////////#///////////////////////
def sigmoid(net):
	return  1.0 / (1.+math.exp(-1*net))

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


inp = 85
out = 10
batch_sz = 100
hidden_layers  = 1
nrns_in_layer =[10]
non_lnrty = "sigmoid"
n  = 0.1



input = NEW_SAMPLES#[NEW_SAMPLES[0] for i in range(1000)]#[[0.0 for i in range(inp)]]

Y =  [[0.0 for j in range(out)]]
Y = []
for sample in range(len(Samples)):
	l = [0.0 for j in range(out)]
	l[Samples[sample][-1]] = 1.
	Y.append(l) 

# Y = [Y[0] for i in range(1000)]
	

# l_size = [.0085,.0015,.0015,.0010,.0015]
# l_size = [-1]
# parameters = [[0.0 for i in range([])]]
def initialize(nrns_in_layer,inp,out):
	net_parameters  = []
	for l in range(len(nrns_in_layer)):
		if (l==0):
			layer_parameters = [[0.0 for i in range(inp+1)] for j in range(nrns_in_layer[l])]
		else:
			layer_parameters =  [[0.0 for i in range(nrns_in_layer[l-1]+1)] for j in range(nrns_in_layer[l])]
		net_parameters.append(layer_parameters)


	layer_parameters = []
	layer_parameters = [[0.0 for i in range(nrns_in_layer[len(nrns_in_layer)-1]+1)] for j in range(out)]
	net_parameters.append(layer_parameters)
	return net_parameters



net_parameters = initialize(nrns_in_layer,inp,out)


# INITIALIZE THE PARAMETERS
for l in xrange(len(net_parameters)):
			for p in xrange(len(net_parameters[l])):
				for par in range(len(net_parameters[l][p])):
					net_parameters[l][p][par] = random.uniform(-1, 1) #random.uniform(0,0.01)#random.uniform(-l_size[l+1],l_size[l])*np.sqrt(2./l_size[l])
					# print net_parameters[l][p][par]
# exit(0)					

#//// net_parameters has all the parameters of the network

# Get the current output w.r.t current parameters
net_output = [[0.0 for i in range(nrns_in_layer[j])] for j in range(len(nrns_in_layer)) ]
net_output.append([0.0 for i in range(out)])

#// Get output of network using forward propagation
# Label 

#TRAIN TIME
#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////
del_jtheeta = initialize(nrns_in_layer,inp,out)
# Batch Size
r = 1
# Total Samples
m = len(input)
error =0.0
erroe_prev = 0.0
EPOCHS = 10000
for epoch in range(EPOCHS):
	error = 0.0
	for b in range(0,m/r):
		
		# INITIALIZE DEL_J_THEETA FOR NEW BATCH
		for l in xrange(len(del_jtheeta)):
			for p in xrange(len(del_jtheeta[l])):
				for par in range(len(del_jtheeta[l][p])):
					del_jtheeta[l][p][par] = 0.0
		

		for sample in range(b*r,min((b+1)*r,m)):
			# for each layer
			for i in range(len(net_parameters)):
				# for each perceptron in layer
				for j in range(len(net_parameters[i])):
					net = 0
					# for all inputs to this perceptron except bias
					for k in range(len(net_parameters[i][j])-1):
						#// IF LAYER IS ZERO xij is INPUT 
						if (i==0):
							#net += xij*Tij 
							net += input[sample][k]*net_parameters[i][j][k] 
						#ELSE xij is the OUTPUT of previous layer
						else:
							net += net_output[i-1][k]*net_parameters[i][j][k]


					# ADD THE BIAS
					net += net_parameters[i][j][len(net_parameters[i][j])-1]
					# if (b==1):
					# 	print "Net:",net
					net_output[i][j] = activate("sigmoid",net)
					# if (b==1):
						# print "Output:" , net_output[i][j]

			# exit(0)
			# if (b==1 ):	
			# 	print "Net Output"  
			# 	for no in net_output:
			# 		print no
			# 	exit(0)


			# // WE HAVE DONE  A FORWARD PROPAGATION

			# LETS DO BACK PROPAGATION
			del_jtheeta_net = [[0.0 for i in range(nrns_in_layer[j])] for j in range(len(nrns_in_layer)) ]
			del_jtheeta_net.append([0.0 for i in range(out)])

			# // UPDATE the del_jtheeta_net matrix

			# for each layer l
			for l in range(len(del_jtheeta_net)-1,-1,-1):
					if (l==len(del_jtheeta_net)-1):
						#for each perceptron in layer l 
						for p in range(len(del_jtheeta_net[l])):
							# for d in range(out):
							del_jtheeta_net[l][p] = (Y[sample][p]-net_output[l][p])*net_output[l][p]*(1-net_output[l][p])
							# print Y[sample][d] , net_output[l][d]*(1-net_output[l][d])
							del_jtheeta_net[l][p] = -1*del_jtheeta_net[l][p]
							# print del_jtheeta_net[l][p]
						# exit(0)
					else:
						for p in range(len(del_jtheeta_net[l])):
							for d in range(len(del_jtheeta_net[l+1])):
								del_jtheeta_net[l][p] += del_jtheeta_net[l+1][d]*net_output[l][p]*(1-net_output[l][p])*net_parameters[l+1][d][p]
							
			# if (b==1):	
			# 	print "DEL_JT_NET" ,del_jtheeta_net
			# 	exit(0)
			# UPDATE J_THEETA
			for l in xrange(len(del_jtheeta)):
				for p in xrange(len(del_jtheeta[l])):
					#del (J(0)/d0) += 
					for par in range(len(del_jtheeta[l][p])-1):
						if (l ==0):
							# print l , p , par ,del_jtheeta[l][p][par] , del_jtheeta_net[l][p],input[sample][par]
							del_jtheeta[l][p][par] += del_jtheeta_net[l][p]*input[sample][par]
						else:
							del_jtheeta[l][p][par] += del_jtheeta_net[l][p]*net_output[l-1][par]
					#del (J(0)/d0) += bias term
					del_jtheeta[l][p][len(del_jtheeta[l][p])-1] += del_jtheeta_net[l][p]*1		
			

			for d in range(out):
				error += (Y[sample][d]-net_output[-1][d])*(Y[sample][d]-net_output[-1][d])*1./2

			# if (b==1 or b==0):
			# 	# print "DEL_JT", b  ,del_jtheeta_net
			# 	if (b==1):	
			# 		exit(0)

		# if (b==1):
		# 	print "DEL_JT" ,del_jtheeta
		# 	exit(0)

		# exit(0)
			# 
		#//UPDATE AFTER ONE BATCH
		for l in xrange(len(del_jtheeta)):
			for p in xrange(len(del_jtheeta[l])):
				for par in range(len(del_jtheeta[l][p])):
					# print "Before: ",net_parameters[l][p][par],del_jtheeta[l][p][par]	
					net_parameters[l][p][par] += -n*del_jtheeta[l][p][par]	
					# print "After: ",net_parameters[l][p][par]	
		
		# print "net_parameters",net_parameters
		# print "Epoch : ", epoch ,"Sample : ", sample
		# if (b==0):
		# 	exit(0)
	

	TEST_SAMPLES = input #NEW_SAMPLES
	TEST_OUTPUT  = Y
	correct  = 0
	for sample in range(len(TEST_SAMPLES)):
			# for each layer
			for i in range(len(net_parameters)):
				
				# for each perceptron in layer
				for j in range(len(net_parameters[i])):
					net = 0
					# for all inputs to this perceptron except bias
					for k in range(len(net_parameters[i][0])-1):
						#// IF LAYER IS ZERO
						if (i==0):
							net += input[sample][k]*net_parameters[i][j][k] 
						else:
							net += net_output[i-1][k]*net_parameters[i][j][k]

					# ADD THE BIAS
					net += net_parameters[i][j][-1]

					net_output[i][j] = activate("sigmoid",net)

			max_ind  = 0
			curr_max = -1000
			for x in xrange(len(net_output[-1])):
				if (net_output[-1][x] > curr_max):
					curr_max = net_output[-1][x]
					max_ind = x

			test_ind = 0
			for y in range(len(TEST_OUTPUT[sample])):
				if (TEST_OUTPUT[sample][y]==1):
					test_ind = y
			print "P",max_ind , "A",test_ind
			if (max_ind == test_ind):
				correct += 1
				# print "Got it right !!! " , max_ind



	print "Accuracy : ", float(correct)/len(TEST_OUTPUT)







	print "Epoch ",epoch," completed :) error",error
	print "con",error - erroe_prev
	erroe_prev = error

#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////




# TEST TIME
#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////
TEST_SAMPLES = NEW_SAMPLES
TEST_OUTPUT  = Y
correct  = 0
for sample in range(len(TEST_SAMPLES)):
		# for each layer
		for i in range(len(net_parameters)):
			
			# for each perceptron in layer
			for j in range(len(net_parameters[i])):
				net = 0
				# for all inputs to this perceptron except bias
				for k in range(len(net_parameters[i][0])-1):
					#// IF LAYER IS ZERO
					if (i==0):
						net += input[sample][k]*net_parameters[i][j][k] 
					else:
						net += net_output[i-1][k]*net_parameters[i][j][k]

				# ADD THE BIAS
				net += net_parameters[i][j][-1]

				net_output[i][j] = activate("sigmoid",net)

		max_ind  = 0
		curr_max = -1000
		for x in xrange(len(net_output[-1])):
			if (net_output[-1][x] > curr_max):
				curr_max = net_output[-1][x]
				max_ind = x

		test_ind = 0
		for y in range(len(TEST_OUTPUT[sample])):
			if (TEST_OUTPUT[sample][y]==1):
				test_ind = y

		if (max_ind == test_ind):
			correct += 1
			print "Got it right !!! "



print "Accuracy : ", float(correct)/len(TEST_OUTPUT)
#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////#/////////////////























