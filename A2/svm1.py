import sys
import numpy as np
from numpy import linalg
from numpy import array
from numpy import matrix 
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import math

if (sys.argv[3]=="0" and sys.argv[4]=="a"):
	
	#read train

	file = open(sys.argv[1],"r")
	train_dataset = file.read()
	train_samples = train_dataset.split("\n")
	zero_samples = []
	zo_samples = []
	one_samples = []
	zo_label = [] 
	# print len(train_samples)
	for i in xrange(len(train_samples)-1):
		sample = train_samples[i].split(',')
		# print sample[-1].type()
		# exit(0)
		if (sample[-1]== "1"):
			zo_samples.append(sample[:-1])
			zo_label.append("1")
		if (sample[-1]== "0"):
			zo_samples.append(sample[:-1])
			zo_label.append("-1")
		
	one_samples.append(sample[:-1])

	zo_samples = array(zo_samples)
	zo_label   = array(zo_label)

	zo_samples = array(zo_samples.astype(np.float))
	zo_label   = array(zo_label.astype(np.float))
	zo_samples = 1.0/255*zo_samples
	
	no_of_samples = len(zo_samples)
	# print zo_label.shape
	Y = np.zeros((no_of_samples,no_of_samples))
	for i in xrange(no_of_samples):
		Y[i][i] = zo_label[i]
	# print Y.shape , zo_samples.shape
	y  = np.copy(zo_label).reshape(-1,1)
	# XY = np.dot(Y,zo_samples)
	XY  = y*zo_samples
	XY_T =  np.transpose(XY)
	# print 	len(zo_samples)
	Q = np.dot(XY,XY_T)


	#convert to float
	Q = Q*1.0#matrix(Q, (no_of_samples, no_of_samples), 'float')
	Q = Q.astype(np.float)


	C = 1

	# print Q.shape
	# Q = Q.astype(float)
	P = cvxopt_matrix(Q)
	q = cvxopt_matrix(-np.ones((no_of_samples, 1)))
	G = cvxopt_matrix(np.vstack((np.eye(no_of_samples)*-1,np.eye(no_of_samples))))
	h = cvxopt_matrix(np.hstack((np.zeros(no_of_samples), np.ones(no_of_samples) * C)).reshape(-1,1))
	# print "A shape ",zo_label.reshape(1, -1).shape

	A = cvxopt_matrix(zo_label.reshape(1, -1))
	b = cvxopt_matrix(np.zeros(1))
	sol = cvxopt_solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])
	# print "Alpha" , alphas.shape
	# print alphas

	p2 = (zo_label.reshape(-1,1)*alphas).transpose()
	# print p2.shape , zo_samples.shape
	w = np.dot(p2,zo_samples)

	# print w.shape , alphas.shape
	epsilon = 1e-4
	found  =-1
	# print epsilon
	sv = []
	sv_index = []
	for x in xrange(len(alphas)):
		if (alphas[x] > 1e-2):
			# print alphas[x]
			sv.append(zo_samples[x])
			sv_index.append(x)
			found = x
	 	
	# print found
	# print len(sv_index),sv_index
	
	b = zo_label[found] - np.dot(w,(zo_samples[found].reshape(-1,1)))

	



	file = open(sys.argv[2],"r")
	test_train_dataset = file.read()
	test_train_samples = test_train_dataset.split("\n")
	test_zero_samples = []
	test_zo_samples = []
	test_one_samples = []
	test_zo_label = [] 
	# print len(test_train_samples)
	v = dict()
	for i in xrange(len(test_train_samples)-1):
		test_sample = test_train_samples[i].split(',')
		# print sample[-1].type()
		# exit(0)
		if (test_sample[-1] in v):
			v[test_sample[-1]] += 1 
		else:
			v[test_sample[-1]] = 1
		if (test_sample[-1]== "1"):
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append(test_sample[-1])
		if (test_sample[-1]== "0"):
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append("-1")
	
	# print w , b	



	test_zo_samples = array(test_zo_samples).astype(np.float)
	test_zo_label   = array(test_zo_label).astype(np.float)

	test_zo_samples = 1./255*test_zo_samples
	test_zo_label = test_zo_label

	to_predict = test_zo_samples
	to_predict_label = test_zo_label

	prediction = []
	for i in xrange(len(to_predict)):
		if (np.dot(w,to_predict[i]) + b >= 0):
			prediction.append(1.0)
		else:
			prediction.append(-1.0)
	
	count  = 0

	for i in xrange(len(to_predict_label)):
		# print prediction[i]
		if (int(prediction[i]) == int(to_predict_label[i])):
			count += 1

	print float(count)/len(to_predict)*100
	#Q is (XiYi)' * (XiYi)




# # // I have zero and one samples 
elif (sys.argv[3]=="0" and sys.argv[4]=="b"):
	
	import scipy
	from scipy.spatial.distance import pdist , squareform

	# 99.81 and 100
	def gaussian_kernel(x, y, gamma = 0.05):
		# return np.exp(-linalg.norm(x-y)**2 * gamma)
		return np.dot(np.subtract(x,y),np.subtract(x,y))
	
	def gaussian_kernel2(x, y, gamma = 0.05):
		return np.exp(-linalg.norm(x-y)**2 * gamma)
		# return np.dot(np.subtract(x,y),np.subtract(x,y))


	file = open(sys.argv[2],"r")
	test_train_dataset = file.read()
	test_train_samples = test_train_dataset.split("\n")
	test_zero_samples = []
	test_zo_samples = []
	test_one_samples = []
	test_zo_label = [] 
	# print len(test_train_samples)
	v = dict()
	for i in xrange(len(test_train_samples)-1):
		test_sample = test_train_samples[i].split(',')
		# print sample[-1].type()
		# exit(0)
		if (test_sample[-1] in v):
			v[test_sample[-1]] += 1 
		else:
			v[test_sample[-1]] = 1
		if (test_sample[-1]== "1"):
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append(test_sample[-1])
		if (test_sample[-1]== "0"):
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append("-1")
		

	test_zo_samples = array(test_zo_samples).astype(np.float)
	test_zo_label   = array(test_zo_label).astype(np.float)

	test_zo_samples = 1./255*test_zo_samples
	test_zo_label = test_zo_label




	file = open(sys.argv[1],"r")
	train_dataset = file.read()
	train_samples = train_dataset.split("\n")
	zero_samples = []
	zo_samples = []
	one_samples = []
	zo_label = [] 
	# print len(train_samples)
	for i in xrange(len(train_samples)-1):
		sample = train_samples[i].split(',')
		# print sample[-1].type()
		# exit(0)
		if (sample[-1]== "1"):
			zo_samples.append(sample[:-1])
			zo_label.append(sample[-1])
		if (sample[-1]== "0"):
			zo_samples.append(sample[:-1])
			zo_label.append("-1")
		
		# 	if (sample[-1]==0):
	# 		zero_samples.append(sample[:-1])

	# 	elif (sample[-1]==1):
	# 		one_samples.append(sample[:-1])

	zo_samples = array(zo_samples)
	zo_label   = array(zo_label)

	zo_samples = array(zo_samples.astype(np.float))
	zo_label   = array(zo_label.astype(np.float))
	zo_samples = 1.0/255*zo_samples


	no_of_samples = len(zo_samples)
	# print zo_label.shape
	Y = np.zeros((no_of_samples,no_of_samples))
	for i in xrange(no_of_samples):
		Y[i][i] = zo_label[i]
	# print Y.shape , zo_samples.shape
	y  = np.copy(zo_label).reshape(-1,1)
	# XY = np.dot(Y,zo_samples)
	XY  = y*zo_samples
	XY_T =  np.transpose(XY)
	# print 	len(zo_samples)
	Q = np.dot(XY,XY_T)
	t1 = np.zeros((no_of_samples, no_of_samples))
	t2 = np.zeros((no_of_samples, no_of_samples))
	# t1t = np.zeros((no_of_samples, no_of_samples))


	kernel_matrix = np.zeros((no_of_samples, no_of_samples))
	for i in range(no_of_samples):
		p = np.dot(zo_samples[i], zo_samples[i])
		for j in range(no_of_samples):
			# print i,j 
			kernel_matrix[i][j] = gaussian_kernel(zo_samples[i], zo_samples[j])
			t1[i][j] = p 
	t2 = np.dot(zo_samples,zo_samples.transpose())
	t3 = t1.transpose()
	# print t1.shape , t2.shape , t3.shape


	kernel_matrix = np.exp(-0.05*kernel_matrix) #np.subtract(np.subtract(t1,2*t2),t3))		
	# print "done"

	#LIBRARY
	# pw_sq_dist = squareform(pdist(zo_samples,'sqeuclidean'))
	# kernel_matrix = scipy.exp(-pw_sq_dist*0.05)
	# print "done2"
	#convert to float
	Q = Q*1.0#matrix(Q, (no_of_samples, no_of_samples), 'float')
	Q = Q.astype(np.float)
	kernel_matrix = kernel_matrix*1.0


	C = 1

	# print kernel_matrix.shape
	# Q = Q.astype(float)
	P = cvxopt_matrix(np.outer(y,y)*kernel_matrix)
	q = cvxopt_matrix(-np.ones((no_of_samples, 1)))
	G = cvxopt_matrix(np.vstack((np.eye(no_of_samples)*-1,np.eye(no_of_samples))))
	h = cvxopt_matrix(np.hstack((np.zeros(no_of_samples), np.ones(no_of_samples) * C)).reshape(-1,1))
	# print "A shape ",zo_label.reshape(1, -1).shape

	A = cvxopt_matrix(zo_label.reshape(1, -1))
	b = cvxopt_matrix(np.zeros(1))
	sol = cvxopt_solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])
	# print "Alpha" , alphas.shape
	# print alphas

	p2 = (zo_label.reshape(-1,1)*alphas).transpose()
	# print p2.shape , zo_samples.shape



	# print w.shape , alphas.shape
	epsilon = 1e-4
	found  =-1
	# print epsilon
	sv =[]
	sv_index = []
	for x in xrange(len(alphas)):
		if (alphas[x] > 1e-4):
			# print alphas[x]
			found = x
	 		sv_index.append(x)
	 		sv.append(zo_samples[x])
	# print found
	# print len(sv_index),sv_index

 	######################################
	# find b
	z = 0.0
	for j in xrange(len(alphas)):
			# print "b ",j 
			# Use only alphas greater than 0
			if (j in sv_index):
				z += alphas[j]*zo_label[j]*gaussian_kernel2(zo_samples[j],zo_samples[found])

	# z is wTx


	b = zo_label[found] - z

	##########################################
	to_predict = test_zo_samples
	to_predict_label = test_zo_label
	prediction = []
	for i in xrange(len(to_predict)):
		z = 0.0
		for j in xrange(len(alphas)):
		# Use only alphas greater than 0
			if j in sv_index:	
				z += alphas[j]*zo_label[j]*gaussian_kernel2(zo_samples[j],to_predict[i])
		# print i
		if (z+b >=0):
			prediction.append(1.0)
		else:
			prediction.append(-1.0)

	# exit(0)
	count  = 0
	for i in xrange(len(to_predict_label)):
		if (int(prediction[i]) == int(to_predict_label[i])):
			count += 1

	print "Test acc ", float(count)/len(to_predict_label)*100

	to_predict = zo_samples
	to_predict_label = zo_label
	prediction = []
	for i in xrange(len(to_predict)):
		z = 0.0
		for j in xrange(len(alphas)):
			# Use only alphas greater than 0
			if j in sv_index:	
				z += alphas[j]*zo_label[j]*gaussian_kernel2(zo_samples[j],to_predict[i])
		# print i

		if (z+b >=0):
			prediction.append(1.0)
		else:
			prediction.append(-1.0)

	
	count  = 0
	for i in xrange(len(to_predict_label)):
		# print prediction[i]
		if (int(prediction[i]) == int(to_predict_label[i])):
			count += 1

	print "Train acc ",float(count)/len(to_predict_label)*100
	##########################################

elif (sys.argv[3]=="0" and sys.argv[4]=="c"):
	
	# L 99.9
	# G 100 and 99.8

	sys.path.append("/home/shashwat/Desktop/COL774/libsvm-3.23/python")
	import svmutil

# TRAIN FILE	
	file = open(sys.argv[1],"r")
	train_dataset = file.read()
	train_samples = train_dataset.split("\n")
	zero_samples = []
	zo_samples = []
	one_samples = []
	zo_label = [] 
	# print len(train_samples)
	for i in xrange(len(train_samples)-1):
		sample = train_samples[i].split(',')
		# print sample[-1].type()
		# exit(0)
		if (sample[-1]== "1"):
			zo_samples.append(sample[:-1])
			zo_label.append(sample[-1])
		if (sample[-1]== "0"):
			zo_samples.append(sample[:-1])
			zo_label.append("-1")



# TEST FILE
	file = open(sys.argv[2],"r")
	test_train_dataset = file.read()
	test_train_samples = test_train_dataset.split("\n")
	test_zero_samples = []
	test_zo_samples = []
	test_one_samples = []
	test_zo_label = [] 
	# print len(test_train_samples)
	# v = dict()
	for i in xrange(len(test_train_samples)-1):
		test_sample = test_train_samples[i].split(',')
		# print sample[-1].type()
		# exit(0)
		# if (test_sample[-1] in v):
		# 	v[test_sample[-1]] += 1 
		# else:
		# 	v[test_sample[-1]] = 1
		if (test_sample[-1]== "1"):
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append(test_sample[-1])
		if (test_sample[-1]== "0"):
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append("-1")
		

	test_zo_samples = array(test_zo_samples).astype(np.float)
	test_zo_label   = array(test_zo_label).astype(np.float)
	# print len(test_zo_label),len(test_zo_samples)
	# print v
	zo_samples = array((zo_samples)).astype(np.float)
	zo_label   = array((zo_label)).astype(np.float)
	zo_samples = 1./255*zo_samples
	test_zo_samples = 1./255*test_zo_samples
	print "linear_kernel"
	# test_zo_label = [5.0 for x in range(len(test_zo_label))]
	m =  svmutil.svm_train(zo_label,zo_samples,"-t 0" )	
	p ,acc, d=  svmutil.svm_predict(test_zo_label,test_zo_samples,m)
	

	print "gaussian_kernel"
	m =  svmutil.svm_train(zo_label,zo_samples,"-t 2 -g 0.05" )	
	p ,acc, d=  svmutil.svm_predict(test_zo_label,test_zo_samples,m)

	print acc




elif (sys.argv[3]=="1" and sys.argv[4]=="a"):
	
	def gaussian_kernel(x, y, gamma = 0.05):
		# return np.exp(-linalg.norm(x-y)**2 * gamma)
		return np.dot(np.subtract(x,y),np.subtract(x,y))
	
	def gaussian_kernel2(x, y, gamma = 0.05):
		return np.exp(-linalg.norm(x-y)**2 * gamma)
		# return np.dot(np.subtract(x,y),np.subtract(x,y))



	def get_max(d):
		max = 0
		for x in xrange(1,10):
			if (d[max] < d[x]):
				max = x

		return max


	# j >i
	def get_classifier(i ,j,train_samples):
		
		zero_samples = []
		zo_samples = []
		one_samples = []
		zo_label = [] 
		
		for k in xrange(len(train_samples)):
			sample = train_samples[k].split(',')
			
			if (sample[-1]== str(j)):
				zo_samples.append(sample[:-1])
				zo_label.append("1")
			if (sample[-1]== str(i)):
				zo_samples.append(sample[:-1])
				zo_label.append("-1")
		
		zo_samples = array((zo_samples)).astype(np.float)
		zo_label   = array((zo_label)).astype(np.float)
		zo_samples = 1./255*zo_samples
		

		no_of_samples = len(zo_samples)
		# print zo_label.shape
		Y = np.zeros((no_of_samples,no_of_samples))
		for i in xrange(no_of_samples):
			Y[i][i] = zo_label[i]
		# print Y.shape , zo_samples.shape
		y  = np.copy(zo_label).reshape(-1,1)
		# XY = np.dot(Y,zo_samples)
		XY  = y*zo_samples
		XY_T =  np.transpose(XY)
		# print 	len(zo_samples)
		Q = np.dot(XY,XY_T)
		t1 = np.zeros((no_of_samples, no_of_samples))
		t2 = np.zeros((no_of_samples, no_of_samples))


		kernel_matrix = np.zeros((no_of_samples, no_of_samples))
		for i in range(no_of_samples):
			p = np.dot(zo_samples[i], zo_samples[i])
			for j in range(no_of_samples):
				kernel_matrix[i][j] = gaussian_kernel(zo_samples[i], zo_samples[j])
				t1[i][j] = p 
		t2 = np.dot(zo_samples,zo_samples.transpose())
		t3 = t1.transpose()
		# print t1.shape , t2.shape , t3.shape


		kernel_matrix = np.exp(-0.05*kernel_matrix) #np.subtract(np.subtract(t1,2*t2),t3))		
		
		Q = Q*1.0#matrix(Q, (no_of_samples, no_of_samples), 'float')
		Q = Q.astype(np.float)
		kernel_matrix = kernel_matrix*1.0


		C = 1

		
		P = cvxopt_matrix(np.outer(y,y)*kernel_matrix)
		q = cvxopt_matrix(-np.ones((no_of_samples, 1)))
		G = cvxopt_matrix(np.vstack((np.eye(no_of_samples)*-1,np.eye(no_of_samples))))
		h = cvxopt_matrix(np.hstack((np.zeros(no_of_samples), np.ones(no_of_samples) * C)).reshape(-1,1))

		A = cvxopt_matrix(zo_label.reshape(1, -1))
		b = cvxopt_matrix(np.zeros(1))
		sol = cvxopt_solvers.qp(P, q, G, h, A, b)
		alphas = np.array(sol['x'])

		sv_index = []
		found =0
		for x in xrange(len(alphas)):
			if (alphas[x] > 1e-4):
				# print alphas[x]
				found = x
		 		sv_index.append(x)
	 		# sv.append(zo_samples[x])
	 	######################################
		# find b
		z = 0.0
		for j in xrange(len(alphas)):
				# print "b ",j 
				# Use only alphas greater than 0
				if (j in sv_index):
					z += alphas[j]*zo_label[j]*gaussian_kernel2(zo_samples[j],zo_samples[found])

		# z is wTx


		b = zo_label[found] - z
		return alphas,b,zo_samples,zo_label,sv_index




	file = open(sys.argv[1],"r")
	train_dataset = file.read()
	train_samples = train_dataset.split("\n")
	# print len(train_samples)
	train_samples = train_samples[:-1]
	# train_samples = train_samples[:int(9./10*len(train_samples))]

	DIGIT =10
		
	classifiers = []
	B = []
	TrainSamples = []
	TrainLabels = []
	NonZeroAlphas = []
	k = 0
	for i in xrange(DIGIT):
		for j in xrange(i+1,DIGIT):
				model , b , train_X , train_Y , non_zero_index  = get_classifier(i,j,train_samples)
				classifiers.append(model)
				B.append(b)
				TrainSamples.append(train_X)
				TrainLabels.append(train_Y)
				NonZeroAlphas.append(non_zero_index)
				k+=1
				print "classifier ", k


	print "Reading Test File"
# TEST FILE
	file = open(sys.argv[2],"r")
	test_train_dataset = file.read()
	test_train_samples = test_train_dataset.split("\n")
	test_zero_samples = []
	test_zo_samples = []
	test_one_samples = []
	test_zo_label = [] 
	# print len(test_train_samples)
	# v = dict()
	for i in xrange(len(test_train_samples)-1):
		test_sample = test_train_samples[i].split(',')
		
		# if (test_sample[-1]== "1" or test_sample[-1]== "0" or test_sample[-1]== "2"):
		test_zo_samples.append(test_sample[:-1])
		test_zo_label.append(test_sample[-1])
		# if (test_sample[-1]== "0"):
			# test_zo_samples.append(test_sample[:-1])
			# test_zo_label.append(test_sample[-1])
		

	# smaples with label as the digit
	test_zo_samples = array(test_zo_samples).astype(np.float)
	test_zo_label   = array(test_zo_label).astype(np.float)
	test_zo_samples = 1./255*test_zo_samples

	# print len(test_zo_label),len(test_zo_samples)
	
	

	

	
	####################################################################################
	
	# p = prediction


	l = []
	for i in xrange(len(test_zo_samples)):
		d = dict()
		for j in range(10):
			d[j] = 0

		l.append(d)

	k=0
	for i in xrange(DIGIT):
		for j in xrange(i+1,DIGIT):

			# p ,a, c=  svmutil.svm_predict(test_zo_label,test_zo_samples,classifiers[k])
					##########################################
			to_predict = test_zo_samples
			to_predict_label = test_zo_label
			prediction = []
			for i in xrange(len(to_predict)):
				z = 0.0
				for j in xrange(len(classifiers[k])):
				# Use only alphas greater than 0
					if j in NonZeroAlphas[k]:	
						z += classifiers[k][j]*TrainLabels[k][j]*gaussian_kernel2(TrainSamples[k][j],to_predict[i])
				# print i
				if (z+B[k] >=0):
					prediction.append(1.0)
				else:
					prediction.append(-1.0)

			
			p = prediction
			
			for x in xrange(len(p)):
				if (p[x]>0):
					l[x][j] += 1
					# print j
				else:
					l[x][i] += 1
					# print i
			k+=1
			# print "Testing Classifier : ", k





	prediction =[]

	for i in xrange(len(test_zo_samples)):
		m = get_max(l[i])
		prediction.append(m)
		
	# print prediction
	conf_mat  = [[0 for x in range(10)] for x in range(10)]			

	count = 0
	for i in xrange(len(test_zo_samples)):

		conf_mat[prediction[i]][int(test_zo_label[i])] += 1
		# print prediction[i], test_zo_label[i]
		if (int(prediction[i])==int(test_zo_label[i])):
			count += 1


	for x in xrange(10):
		print conf_mat[x]

	print "Acc : ",float(count)/len(test_zo_label) * 100



	


elif (sys.argv[3]=="1" and (sys.argv[4]=="b" or sys.argv[4]=="c")):
	sys.path.append("/home/shashwat/Desktop/COL774/libsvm-3.23/python")
	import svmutil


# 	sys.path.append("/home/shashwat/Desktop/COL774/libsvm-3.23/python")
# 	import svmutil

# 	def get_max(d):
# 		max = 0
# 		for x in xrange(1,10):
# 			if (d[max] < d[x]):
# 				max = x

# 		return max
# # TRAIN FILE	
# 	file = open(sys.argv[1],"r")
# 	train_dataset = file.read()
# 	train_samples = train_dataset.split("\n")
# 	zero_samples = []
# 	zo_samples = []
# 	one_samples = []
# 	zo_label = [] 
# 	print len(train_samples)
# 	for i in xrange(len(train_samples)-1):
# 		sample = train_samples[i].split(',')
# 		# print sample[-1].type()
# 		# exit(0)
# 		if (sample[-1]== "1"):
# 			zo_samples.append(sample[:-1])
# 			zo_label.append(sample[-1])
# 		if (sample[-1]== "0"):
# 			zo_samples.append(sample[:-1])
# 			zo_label.append("-1")



# # TEST FILE
# 	file = open(sys.argv[2],"r")
# 	test_train_dataset = file.read()
# 	test_train_samples = test_train_dataset.split("\n")
# 	test_zero_samples = []
# 	test_zo_samples = []
# 	test_one_samples = []
# 	test_zo_label = [] 
# 	print len(test_train_samples)
# 	# v = dict()
# 	for i in xrange(len(test_train_samples)-1):
# 		test_sample = test_train_samples[i].split(',')
# 		# print sample[-1].type()
# 		# exit(0)
# 		# if (test_sample[-1] in v):
# 		# 	v[test_sample[-1]] += 1 
# 		# else:
# 		# 	v[test_sample[-1]] = 1
# 		if (test_sample[-1]== "1"):
# 			test_zo_samples.append(test_sample[:-1])
# 			test_zo_label.append(test_sample[-1])
# 		if (test_sample[-1]== "0"):
# 			test_zo_samples.append(test_sample[:-1])
# 			test_zo_label.append(test_sample[-1])#("-1")
		

# 	test_zo_samples = array(test_zo_samples).astype(np.float)
# 	test_zo_label   = array(test_zo_label).astype(np.float)
# 	print len(test_zo_label),len(test_zo_samples)
# 	# print v
# 	zo_samples = array((zo_samples)).astype(np.float)
# 	zo_label   = array((zo_label)).astype(np.float)
# 	zo_samples = 1./255*zo_samples
# 	test_zo_samples = 1./255*test_zo_samples
# 	# print "linear_kernel"
# 	# test_zo_label = [5.0 for x in range(len(test_zo_label))]
# 	# m =  svmutil.svm_train(zo_label,zo_samples,"-t 0" )	

	
	
# 	l = []
# 	for i in xrange(len(test_zo_samples)):
# 		d = dict()
# 		for j in range(10):
# 			d[j] = 0

# 		l.append(d)

# 			# print p
# 			# exit(0)
	



# 	print "gaussian_kernel"
# 	m =  svmutil.svm_train(zo_label,zo_samples,"-t 2 -g 0.05" )	
# 	def get_classifier(i ,j):
# 		file = open(sys.argv[1],"r")
# 		train_dataset = file.read()
# 		train_samples = train_dataset.split("\n")
# 		zero_samples = []
# 		zo_samples = []
# 		one_samples = []
# 		zo_label = [] 
# 		print len(train_samples)
# 		for k in xrange(len(train_samples)-1):
# 			sample = train_samples[k].split(',')
# 			# print sample[-1].type()
# 			# exit(0)
# 			if (sample[-1]== str(j)):
# 				zo_samples.append(sample[:-1])
# 				zo_label.append("1")
# 			if (sample[-1]== str(i)):
# 				zo_samples.append(sample[:-1])
# 				zo_label.append("-1")
		
# 		zo_samples = array((zo_samples)).astype(np.float)
# 		zo_label   = array((zo_label)).astype(np.float)
# 		zo_samples = 1./255*zo_samples
# 		print zo_label
# 		m =  svmutil.svm_train(zo_label,zo_samples,"-t 2 -g 0.05" )	
# 		return m
# 	m = get_classifier(0,1)
# 	classifier = []
# 	classifier.append(m)
# 	p ,acc, d=  svmutil.svm_predict(test_zo_label,test_zo_samples,classifier[0])
# 	print acc

# 	for x in xrange(len(p)):
# 				if (p[x]>0):
# 					l[x][1] += 1
# 					# print j
# 				else:
# 					l[x][0] += 1
# 					# print i

# 	prediction = []
# 	for i in xrange(len(test_zo_samples)):
# 		m = get_max(l[i])
# 		prediction.append(m)

# 	count = 0
# 	for i in xrange(len(test_zo_samples)):

# 		# conf_mat[prediction[i]][int(test_zo_label[i])] += 1
# 		# print prediction[i], test_zo_label[i]
# 		if (int(prediction[i])==int(test_zo_label[i])):
# 			count += 1


# 	# for x in xrange(10):
# 		# print conf_mat[x]
# 	print count , len(test_zo_label)
# 	print "Acc : ",float(count)/len(test_zo_label) * 100
# 	# print acc



# TRAIN FILE	
	

	def get_max(d):
		max = 0
		for x in xrange(1,10):
			if (d[max] < d[x]):
				max = x

		return max


	# j >i
	def get_classifier(i ,j,train_samples):
		
		zero_samples = []
		zo_samples = []
		one_samples = []
		zo_label = [] 
		# print len(train_samples)
		# train_samples = train_samples[:-1]
		# train_samples = train_samples[int(9./10*len(train_samples)):]
		for k in xrange(len(train_samples)):
			sample = train_samples[k].split(',')
			# print sample[-1].type()
			# exit(0)
			if (sample[-1]== str(j)):
				zo_samples.append(sample[:-1])
				zo_label.append("1")
			if (sample[-1]== str(i)):
				zo_samples.append(sample[:-1])
				zo_label.append("-1")
		
		zo_samples = array((zo_samples)).astype(np.float)
		zo_label   = array((zo_label)).astype(np.float)
		zo_samples = 1./255*zo_samples
		# print zo_label
		m =  svmutil.svm_train(zo_label,zo_samples,"-t 2 -g 0.05 -c 1" )	
		return m




	file = open(sys.argv[1],"r")
	train_dataset = file.read()
	train_samples = train_dataset.split("\n")
	# print len(train_samples)
	train_samples = train_samples[:-1]
	# train_samples = train_samples[:int(9./10*len(train_samples))]

	DIGIT =10
		
	classifiers = []
	k = 0
	for i in xrange(DIGIT):
		for j in xrange(i+1,DIGIT):
				classifiers.append(get_classifier(i,j,train_samples))
				k+=1
				# print "classifier ", k


	print "Reading Test File"
# TEST FILE
	file = open(sys.argv[2],"r")
	test_train_dataset = file.read()
	test_train_samples = test_train_dataset.split("\n")
	test_zero_samples = []
	test_zo_samples = []
	test_one_samples = []
	test_zo_label = [] 
	# print len(test_train_samples)
	# v = dict()
	for i in xrange(len(test_train_samples)-1):
		test_sample = test_train_samples[i].split(',')
		
		# if (test_sample[-1]== "1" or test_sample[-1]== "0" or test_sample[-1]== "2"):
		test_zo_samples.append(test_sample[:-1])
		test_zo_label.append(test_sample[-1])
		# if (test_sample[-1]== "0"):
			# test_zo_samples.append(test_sample[:-1])
			# test_zo_label.append(test_sample[-1])
		

	# smaples with label as the digit
	test_zo_samples = array(test_zo_samples).astype(np.float)
	test_zo_label   = array(test_zo_label).astype(np.float)
	test_zo_samples = 1./255*test_zo_samples

	# print len(test_zo_label),len(test_zo_samples)
	
	

	l = []
	for i in xrange(len(test_zo_samples)):
		d = dict()
		for j in range(10):
			d[j] = 0

		l.append(d)

	k=0
	for i in xrange(DIGIT):
		for j in xrange(i+1,DIGIT):

			p ,a, c=  svmutil.svm_predict(test_zo_label,test_zo_samples,classifiers[k])
			
			for x in xrange(len(p)):
				if (p[x]>0):
					l[x][j] += 1
					# print j
				else:
					l[x][i] += 1
					# print i
			k+=1
			# print "Testing Classifier : ", k






	prediction =[]

	for i in xrange(len(test_zo_samples)):
		m = get_max(l[i])
		prediction.append(m)
		
	# print prediction
	conf_mat  = [[0 for x in range(10)] for x in range(10)]			

	count = 0
	for i in xrange(len(test_zo_samples)):

		conf_mat[prediction[i]][int(test_zo_label[i])] += 1
		# print prediction[i], test_zo_label[i]
		if (int(prediction[i])==int(test_zo_label[i])):
			count += 1


	for x in xrange(10):
		print conf_mat[x]

	print "Acc : ",float(count)/len(test_zo_label) * 100
	
# [969, 0, 4, 0, 0, 2, 6, 1, 4, 4]
# [0, 1122, 0, 0, 0, 0, 3, 4, 0, 4]
# [1, 3, 1000, 8, 4, 3, 0, 19, 3, 3]
# [0, 2, 4, 985, 0, 6, 0, 2, 10, 8]
# [0, 0, 2, 0, 962, 1, 4, 4, 3, 13]
# [3, 2, 0, 4, 0, 866, 4, 0, 5, 4]
# [4, 2, 1, 0, 6, 7, 939, 0, 1, 0]
# [1, 0, 6, 6, 0, 1, 0, 987, 3, 9]
# [2, 3, 15, 5, 2, 5, 2, 2, 942, 12]
# [0, 1, 0, 2, 8, 1, 0, 9, 3, 952]
# Acc :  97.24		


elif (sys.argv[3]=="1" and sys.argv[4]=="d"):
	

	# sys.path.append("/home/shashwat/Desktop/COL774/libsvm-3.23/python")
	import svmutil
	import random


	# towritefile = open("14.txt","w")

	def get_max(d):
		max = 0
		for x in xrange(1,10):
			if (d[max] < d[x]):
				max = x

		return max


	# j >i
	def get_classifier(i ,j,train_samples,C):
		
		zero_samples = []
		zo_samples = []
		one_samples = []
		zo_label = [] 
		# SHUFFLE

		random.shuffle(train_samples)
		# print len(train_samples)
		# TAKE 90% To Train	
		for k in xrange((len(train_samples))):
			sample = train_samples[k].split(',')
			

			if (sample[-1]== str(j)):
				zo_samples.append(sample[:-1])
				zo_label.append("1")
			if (sample[-1]== str(i)):
				zo_samples.append(sample[:-1])
				zo_label.append("-1")
		
		zo_samples = array((zo_samples)).astype(np.float)
		zo_label   = array((zo_label)).astype(np.float)
		zo_samples = 1./255*zo_samples
		# print zo_label
		m =  svmutil.svm_train(zo_label,zo_samples,"-t 2 -g 0.05 -c "+str(C) )	
		return m


	file = open(sys.argv[1],"r")
	train_dataset = file.read()
	train_samples = train_dataset.split("\n")
	
	train_samples = train_samples[:-1]
	smpl = len(train_samples)
	
	validation_set = train_samples[int(smpl*9./10):]
	#  TRAINING SET
	train_samples = train_samples[:int(smpl*9./10)]

	DIGIT =10
		
	def give_classifiers_C(C):	
		classifiers = []
		k = 0
		for i in xrange(DIGIT):
			for j in xrange(i+1,DIGIT):
					classifiers.append(get_classifier(i,j,train_samples,C))
					k+=1
					# print "classifier ", k,C

		return classifiers

	C =  [0.00001,0.001,1,5,10]
	#ACC= [71.76,71.76,97.11, , 97.24]

	for cost in xrange(5):
	
		classifiers = give_classifiers_C(C[cost])



		












		print "Reading Test File"
	# TEST FILE
		file = open(sys.argv[2],"r")
		test_train_dataset = file.read()
		test_train_samples = test_train_dataset.split("\n")
		test_zero_samples = []
		test_zo_samples = []
		test_one_samples = []
		test_zo_label = [] 
		# print len(test_train_samples)
		# v = dict()
		for i in xrange(len(test_train_samples)-1):
			test_sample = test_train_samples[i].split(',')
			
			# if (test_sample[-1]== "1" or test_sample[-1]== "0" ):
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append(test_sample[-1])
			# if (test_sample[-1]== "0"):
				# test_zo_samples.append(test_sample[:-1])
				# test_zo_label.append(test_sample[-1])
			

		# smaples with label as the digit
		test_zo_samples = array(test_zo_samples).astype(np.float)
		test_zo_label   = array(test_zo_label).astype(np.float)
		test_zo_samples = 1./255*test_zo_samples

		# print len(test_zo_label),len(test_zo_samples)
		
		

		l = []
		for i in xrange(len(test_zo_samples)):
			d = dict()
			for j in range(10):
				d[j] = 0

			l.append(d)

		k=0
		for i in xrange(DIGIT):
			for j in xrange(i+1,DIGIT):

				p ,a, c=  svmutil.svm_predict(test_zo_label,test_zo_samples,classifiers[k])
				
				for x in xrange(len(p)):
					if (p[x]>0):
						l[x][j] += 1
						# print j
					else:
						l[x][i] += 1
						# print i
				k+=1
				# print "Testing Classifier : ", k






		prediction =[]

		for i in xrange(len(test_zo_samples)):
			m = get_max(l[i])
			prediction.append(m)

		count = 0
		for i in xrange(len(test_zo_samples)):

			if (int(prediction[i])==int(test_zo_label[i])):
				count += 1

		print "Acc : C = "+str(C[cost])+" ",float(count)/len(test_zo_label) * 100
			
#################################################################################
		print "validation starting .. "


		test_zero_samples = []
		test_zo_samples = []
		test_one_samples = []
		test_zo_label = [] 
		
		test_train_samples = validation_set
		print len(test_train_samples)
		# v = dict()

		for i in xrange(len(test_train_samples)):
			test_sample = test_train_samples[i].split(',')
			
			test_zo_samples.append(test_sample[:-1])
			test_zo_label.append(test_sample[-1])
		
			

		# smaples with label as the digit
		test_zo_samples = array(test_zo_samples).astype(np.float)
		test_zo_label   = array(test_zo_label).astype(np.float)
		test_zo_samples = 1./255*test_zo_samples

	
		
		

		l = []
		for i in xrange(len(test_zo_samples)):
			d = dict()
			for j in range(10):
				d[j] = 0

			l.append(d)

		k=0
		for i in xrange(DIGIT):
			for j in xrange(i+1,DIGIT):

				p ,a, c=  svmutil.svm_predict(test_zo_label,test_zo_samples,classifiers[k])
				
				for x in xrange(len(p)):
					if (p[x]>0):
						l[x][j] += 1
						# print j
					else:
						l[x][i] += 1
						# print i
				k+=1
			






		prediction =[]

		for i in xrange(len(test_zo_samples)):
			m = get_max(l[i])
			prediction.append(m)
				

		count = 0
		for i in xrange(len(test_zo_samples)):

			if (int(prediction[i])==int(test_zo_label[i])):
				count += 1


		print "Validation Acc : C = "+str(C[cost])+" ",float(count)/len(test_zo_label) * 100



		
