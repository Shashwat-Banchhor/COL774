import sys
import numpy as np
from numpy import array as arr
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn import tree


def read_data(filename):
	f = open(filename)
	data = f.read()
	data_lines = data.split("\n")
	Samples = []
	labels = []
	for x in xrange(2,len(data_lines)-1):
		
		l = data_lines[x].split(",")
		p = []
		
		for j in xrange(1,len(l)-1):
			p.append(int(l[j]))

		Samples.append(p)

		labels.append([int(l[-1])])

	Samples  = arr(Samples)
	labels = arr(labels)

	return Samples , labels
def prediction(clf_entropy,X,Y):
	pred = clf_entropy.predict(X)
	correct = 0.0
	for i in range(len(X)):
		if (pred[i]==Y[i]):
			correct +=1

	return  correct / len(X)


# f = open("decision_tree_log.txt","w")




X_train,Y_train = read_data(sys.argv[1])
X_val , Y_val= read_data(sys.argv[2])
X_test , Y_test =  read_data(sys.argv[3])

# clf_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=23, min_samples_leaf= 3 ,min_samples_split = 5)
# print prediction(clf_entropy,X_train,Y_train)
if (sys.argv[4]=="4"):
	print sys.argv[4]
	h = 4
	l = 22
	s = 92
	clf_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=h, min_samples_leaf= l ,min_samples_split = s)
	clf_entropy.fit(X_train, Y_train)

	score = prediction(clf_entropy,X_val,Y_val)
	print "Validation ",score
	score = prediction(clf_entropy,X_train,Y_train)
	print "Train",score
	score = prediction(clf_entropy,X_test,Y_test)
	print "Test", score

	# LOG ALL THE SCORES

	# for h in xrange(1,23):
	# 	for l in range(2,100,10):	
	# 		for s in range(2,100,10):
	# 			clf_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=h, min_samples_leaf= l ,min_samples_split = s)
	# 			clf_entropy.fit(X_train, Y_train)

	# 			score = prediction(clf_entropy,X_val,Y_val)
	# 			print h , l,s, score
	# 			f.write(str(h)+","+str(l)+","+str(s)+","+str(score))


elif (sys.argv[4]=="5" or "6"):

	# df = pd.read_csv(sys.argv[1])
	print sys.argv[4]	
	def ohe(X_train):
		Categorical = ["X3","X4","X6","X7","X8","X9","X10","X11"]
		Categorical = [2,3,5,6,7,8,9,10]
		Categorical_data_frames = []
		for category in Categorical:
			# print category
			# print len(X_train[0])
			dummies = pd.get_dummies(X_train[:,category])
			# print type(dummies)
			# merged = pd.concat([X_train,dummies],axis="columns")

			dummies = dummies.values
			dummies = arr(dummies)
			# type(dummies)
			Categorical_data_frames.append(dummies)
			
		
		X_train = np.delete(X_train,Categorical,1)

		for i in range(len(Categorical_data_frames)):
			# print "r",len(Categorical_data_frames[i])
			# print "c",i,len(Categorical_data_frames[i][0])
			X_train = np.append(X_train,Categorical_data_frames[i],axis=1)

		return X_train


	X = np.append(X_train,X_test,axis=0)
	X = np.append(X,X_val,axis=0)
	print len(X)
	# exit(0)



	X = ohe(X)
	X_train = X[:len(X_train),:]
	X_test = X[len(X_train):len(X_train)+len(X_test),:]
	X_val = X[len(X_train)+len(X_test): , :]
	# print len(X_train)
	# print len(X_test)
	# print len(X_val)
	# exit(0)
	# X_test  = ohe(X_test)
	# print len(X_train[0])
	# print len(X_test[0])





	# print X_train[0]
	# X_train = pd.get_dummies(X_train[2])
	# print X_train[0]
	# exit(
	h = 4
	l = 22
	s = 92


	if (sys.argv[4]=="5"):
		clf_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=h, min_samples_leaf= l ,min_samples_split = s)
		clf_entropy.fit(X_train, Y_train)
		# exit(0)
		score = prediction(clf_entropy,X_val,Y_val)
		print "Validation ",score
		score = prediction(clf_entropy,X_train,Y_train)
		print "Train",score
		score = prediction(clf_entropy,X_test,Y_test)
		print "Test", score
	
	else:
		BOOT = [True,False]
		# for trees in range(1,40,2):
		# 	for boot in range(2):
		# 		for n_features in range(20,85,5):
		# 			clf_entropy = RandomForestClassifier(n_estimators=trees, max_depth=h, min_samples_leaf= l ,min_samples_split = s,random_state = 100,bootstrap = BOOT[boot],max_features = n_features)
		# 			clf_entropy.fit(X_train, Y_train)
					
		# 			score = prediction(clf_entropy,X_val,Y_val)
		# 			print "TREES",trees,"BOOTS",BOOT[boot],"N_FTRS",n_features,"ACC",score
			

		trees = 7
		boot = 1
		n_features = 30
		
		clf_entropy = RandomForestClassifier(n_estimators=trees, max_depth=h, min_samples_leaf= l ,min_samples_split = s,random_state = 100,bootstrap = BOOT[boot],max_features = n_features)
		clf_entropy.fit(X_train, Y_train)
		score = prediction(clf_entropy,X_val,Y_val)
		print "Validation ",score
		score = prediction(clf_entropy,X_train,Y_train)
		print "Train",score
		score = prediction(clf_entropy,X_test,Y_test)
		print "Test", score






