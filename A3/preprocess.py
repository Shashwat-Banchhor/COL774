import sys
import numpy as np
from numpy import array as arr
import math
import copy


def MakeSample(filename):	
	data = filename.read()
	data_lines = data.split("\n")
	Samples = []
	# f = open("pre_processed_data.csv","w")



	con_val = [1,5,12,13,14,15,16,17,18,19,20,21,22,23]


	for x in xrange(2,len(data_lines)-1):
		
		l = data_lines[x].split(",")
		p = []
		
		for j in xrange(len(l)):
			p.append(int(l[j]))

		Samples.append(p)

	Samples  = arr(Samples)
	count  = 0
	# MEDIAN = []
	# print	len(Samples)
	for x in con_val :
		# median = np.median(Samples[:,x])
		# MEDIAN.append(median)
		# print x , ":", median
		for i in xrange(len(Samples)):
			# print i , x ,count
			if (Samples[i][x] > MEDIAN[count]):
				Samples[i][x] = 1
			else :
				Samples[i][x] = 0
		count += 1

	return Samples

def find_max(l):
	max = -100
	if (l[i]>max):
		max = l[i]
	return max

def Acc_Node(Data):
	Bclass = [0.0,0.0]
	for sample in Data:

		if (sample[-1]==0):
			Bclass[0] += 1
		
		else:
			Bclass[1] += 1
	
	if (Bclass[0] >= Bclass[1]):
		return (Bclass[0])#/(Bclass[0]+Bclass[1])
	else:
		return (Bclass[1])#/(Bclass[0]+Bclass[1])


def majority_Class(Data):
	Bclass = [0.0,0.0]
	for sample in Data:

		if (sample[-1]==0):
			Bclass[0] += 1
		
		else:
			Bclass[1] += 1
	
	if (Bclass[0] >= Bclass[1]):
		return 0
	else:
		return 1

class Node:

	# self.children = []
	# self.data = []
	# self.variable = None
	# self.value = None
	# self. parent = None

	"""docstring for Node"""
	def __init__(self, data=None, variable=None , value=None,children = [],height = 0,already_split=[],predict = None,parent = None,count = 0):
		
		self.data = data
		self.variable = variable
		self.value = value
		self.children = children
		self.parent = parent
		self.height = height
		self.predict = majority_Class(data)
		self.count = count
		self.already_split = already_split


	def set_Parent(self, P):
		self.parent = P

	def add_child(self, C):
		if (self.children==None):
			self.children = []
		self.children.append(C)


# Assuming it is 0 , 1 Classification
def Entropy(Data):

	Bclass = [0.0,0.0]
	for sample in Data:

		if (sample[-1]==0):
			Bclass[0] += 1
		
		else:
			Bclass[1] += 1
	
	# print Bclass

	e = 0.0
	for i in xrange(0,2):
		
		if (Bclass[i]!=0):	
			e -= Bclass[i]/len(Data) * math.log(Bclass[i]/len(Data),2)

	return	e


def split(v,Samples):

		v_types = []
		v_count = dict()

		for i in xrange(len(Samples)):
			if (Samples[i][v] not in v_types):
				v_types.append(Samples[i][v])

		# print v, ":", v_types

		# Initialize the v_count 
		for j in v_types:
			v_count[j] = []

		
		for i in xrange(len(Samples)):
			
			v_count[Samples[i][v]].append(Samples[i])					

		s = 0
		for i in v_types:
			s += len(v_count[i])

		return v_types , v_count , s


def chooseBestVarToSplit(Samples,AlreadySplit):

	# for all variables do

	Entropies = [0]

	for v in xrange(1,len(Samples[0])-1):

		
		# v_vals  v_dictionary
		v_types , v_count , s = split(v,Samples)
			# print "		", i ,":",len(v_count[i])
		# print "		 SUM : ", s
		s = float(s)
		

		entropy = 0.0

		for i in v_types:
			entropy += (len(v_count[i])/s)*Entropy(v_count[i])

		if (len(v_types)==1):
			entropy = 100.0
		Entropies.append(entropy)
		# print v , ":",entropy
		# print Entropies

	# Split a variable iff it hasnt been splited before
	for var in AlreadySplit:
		Entropies[var] = 100.0

	# for var in range(1,len(Entropies)):
		# if (v_)

	# print AlreadySplit , Entropies
	m_ind  = 0
	m = 100
	for i in range(1,len(Entropies)):
		
		if (Entropies[i]<m):
			m = Entropies[i]
			m_ind = i
	

	return m_ind 
	# print  

count = 0


def predict(sample , root):
	node   = root
	assign = False

	while (True):
		var  = node.variable
		assign = False
		for i in range(len(node.children)):
		
			if (node.children[i].value == sample[var]):
				node =  node.children[i]
				assign = True
				break

		if (assign == False):
			return majority_Class(node.data)


# Make predictions on the given samples 
def prediction(Samples,root):
	correct = 0
	Samples = np.ndarray.tolist(Samples)
	for i in range(len(Samples)):
		
		p  = predict(Samples[i],root)
		if (p==Samples[i][-1]):
		
			correct += 1

	return float(correct)/len(Samples)


def predict_all(f,ROOT):
	print "Predicting All..."
	score_train = prediction(Samples_train,ROOT)
	print "Done"
	score_test  = prediction(Samples_test , ROOT)
	print "Done"
	score_val   = prediction(Samples_val, ROOT)
	f.write(str(count)+","+str(score_train)+","+str(score_test)+","+str(score_val)+"\n")
	f.flush()
	print str(count)+","+str(score_train)+","+str(score_test)+","+str(score_val)

def split_node(root):
	
	global count
	count += 1
	# if (count%250==0 and count > 900):
	# 	predict_all(f,ROOT)
	InternalNode = root
	Samples = root.data
	AlreadySplit = root.already_split
	if (root.parent!=None):
		# print "InSN",len(root.parent.children) , len(InternalNode.children) , "HT ", root.height
		pass
	
	if ( root.height >= 23):
		count -=1
		return root
	# else:
		# count += 1
	

	v_index  = chooseBestVarToSplit(Samples,AlreadySplit)
	if (v_index	==0):
		return root
	root.already_split.append(v_index)
	# print v_index
	v_types , v_count , s = split(v_index,Samples)
	if (root.parent!=None):
		# print "InSN1",len(root.parent.children) , len(InternalNode.children)
		pass

	InternalNode.variable = v_index
	for x in xrange(len(v_types)):
		l = copy.deepcopy(root.already_split)
		C = Node(v_count[v_types[x]],v_index,v_types[x],[],InternalNode.height+1,l)
		# print "New Node ", len(C.childsren)
		C.set_Parent(InternalNode)
		C.count = count
		InternalNode.add_child(C)

	if (root.parent!=None):
		# print "InSN2",len(root.parent.children), len(InternalNode.children)
		pass


	for child in InternalNode.children:
		child = split_node(child)
		# break

	return root


#// if no child with var type acts as leaf and returns the majority class





def PruneSubtree(root):
	node = root
	
	acc_node = Acc_Node(node.data)
	acc_child = 0.0
	if (node.children ==[]):
		return node, False
	
	for i in xrange(len(node.children)):
		acc_child += Acc_Node(node.children[i].data)
	
	if (acc_node >= acc_child ):
		node.children = []
		return node , True

	for x in xrange(len(node.children)):
		node.children[i] , ret  = PruneSubtree(node.children[i])
		if (ret == True ):
			return node , True
	
	return node , False


def visit(Samples_val,copy_tree_node,best_acc,best_tree,copy_tree):
	
	# if (len(copy_tree_node.data)<=10):
	# 	# keep a pointer to the parent
	# 	father = copy_tree_node.parent
		
	# 	#index of child to prunce
	# 	prune_child_index  = father.children.index(copy_tree_node)
		
	# 	# store the pruned the node
	# 	child_pruned = father.children.pop(prune_child_index)
	# 	return best_acc

	if (copy_tree_node.height >2):
		return best_acc , best_tree
	# keep a pointer to the parent
	father = copy_tree_node.parent
	
	#index of child to prunce
	prune_child_index  = father.children.index(copy_tree_node)
	
	# store the pruned the node
	child_pruned = father.children.pop(prune_child_index)
	
	# get accuracy on Validation Set
	curr_acc = prediction(Samples_val,copy_tree)

	# update if necessary
	print "		New acc" , curr_acc , "No. ",child_pruned.count,"ht ", child_pruned.height ,"Size", len(child_pruned.data)
	if (curr_acc>best_acc):
		print curr_acc , "Increament:",curr_acc-best_acc
		best_tree = copy.deepcopy(copy_tree)
		best_acc = curr_acc
		print "Updated_best_acc", best_acc
	# put the node back in the tree
	father.children.insert(prune_child_index,child_pruned)
	# Update its parent
	father.children[prune_child_index].parent = father

	#We now have the copy_tree we started with
	# Continue pruning beneath
	for child_no in range(len(copy_tree_node.children)):
		best_acc,best_tree = visit(Samples_val,copy_tree_node.children[child_no],best_acc,best_tree,copy_tree)


	return best_acc,best_tree

def chooseBestVarToPrune(Samples_val,root):
	best_tree = copy.deepcopy(root)
	best_acc  = prediction(Samples_val,root)
	print "Original Best Acc", best_acc
	copy_tree = copy.deepcopy(root)


	for child_no in range(len(copy_tree.children)):
		best_acc,best_tree = visit(Samples_val,copy_tree.children[child_no],best_acc,best_tree,copy_tree)


	return best_tree

	# # parent = co
	# child = copy_tree.children.pop(root.children.index(curr_best))
	# print len(root.children)
	# print len(copy_tree.children)
	
	# copy_tree_node  = copy_tree.children[0]
	# father = copy_tree_node.parent
	# #indec of child to prunce
	# prune_child_index  = father.children.index(copy_tree_node)
	
	# # prune the node
	# child_pruned = father.children.pop(prune_child_index)
	# print "After Prune",len(copy_tree.children)

	# # get accuracy on Validation Set
	# curr_acc = prediction(Samples,copy_tree)
	# # update if necessary
	
	# father.children.insert(prune_child_index,child_pruned)
	# print "After insert",len(copy_tree.children)



	# exit(0)

	
def Prune(Samples_val,root):
	T = root
	
	while (True):

		oldAcc = prediction(Samples_val,T)
		print "oldAcc" , oldAcc
		# n = BestNodeToPrune(T)
		# T, Var  = PruneSubtree(T)
		T_prime= chooseBestVarToPrune(Samples_val,T)


		newAcc = prediction(Samples_val,T_prime)
		print "Size OF UPDATED TREE", get_size(T_prime)	
		print "newAcc val",newAcc,"Diff", newAcc - oldAcc
		newAcc_tra = prediction(Samples_train,T_prime)

		newAcc_test = prediction(Samples_test,T_prime)
		print "Tr : ", newAcc_tra , "Te",newAcc_test

		if (newAcc <= oldAcc):
			break
		T = copy.deepcopy(T_prime)

	return T
def get_size(root):
	# global n_c
	n_c  = 1
	for i in xrange(len(root.children)):
		n_c +=  get_size(root.children[i])

	return n_c 


#PREPROCESSING
###################################################################################################
filename = open(sys.argv[1])
data = filename.read()
data_lines = data.split("\n")
Samples = []
f = open("pre_processed_data.csv","w")


# print "Hai"
con_val = [1,5,12,13,14,15,16,17,18,19,20,21,22,23]


for x in xrange(2,len(data_lines)-1):
	
	l = data_lines[x].split(",")
	p = []
	
	for j in xrange(len(l)):
		p.append(int(l[j]))

	Samples.append(p)

Samples  = arr(Samples)
count  = 0
MEDIAN = []
for x in con_val :
	median = np.median(Samples[:,x])
	MEDIAN.append(median)
	# print x , ":", median
	for i in xrange(len(Samples)):
		
		if (Samples[i][x] > MEDIAN[count]):
			Samples[i][x] = 1
		else :
			Samples[i][x] = 0
	count += 1


for x in xrange(len(Samples)):
	p =  []
	for j in xrange(len(Samples[x])):
		p.append(str(Samples[x][j]))

	f.write(",".join(p) + "\n")


# print len(Samples)
f.close()
# print Samples[0]
Samples_train = Samples

# exit(0)
###################################################################################################
# print Samples[0]

filename = open(sys.argv[2])
Samples_val= MakeSample(filename)#np.ndarray.tolist(Samples)
filename = open(sys.argv[3])
Samples_test = MakeSample(filename)#np.ndarray.tolist(Samples)

f = open("decision_graph_log.txt","a")

count = 0
root = Node(Samples_train)
ROOT = root
split_node(root)
print "Total_Nodes",count 
n_count = get_size(root)
print "Get Size", n_count 

# ///////// TRAIN CHECK


# Train Prediction
# print prediction(Samples,root)
# f = open("ACCURACIES.txt","a")

t = prediction(Samples_train,root)
print "Train Acc",  t
v = prediction(Samples_val , root)
print "Val Acc" ,  v
# root  = Prune(Samples_val,root)
# TEST TIME
test = prediction(Samples_test,root)
print "Test  Acc" ,  test
# f.write(str(count)+","+str(t)+","+str(v)+","+str(test)+"\n")
# f.flush()
# f.close()

if (sys.argv[4]=="2"):
	Prune(Samples_val,root)






