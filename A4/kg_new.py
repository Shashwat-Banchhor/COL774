from skimage import data, io
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn import svm
import numpy as np
import os , glob 
from random import shuffle
import copy , pickle, random
import cv2

# Make train Samples from
D_COUNT = 5
# Make PCA from
PCA_COUNT = 3000
PCA_count = 0


def get_sequences(TRAIN_SAMPLES,REWARD):
	# TRAIN_SAMPLES = TRAIN_SAMPLES.tolist()

	NEW_SAMPLES = []
	NEW_REWARD = []


	selector = [1,1,1,1,0,0]
	start = 0
	end = 7
	m = len(TRAIN_SAMPLES)
	while(end< m):
		for j in range(start,end):
			if (REWARD[j]==-1):
				start = j+1
				end = start + 7 
		if (end >= m ):
			break


		shuffle(selector)
		sample = []
		for seq in range(start,end-1):
			if (selector[seq -start]==1):
				# print type(TRAIN_SAMPLES[seq])
				sample = sample + TRAIN_SAMPLES[seq]
		sample = sample + TRAIN_SAMPLES[end-1]
		NEW_SAMPLES.append(copy.deepcopy(sample))
		if (REWARD[end]==1):	
			NEW_REWARD.append(1)
		else:
			NEW_REWARD.append(-1)


		start = start +1;
		end   = end +1

	return NEW_SAMPLES , NEW_REWARD	







import os
directory = os.getcwd()
# l =[x[0] for x in os.walk(directory)]

train_directory = glob.glob(directory+"/00*")
train_directory = sorted(train_directory)


################################################# GET THE TOTAL SAMPLE LENGTH
# len_samples = 0;
# for directories in train_directory:
# 	directory_files = sorted(glob.glob(directories+"/*.png")) 
# 	# for file in directory_files:
# 	len_samples += len(directory_files)
# print len_samples
# len_samples = len_samples/10000
#################################################



#################################################
TRAIN_SAMPLES =  []#np.zeros((len_samples,210*160))
REWARD        =  []
sample = 0
dir_count = 0
positive_samples_index = [] 
negative_sample_index = []
prev_sample = 10
prev_pos_sample = 0

for directories in train_directory:
	# print (directories) 

	dir_count += 1
	print (dir_count)
	directory_files = sorted(glob.glob(directories+"/*.png")) 
	
	# REWARD READING 
	count = 0
	rew_file = open(directories+"/rew.csv","r")
	reward = rew_file.read()
	reward = reward.split("\n")


	for file in directory_files:
		if (sample%1000==0):
			print(1,sample)
		if (count==0):
			REWARD.append(-1);

		else:
			REWARD.append(float(reward[count-1]));
			if(float(reward[count-1])>0.0):
				if (sample - prev_pos_sample>8):
					
					positive_samples_index.append(sample);
					prev_pos_sample = sample
			else:
				if (count>10):
					if (sample - prev_sample>8):
						negative_sample_index.append(sample)
						prev_sample = sample


		# print file

		# Image is taken as gray_scale
		# we will apply PCA only on images of first 50 episodes.
		# if (random.randint(1,70)<=2):	
		if(PCA_count<=PCA_COUNT ):
					PCA_count += 1
					if (PCA_count%100==0):	
						print("PCA_count",PCA_count)
					# img = io.imread(file,as_gray = True)#,pilmode = "RGB")
					img = cv2.imread(file,0)
					img = img.reshape(-1)
					# img = np.ndarray.tolist(img)
					TRAIN_SAMPLES.append(img); 
		sample += 1
		count += 1
	



	if (dir_count==D_COUNT):
		break;
	# CODE CHECK
		###########################
			# for x in xrange(len(TRAIN_SAMPLES)):
			# 	print x+1 , REWARD[x]

			# exit(0)

		# if(sample>200):
		# 	break	
			
		##################################



	# if(sample>200):
	# 		break
	# exit(0)
# print l
print ("positive_samples_index" , len(positive_samples_index))

negative_sample_index = negative_sample_index[:3*len(positive_samples_index)]
print ("negative_sample_index"  , len(negative_sample_index))

print (len(TRAIN_SAMPLES) , len(TRAIN_SAMPLES[0]))
# exit(0)
print (type(TRAIN_SAMPLES[0]))
# exit(0)




# print img.shape
# io.imshow(img)

# DUMP THE PCA ------------------------------------------------
# fit the PCA  on 1st 50 episodes and dump it 
pca  = PCA(n_components = 50)
pca.fit(TRAIN_SAMPLES)
# pca_pick = open("pickle_pca.obj","w")
# pickle.dump(pca,pca_pick)
# print ("pca fitted and dumped")
# pca_pick = open("pickle_pca.obj","r")
# pca = pickle.load(pca_pick)
print("pca read")

#----------------------------------------------------------------------


########//////////////////////////////////////

TRAIN_SAMPLES =  []#np.zeros((len_samples,210*160))
REWARD        =  []
sample = 0
dir_count = 0
add_positive_sample = 1
pindex = 0
sequence = []
selector = [1,1,1,1,0,0]
zero_sample_indx = 0





for directories in train_directory:
	print (directories) 

	dir_count += 1
	print ("pos",dir_count)
	directory_files = sorted(glob.glob(directories+"/*.png")) 
	
	# REWARD READING 
	count = 0
	rew_file = open(directories+"/rew.csv","r")
	reward = rew_file.read()
	reward = reward.split("\n")

	sequence = []
	add_positive_sample  =1
	for file in directory_files:
		
		# if (count==0):
		# 	REWARD.append(-1);
		# else:
		# 	REWARD.append(float(reward[count-1]));
			# if(float(reward[count-1])>0.0):
			# 	positive_samples_index.append(sample);


		# print file

		# Image is taken as gray_scale
		# print ("2",sample,add_positive_sample,positive_samples_index[pindex],len(positive_samples_index))
		if (add_positive_sample==1 and (sample>=positive_samples_index[pindex]-7 and sample<positive_samples_index[pindex])):
			
			if (sample %1000==0):	

				print ("+sample" , positive_samples_index[pindex])
			# img = io.imread(file,as_gray = True)#,pilmode = "RGB")
			img = cv2.imread(file,0)

			img = img.reshape(-1)
			img = pca.transform([img])
			# print type(img[0]) , "-----------//////////////---------------"

			# print ("_------------------------------ ", len(img),len(img[0]))
			img = np.ndarray.tolist(img[0])
			sequence.append(copy.deepcopy(img))
			# print (len(sequence))
			if (len(sequence)==7):
				shuffle(selector)
				seq_sample = []
				for x in xrange(6):
					if (selector[x]==1):
						seq_sample = seq_sample+sequence[x]
				seq_sample = seq_sample + sequence[6]
				TRAIN_SAMPLES.append(copy.deepcopy(seq_sample))		
				REWARD.append(1);
				sequence = []
				add_positive_sample = 1
				if (pindex+1 < len(positive_samples_index)):
					# if(positive_samples_index[pindex+1] - positive_samples_index[pindex] >30):
					# 	pindex +=1
					# else:	
					# 	nindex = pindex+1
					# 	while(nindex < len(positive_samples_index) and positive_samples_index[nindex] - positive_samples_index[pindex] <23):
					# 		nindex += 1
					# 	if(nindex==len(positive_samples_index)):
					# 		break
					# 	pindex = nindex
					pindex += 1
				else:
					break;


		elif(add_positive_sample==0):

			# print ("-sample")

			# img = io.imread(file,as_grey = True)#,pilmode = "RGB")
			img = cv2.imread(file,0)

			img = img.reshape(-1)
			img = pca.transform([img])
			img = np.ndarray.tolist(img[0])
			sequence.append(copy.deepcopy(img))

			if (len(sequence)==7):
				shuffle(selector)
				seq_sample = []
				for x in xrange(6):
					if (selector[x]==1):
						seq_sample = seq_sample+sequence[x]
				seq_sample = seq_sample + sequence[6]

				TRAIN_SAMPLES.append(copy.deepcopy(seq_sample))	
				REWARD.append(-1)

				sequence = []
				zero_sample_indx +=1

			if(zero_sample_indx == 3):
				zero_sample_indx = 0
				add_positive_sample = 1 



		 
		sample += 1
		count += 1
	
	if (dir_count==D_COUNT):
		break



# TRAIN_SAMPLES =  []#np.zeros((len_samples,210*160))
# REWARD        =  []
sample = 0
dir_count = 0
# add_positive_sample = 1
print len(negative_sample_index)
pindex = 0
# sequence = []
selector = [1,1,1,1,0,0]
# zero_sample_indx = 0




add_positive_sample  =1
for directories in train_directory:
	print (directories) 

	dir_count += 1
	print ("neg",dir_count)
	directory_files = sorted(glob.glob(directories+"/*.png")) 
	
	# REWARD READING 
	count = 0
	rew_file = open(directories+"/rew.csv","r")
	reward = rew_file.read()
	reward = reward.split("\n")

	sequence = []
	add_positive_sample  =1
	for file in directory_files:
		
		# if (count==0):
		# 	REWARD.append(-1);
		# else:
		# 	REWARD.append(float(reward[count-1]));
			# if(float(reward[count-1])>0.0):
			# 	positive_samples_index.append(sample);


		# print file

		# Image is taken as gray_scale
		if (sample %1000==0):	
			print ("3",sample,add_positive_sample,negative_sample_index[pindex],len(negative_sample_index))
		if (add_positive_sample==1 and (sample>=negative_sample_index[pindex]-7 and sample<negative_sample_index[pindex])):

			# print ("-sample" , negative_sample_index[pindex])
			# img = io.imread(file,as_grey = True)#,pilmode = "RGB")
			img = cv2.imread(file,0)

			img = img.reshape(-1)
			img = pca.transform([img])
			# print type(img[0]) , "-----------//////////////---------------"

			# print ("_------------------------------ ", len(img),len(img[0]))
			img = np.ndarray.tolist(img[0])
			sequence.append(copy.deepcopy(img))
			# print (len(sequence))
			if (len(sequence)==7):
				shuffle(selector)
				seq_sample = []
				for x in xrange(6):
					if (selector[x]==1):
						seq_sample = seq_sample+sequence[x]
				seq_sample = seq_sample + sequence[6]
				TRAIN_SAMPLES.append(copy.deepcopy(seq_sample))		
				REWARD.append(-1);
				sequence = []
				add_positive_sample = 1

				if (pindex+1 < len(negative_sample_index)):
					# if(positive_samples_index[pindex+1] - positive_samples_index[pindex] >30):
					# 	pindex +=1
					# else:	
					# 	nindex = pindex+1
					# 	while(nindex < len(positive_samples_index) and positive_samples_index[nindex] - positive_samples_index[pindex] <23):
					# 		nindex += 1
					# 	if(nindex==len(positive_samples_index)):
					# 		break
					# 	pindex = nindex
					pindex +=1 
				else:
					break;



		# elif(add_positive_sample==0):

		# 	print ("-sample")

		# 	img = io.imread(file,as_gray = True)#,pilmode = "RGB")
		# 	img = img.reshape(-1)
		# 	img = pca.transform([img])
		# 	img = np.ndarray.tolist(img[0])
		# 	sequence.append(copy.deepcopy(img))

		# 	if (len(sequence)==7):
		# 		shuffle(selector)
		# 		seq_sample = []
		# 		for x in xrange(6):
		# 			if (selector[x]==1):
		# 				seq_sample = seq_sample+sequence[x]
		# 		seq_sample = seq_sample + sequence[6]

		# 		TRAIN_SAMPLES.append(copy.deepcopy(seq_sample))	
		# 		REWARD.append(-1)

		# 		sequence = []
		# 		zero_sample_indx +=1

		# 	if(zero_sample_indx == 3):
		# 		zero_sample_indx = 0
		# 		add_positive_sample = 1 



		# img = io.imread(file,as_gray = True)#,pilmode = "RGB")
		# img = img.reshape(-1)
		# img = pca.transform(img)
		# img = np.ndarray.tolist(img)
		# TRAIN_SAMPLES.append(img); 
		sample += 1
		count += 1
	
	if (dir_count==D_COUNT):
		break



#########?////////////////////////////////////

# file_pick = open("pickle_train.obj","w")
# pickle.dump(TRAIN_SAMPLES,file_pick)
# TRAIN_SAMPLES = np.array(TRAIN_SAMPLES)

print ("Train Samples have been dumped +ve -ve segregated")

# red_TRAIN_SAMPLES  = pca.transform(TRAIN_SAMPLES)
print ("New dim" , len(TRAIN_SAMPLES[0]))
# red_TRAIN_SAMPLES = np.ndarray.tolist(red_TRAIN_SAMPLES)
print REWARD
X_t , Y_t = TRAIN_SAMPLES , REWARD # get_sequences(TRAIN_SAMPLES , REWARD)
# print "New dim" , len(X_t[0])  , len(X_t) 


X_train = []
Y_train = []
# p_d= 0
# for i in range(len(Y_t)):
# 	if (Y_t[i]==1):
# 		p_d +=1
# 		Y_train.append(Y_t[i])
# 		X_train.append(X_t[i])


# print ("total +ve samples ",p_d)
# for i in xrange(len(Y_t)):
# 	if (p_d>0 and Y_t[i]==-1):

# 		p_d -=1
# 		Y_train.append(Y_t[i])
# 		X_train.append(X_t[i])


# print ("Final Y_train is " , Y_train , len(Y_train) , len(X_train))

X_train , Y_train = TRAIN_SAMPLES , REWARD
######################  S V M  ##################################

import sys
# sys.path.append("libsvm-3.23/python")
# import svmutil




# print (Y_train)
print ("svm fitting")
clf  = svm.SVC(kernel="linear",random_state=0, tol=1e-5)
clf.fit(X_train, Y_train)  
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# m =  svmutil.svm_train(Y_train,X_train,"-t 0" )	

file_pick = open("pickle_svm.obj","w")
print ("Dumping SVM model")

pickle.dump(TRAIN_SAMPLES,file_pick)
print ("SVM model also dumped..")
# TEST TIME 


# directory = os.getcwd()

# validation_directory = glob.glob(directory+"/validation_dataset/*")
# validation_directory = sorted(validation_directory)

# # print validation_directory , directory
# rew_file = open(directory+"/validation_dataset/rewards.csv","r")
# reward = rew_file.read()
# reward = reward.split("\n")

# VAL_SAMPLES = []
# VAL_REWARD  = []
# count = 0

# for directories in validation_directory:

# 	directory_files = sorted(glob.glob(directories+"/*.png")) 
# 	if(len(directory_files)==0):
# 		break
# 	sequence = []
# 	add_positive_sample  =1
# 	for file in directory_files:
# 			img = io.imread(file,as_gray = True)#,pilmode = "RGB")
# 			img = img.reshape(-1)
# 			img = pca.transform([img])
			
# 			img = np.ndarray.tolist(img[0])
# 			sequence.append(copy.deepcopy(img))
# 			# print (len(sequence))
# 	seq_sample  = []
# 	for x in xrange(5):
# 		# print len(sequence),x, sequence[x]
# 		seq_sample = seq_sample+sequence[x]
# 	VAL_SAMPLES.append(copy.deepcopy(seq_sample))		
# 	VAL_REWARD.append(2*float(reward[count].split(",")[1])-1)
	
# 	count += 1
# 	print "Val count :",count
	# if (count==):
	# 	break

# print VAL_REWARD
# X_test = VAL_SAMPLES
# Y_test = VAL_REWARD

# p ,acc, d=  svmutil.svm_predict(Y_train,X_train,m)
y_train_pred = clf.predict(X_train)
print (f1_score(Y_train,y_train_pred,average="binary"))

# p ,acc, d=  svmutil.svm_predict(Y_test,X_test,m)
# y_test_pred = clf.predict(X_test)
# print (f1_score(Y_test,y_test_pred,average="binary"))






directory = os.getcwd()

validation_directory = glob.glob(directory+"/test_dataset/*")
validation_directory = sorted(validation_directory)

# print validation_directory , directory
# rew_file = open(directory+"/validtion_dataset/rewards.csv","r")
# reward = rew_file.read()
# reward = reward.split("\n")
# TEST_D_COUNT = 277
VAL_SAMPLES = []
# VAL_REWARD  = []
count = 0

for directories in validation_directory:

	directory_files = sorted(glob.glob(directories+"/*.png")) 
	if(len(directory_files)==0):
		break
	sequence = []
	add_positive_sample  =1
	for file in directory_files:
			# img = io.imread(file,as_gray = True)#,pilmode = "RGB")
			img = cv2.imread(file,0)

			img = img.reshape(-1)
			img = pca.transform([img])
			
			img = np.ndarray.tolist(img[0])
			sequence.append(copy.deepcopy(img))
			# print (len(sequence))
	seq_sample  = []
	for x in xrange(5):
		# print len(sequence),x, sequence[x]
		seq_sample = seq_sample+sequence[x]
	VAL_SAMPLES.append(copy.deepcopy(seq_sample))		
	# VAL_REWARD.append(2*float(reward[count].split(",")[1])-1)
	
	count += 1
	if (count%1000==0):	
		print "Test count :",count
	# if (count==TEST_D_COUNT):
		# break

# print VAL_REWARD
X_test = VAL_SAMPLES
# Y_test = VAL_REWARD

y_test_pred = clf.predict(X_test)
f= open("my_prediction.txt","w")
for i in y_test_pred:
	f.write(str(i)+"\n")

f.close()


# print (f1_score(Y_test,y_test_pred,average="binary"))

exit(0)
#############################################


# imgdata.reshape(5,10)
# io.show(imgdata)
# plt.show()


