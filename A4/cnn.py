from skimage import data, io
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os , glob 
from random import shuffle
import copy , pickle , cv2
from numpy import array
import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# import numpy as np
from keras import optimizers
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
import os
directory = os.getcwd()
# l =[x[0] for x in os.walk(directory)]


#######################################################################
from keras import backend as K


# from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




# def nll3(y_true, y_pred):
#     """ Negative log likelihood. """

#     likelihood = K.tf.distributions.Bernoulli(logits=y_pred)

#     return - K.sum(likelihood.log_prob(y_true), axis=-1)





import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):

	def on_train_begin(self, logs={}):
		 self.val_f1s = []
		 self.val_recalls = []
		 self.val_precisions = []
	 
	def on_epoch_end(self, epoch, logs={}):
		 val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
		 val_targ = self.model.validation_data[1]
		 _val_f1 = f1_score(val_targ, val_predict)
		 _val_recall = recall_score(val_targ, val_predict)
		 _val_precision = precision_score(val_targ, val_predict)
		 self.val_f1s.append(_val_f1)
		 self.val_recalls.append(_val_recall)
		 self.val_precisions.append(_val_precision)
		 print ("-val_f1:",_val_f1, "-val_precision:" , _val_precision,"-val_recall" , _val_recall)
		 return








D_COUNT = 20




# exit(0)

train_directory = glob.glob(directory+"/*")
train_directory = sorted(train_directory)

#################################################
# TRAIN_SAMPLES =  []#np.zeros((len_samples,210*160))
# REWARD        =  []
# sample = 0
# dir_count = 0
# positive_samples_index = [] 
# negative_sample_index = []
# prev_sample = 10
# # pos_sample = 0
# prev_pos_sample = 0
# # psample_sz= 0
# for directories in train_directory:
# 	print (directories) 

# 	dir_count += 1
# 	print (dir_count)
# 	directory_files = sorted(glob.glob(directories+"/00*.png")) 
	
# 	# REWARD READING 
# 	count = 0
# 	rew_file = open(directories+"/rew.csv","r")
# 	reward = rew_file.read()
# 	reward = reward.split("\n")


# 	for file in directory_files:
		
# 		# print(1,sample)
# 		if (count==0):
# 			REWARD.append(-1);

# 		else:
# 			REWARD.append(float(reward[count-1]));
# 			if(float(reward[count-1])>0.0):
# 				if (sample-prev_pos_sample > 7):	
# 					positive_samples_index.append(sample);
# 					img = io.imread(file)#,pilmode = "RGB")
# 					img = img.reshape(-1)
# 					# img = np.ndarray.tolist(img)
# 					TRAIN_SAMPLES.append(img); 
# 					prev_pos_sample = sample
# 			else:
# 				if (count>10):
# 					if (sample - prev_sample>8):
# 						negative_sample_index.append(sample)
# 						img = io.imread(file)#,pilmode = "RGB")
# 						img = img.reshape(-1)
# 						# img = np.ndarray.tolist(img)
# 						TRAIN_SAMPLES.append(img); 
# 						prev_sample = sample


# 		# print file

# 		# Image is taken as gray_scale
# 		# we will apply PCA only on images of first 50 episodes.
# 		# if(dir_count<=1):
# 		# 	img = io.imread(file)#,pilmode = "RGB")
# 		# 	img = img.reshape(-1)
# 		# 	# img = np.ndarray.tolist(img)
# 		# 	TRAIN_SAMPLES.append(img); 
# 		sample += 1
# 		count += 1
	
		

# 	if (dir_count==D_COUNT):
# 		break;
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


		
		sample += 1
		count += 1
	



	if (dir_count==D_COUNT):
		break;














print ("P","positive_samples_index" , len(positive_samples_index))
print (positive_samples_index)

negative_sample_index = negative_sample_index[:3*len(positive_samples_index)]
print ("N","negative_sample_index"  , len(negative_sample_index))
print (negative_sample_index)

# print (len(TRAIN_SAMPLES) len(TRAIN_SAMPLES[0]))
# exit(0)
# print (type(TRAIN_SAMPLES[0
# exit(0)
# file_pick = open("pickle_train.obj","w")
# pickle.dump(TRAIN_SAMPLES,file_pick)
# TRAIN_SAMPLES = np.array(TRAIN_SAMPLES)



# ########//////////////////////////////////////

# TRAIN_SAMPLES =  []#np.zeros((len_samples,210*160))
# REWARD        =  []
# sample = 0
# dir_count = 0
# add_positive_sample = 1
# pindex = 0
# sequence = []
# selector = [1,1,1,1,0,0]
# zero_sample_indx = 0





# for directories in train_directory:
# 	print (directories) 

# 	dir_count += 1
# 	print (dir_count)
# 	directory_files = sorted(glob.glob(directories+"/*.png")) 
	
# 	# REWARD READING 
# 	count = 0
# 	rew_file = open(directories+"/rew.csv","r")
# 	reward = rew_file.read()
# 	reward = reward.split("\n")

# 	sequence = []
# 	add_positive_sample  =1
# 	for file in directory_files:
		
# 		if (sample%1000==0):
# 			print ("2",sample,add_positive_sample,positive_samples_index[pindex],len(positive_samples_index))
# 		if (add_positive_sample==1 and (sample>=positive_samples_index[pindex]-7 and sample<positive_samples_index[pindex])):

# 			# print ("+sample" , positive_samples_index[pindex])
# 			img = io.imread(file,as_gray = False)#,pilmode = "RGB")
# 			img = img.reshape(-1)
# 			# img = pca.transform([img])
# 			# print type(img[0]) , "-----------//////////////---------------"

# 			# print ("_------------------------------ ", len(img),len(img[0]))
# 			img = np.ndarray.tolist(img)
# 			sequence.append(copy.deepcopy(img))
# 			# print (len(sequence))
# 			if (len(sequence)==7):
# 				shuffle(selector)
# 				seq_sample = []
# 				for x in xrange(6):
# 					if (selector[x]==1):
# 						seq_sample = seq_sample+sequence[x]
# 				seq_sample = seq_sample + sequence[6]
# 				print ("+sample" , pindex,positive_samples_index[pindex])

# 				TRAIN_SAMPLES.append(copy.deepcopy(seq_sample))	
# 				print ("L",len(TRAIN_SAMPLES))	

# 				arr = [0.0,1.0]	
# 				REWARD.append(1.0);
# 				sequence = []
# 				add_positive_sample = 1
# 				if (pindex+1 < len(positive_samples_index)):
# 					# if(positive_samples_index[pindex+1] - positive_samples_index[pindex] >30):
# 					# 	pindex +=1
# 					# else:	
# 					# 	nindex = pindex+1
# 					# 	while(nindex < len(positive_samples_index) and positive_samples_index[nindex] - positive_samples_index[pindex] <23):
# 					# 		nindex += 1
# 					# 	if(nindex==len(positive_samples_index)):
# 					# 		break
# 					# 	pindex = nindex
# 					pindex += 1
# 				else:
# 					break;


# 		elif(add_positive_sample==0):

# 			print ("-sample")

# 			img = io.imread(file,as_gray = False)#,pilmode = "RGB")
# 			img = img.reshape(-1)
# 			# img = pca.transform([img])
# 			img = np.ndarray.tolist(img)
# 			sequence.append(copy.deepcopy(img))

# 			if (len(sequence)==7):
# 				shuffle(selector)
# 				seq_sample = []
# 				for x in xrange(6):
# 					if (selector[x]==1):
# 						seq_sample = seq_sample+sequence[x]
# 				seq_sample = seq_sample + sequence[6]

# 				TRAIN_SAMPLES.append(copy.deepcopy(seq_sample))	

# 				REWARD.append(0.0)

# 				sequence = []
# 				zero_sample_indx +=1

# 			if(zero_sample_indx == 3):
# 				zero_sample_indx = 0
# 				add_positive_sample = 1 


# 		sample += 1
# 		count += 1
	
# 	if (dir_count==D_COUNT):
# 		break


# # exit(0)

# # TRAIN_SAMPLES =  []#np.zeros((len_samples,210*160))
# # REWARD        =  []
# sample = 0
# dir_count = 0
# # add_positive_sample = 1
# print ("Len NegIndx: ",len(negative_sample_index))
# pindex = 0
# # sequence = []
# selector = [1,1,1,1,0,0]
# # zero_sample_indx = 0




# add_positive_sample  =1
# for directories in train_directory:
# 	print (directories) 

# 	dir_count += 1
# 	print ("neg",dir_count)
# 	directory_files = sorted(glob.glob(directories+"/*.png")) 
	
# 	# REWARD READING 
# 	count = 0
# 	rew_file = open(directories+"/rew.csv","r")
# 	reward = rew_file.read()
# 	reward = reward.split("\n")

# 	sequence = []
# 	add_positive_sample  =1
# 	for file in directory_files:
		
# 		# if (count==0):
# 		# 	REWARD.append(-1);
# 		# else:
# 		# 	REWARD.append(float(reward[count-1]));
# 			# if(float(reward[count-1])>0.0):
# 			# 	positive_samples_index.append(sample);


# 		# print file

# 		# Image is taken as gray_scale
# 		# print ("3",sample,add_positive_sample,negative_sample_index[pindex],len(negative_sample_index))
# 		if(sample%1000==0):
# 			print ("Neg Sample :",sample)
# 		if (add_positive_sample==1 and (sample>=negative_sample_index[pindex]-7 and sample<negative_sample_index[pindex])):

# 			# print ("-sample" , negative_sample_index[pindex])
# 			img = io.imread(file,as_gray = False)#,pilmode = "RGB")
# 			img = img.reshape(-1)
# 			# img = pca.transform([img])
# 			# print type(img[0]) , "-----------//////////////---------------"

# 			# print ("_------------------------------ ", len(img),len(img[0]))
# 			img = np.ndarray.tolist(img)
# 			sequence.append(copy.deepcopy(img))
# 			# print (len(sequence))
# 			if (len(sequence)==7):
# 				shuffle(selector)
# 				seq_sample = []
# 				for x in xrange(6):
# 					if (selector[x]==1):
# 						seq_sample = seq_sample+sequence[x]
# 				seq_sample = seq_sample + sequence[6]
# 				print ("-sample" , pindex,negative_sample_index[pindex])

# 				TRAIN_SAMPLES.append(copy.deepcopy(seq_sample))	
# 				print (len(TRAIN_SAMPLES))	
# 				arr = [1.0,0.0]
# 				REWARD.append(0.0);
# 				sequence = []
# 				add_positive_sample = 1

# 				if (pindex+1 < len(negative_sample_index)):
# 					# if(positive_samples_index[pindex+1] - positive_samples_index[pindex] >30):
# 					# 	pindex +=1
# 					# else:	
# 					# 	nindex = pindex+1
# 					# 	while(nindex < len(positive_samples_index) and positive_samples_index[nindex] - positive_samples_index[pindex] <23):
# 					# 		nindex += 1
# 					# 	if(nindex==len(positive_samples_index)):
# 					# 		break
# 					# 	pindex = nindex
# 					pindex +=1 
# 				else:
# 					break;


# 		sample += 1
# 		count += 1
	
# 	if (dir_count==D_COUNT):
# 		break



# #########?////////////////////////////////////


# print ("New dim sample 0 " , len(TRAIN_SAMPLES[0]),"samples" , len(TRAIN_SAMPLES))
# print REWARD



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
			img = cv2.imread(file,1)

			img = img.reshape(-1)
			# img = pca.transform([img])
			# print type(img[0]) , "-----------//////////////---------------"

			# print ("_------------------------------ ", len(img),len(img[0]))
			img = np.ndarray.tolist(img)
			sequence.append(copy.deepcopy(img))
			if(sample<20):
				print ("LS",len(sequence[0]))
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
			img = cv2.imread(file,1)

			img = img.reshape(-1)
			# img = pca.transform([img])
			img = np.ndarray.tolist(img)
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
			img = cv2.imread(file,1)

			img = img.reshape(-1)
			# img = pca.transform([img])
			# print type(img[0]) , "-----------//////////////---------------"

			# print ("_------------------------------ ", len(img),len(img[0]))
			img = np.ndarray.tolist(img)
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
				REWARD.append(0);
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
print ("Len Samples", len(TRAIN_SAMPLES))


print (REWARD)
X_train = []
Y_train = []
# print ("Dumping X_train...")
# file_pick = open("cnn_pickle_trainX.obj","wb")
# pickle.dump(TRAIN_SAMPLES,file_pick)
# file_pick.close()
# print ("Dumping Y_train...")

# file_pick = open("cnn_pickle_trainY.obj","wb")
# pickle.dump(REWARD,file_pick)
# file_pick.close()
# print ("Dumping Done")
X_train , Y_train = TRAIN_SAMPLES , REWARD
# exit(0)
# ######################  C N N  ##################################



# batch_size = 20
# epochs  = 100
# num_classes = 2

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
# 				 strides = 2,
#                  activation='relu',
#                  input_shape=(210,160,15)))

# model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

# model.add(Conv2D(64, kernel_size = (3, 3),
# 				strides = 2,
# 				activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
# model.add(Dense(2048, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))


######################  C N N  ##################################




batch_size = 250
epochs  = 150
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
				 strides = 2,
                 activation='relu',
                 input_shape=(210,160,15)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(64, kernel_size = (3, 3),
				strides = 2,
				activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation ='softmax'))


model.summary()

sgd = optimizers.SGD(lr=0.00001) #decay=1e-6, momentum=0.9, nesterov=True)



 
metrics = Metrics()



model.compile(loss="binary_crossentropy",
              optimizer=sgd,
              callbacks=[metrics],metrics=['accuracy',f1_m,recall_m,precision_m])

print  ("model compiled")
# shuffle(X_train)
x_train = X_train#[:int(0.8*len(X_train))]
# x_test = X_train[int(0.8*len(X_train)):]
x_train = array(x_train)
# shuffle(x_train)
x_train = x_train.reshape(-1,210,160,15)
Y_train = to_categorical(Y_train)

y_train = Y_train#[:int(0.8*len(X_train))]
# y_test =  Y_train[int(0.8*len(X_train)):]

# print (y_train)
# print (y_test) 

# y_train = to_categorical(y_train)
# y_train = array(y_train).reshape(-1,2)

# y_test = to_categorical(y_test)
# y_test = array(y_test).reshape(-1,2)
# print ("----------------------y_train,", y_train[0])
# x_test = array(x_test)

# x_test =x_test.reshape(-1,210,160,15)


print ("X_T",x_train.shape ,"Y_T", y_train.shape, "\nGathering Validation") 

directory = os.getcwd()

validation_directory = glob.glob(directory+"/validation_dataset/*")
validation_directory = sorted(validation_directory)



VAL_SAMPLES = []
VAL_REWARD  = []
count = 0

rew_file = open(directory+"/validation_dataset/rewards.csv","r")
reward = rew_file.read()
reward = reward.split("\n")

for directories in validation_directory:

	directory_files = sorted(glob.glob(directories+"/*.png")) 
	if(len(directory_files)==0):
		break
	sequence = []
	add_positive_sample  =1
	for file in directory_files:
			img = io.imread(file)#,pilmode = "RGB")
			img = img.reshape(-1)
			# img = pca.transform([img])
			
			img = np.ndarray.tolist(img)
			sequence.append(copy.deepcopy(img))
			# print (len(sequence))
	seq_sample  = []
	for x in xrange(5):
		# print len(sequence),x, sequence[x]
		seq_sample = seq_sample+sequence[x]
	seq_sample = array(seq_sample).reshape(210,160,15)
	VAL_SAMPLES.append(copy.deepcopy(seq_sample))		
	VAL_REWARD.append(2*float(reward[count].split(",")[1])-1)
	
	count += 1
	if (count%50==0):	
		print "Val count :",count
	# if (count==111):
	# 	break

# VAL_REWARD = to_categorical(VAL_REWARD)

# score = model.evaluate(VAL_SAMPLES, VAL_REWARD, verbose=0)
# print('VAL loss:', score[0])
# print('VAL accuracy:', score[1])
VAL_SAMPLES = array(VAL_SAMPLES).reshape(-1,210,160,15)

VAL_REWARD = to_categorical(VAL_REWARD)

print ("Fit model")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,validation_data = (VAL_SAMPLES,VAL_REWARD),
          verbose=1
         )


# TRIAN EVALUTRAIN

# score = model.evaluate(x_train, y_train, verbose=0)
# print('TRAIN loss:', score[0])
# print('TRAIN accuracy:', score[1])

# score = model.evaluate(x_test, y_test, verbose=0)
# print('VAL loss:', score[0])
# print('VAL accuracy:', score[1])

# pred_train = model.predict(x_train)
# file_pick = open("train_pred.obj","w")
# pickle.dump(pred_train,file_pick)

# pred_test = model.predict(x_test)
# print ("Train: ",f1_score(y_train,pred_train,average="binary"))
# print ("VAL: ",f1_score(y_test,pred_test,average="binary"))







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
# 			# img = pca.transform([img])
			
# 			img = np.ndarray.tolist(img[0])
# 			sequence.append(copy.deepcopy(img))
# 			print (len(sequence))
# 	seq_sample  = []
# 	for x in xrange(5):
# 		print len(sequence),x, sequence[x]
# 		seq_sample = seq_sample+sequence[x]
# 	seq_sample = array(seq_sample).reshape(210,160,15)
# 	VAL_SAMPLES.append(copy.deepcopy(seq_sample))		
# 	VAL_REWARD.append(2*float(reward[count].split(",")[1])-1)
	
# 	count += 1
# 	if (count%500==0):	
# 		print "Val count :",count
# 	# if (count==):
# 	# 	break

# VAL_REWARD = to_categorical(VAL_REWARD)

# score = model.evaluate(VAL_SAMPLES, VAL_REWARD, verbose=0)
# print('VAL loss:', score[0])
# print('VAL accuracy:', score[1])

# pred_val = model.predict(VAL_SAMPLES)
# print ("VAL: ",f1_score(VAL_REWARD,pred_val,average="binary"))

directory = os.getcwd()

validation_directory = glob.glob(directory+"/test_dataset/*")
validation_directory = sorted(validation_directory)

# print validation_directory , directory
# rew_file = open(directory+"/validation_dataset/rewards.csv","r")
# reward = rew_file.read()
# reward = reward.split("\n")

VAL_SAMPLES = []
VAL_REWARD  = []
count = 0

for directories in validation_directory:

	directory_files = sorted(glob.glob(directories+"/*.png")) 
	if(len(directory_files)==0):
		break
	sequence = []
	add_positive_sample  =1
	for file in directory_files:
			img = io.imread(file)#,pilmode = "RGB")
			img = img.reshape(-1)
			# img = pca.transform([img])
			
			img = np.ndarray.tolist(img)
			sequence.append(copy.deepcopy(img))
			# print (len(sequence))
	seq_sample  = []
	for x in xrange(5):
		# print len(sequence),x, sequence[x]
		seq_sample = seq_sample+sequence[x]
	seq_sample = array(seq_sample).reshape(210,160,15)
	VAL_SAMPLES.append(copy.deepcopy(seq_sample))		
	# VAL_REWARD.append(2*float(reward[count].split(",")[1])-1)
	
	count += 1
	if (count%50==0):	
		print "Test count :",count
	# if (count==111):
	# 	break

# VAL_REWARD = to_categorical(VAL_REWARD)

# score = model.evaluate(VAL_SAMPLES, VAL_REWARD, verbose=0)
# print('VAL loss:', score[0])
# print('VAL accuracy:', score[1])
VAL_SAMPLES = array(VAL_SAMPLES).reshape(-1,210,160,15)
pred_val = model.predict(VAL_SAMPLES)
prediction=[]
for x in xrange(len(pred_val)):
	print (pred_val[x])
	if (pred_val[x][0]>pred_val[x][1]):
		prediction.append(0)
	else:
		prediction.append(1)
# import numpy
pred_val = np.asarray(prediction)
np.savetxt("cnn_model_prediction.csv", pred_val, delimiter=",")
print("Finished")
# print ("VAL: ",f1_score(VAL_REWARD,pred_val,average="binary"))



exit(0)


