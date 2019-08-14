import sys
import numpy as np
from numpy import array as arr
import os
def read_file(file_train):
	


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
	out = len(Samples[0]) -1
	V_TYPES =[]

	# Assumming only 10 datasets
	for v in xrange(10):
		v_types = []
		for x in xrange(len(Samples)):
			if (Samples[x][v] not in v_types):
				v_types.append(Samples[x][v])
			
		V_TYPES.append(v_types)

	for x in xrange(len(V_TYPES)):
		print V_TYPES[x]


	# print NEW_SAMPLES[0]

	sum = 0
	prefix_sum = []
	for x in xrange(len(V_TYPES)):
		prefix_sum.append(sum)

		sum += len(V_TYPES[x])
	print "PS",prefix_sum
	print sum
	NEW_SAMPLES = [[0 for x in range(sum)] for x in range(len(Samples))]
	print  len(NEW_SAMPLES[0])

	for i in xrange(len(Samples)):
		for j in xrange(out):
			# if (j%2==0):
			# 	NEW_SAMPLES[i][j/2*17+ (Samples[i][j] -1) ] = 1
			# else :
			# 	NEW_SAMPLES[i][j/2*17 + 4 + (Samples[i][j] -1)] = 1
			# print prefix_sum[j]# +  (Samples[i][j] -1)
			NEW_SAMPLES[i][prefix_sum[j] +  (Samples[i][j] -1)] = 1

	Y =  [[0.0 for j in range(out)]]
	Y = []
	for sample in range(len(Samples)):
		l = [0.0 for j in range(out)]
		l[Samples[sample][-1]] = 1.
		Y.append(l)

	for sample in xrange(len(NEW_SAMPLES)):
	 	NEW_SAMPLES[sample].append(Samples[sample][-1])

	return arr(NEW_SAMPLES) , Y


X , Y = read_file(sys.argv[1])
file = sys.argv[3]
os.path.join(file)
f = open(file,"w")
X = X.astype(np.str)
for sample in range(len(X)):
	l = ",".join(X[sample])
	f.write(l+"\n")
f.close()


X , Y = read_file(sys.argv[2])
file = sys.argv[4]
os.path.join(file)
f = open(file,"w")
X = X.astype(np.str)
print len(X)
for sample in range(len(X)):
	l = ",".join(X[sample])
	f.write(l+"\n")
f.close()

print X[0]
