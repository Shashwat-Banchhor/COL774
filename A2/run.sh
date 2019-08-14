# if [ $4 == a ]
# then
# 	python q1.py $1 $2 $3 $4 
# fi

# if [ $4 == b ]
# then
# 	python q1.py $1 $2 $3 $4 
# fi
# if [ $4 == c ]
# then
# 	python q1.py $1 $2 $3 $4 
# fi
# if [ $4 == d ]
# then
# 	python q1.py $1 $2 $3 $4 
# fi

# if [ $4 == e ]
# then
# 	python q1.py $1 $2 $3 $4  
# fi

# if [ $4 == g ]
# then
# 	python q1.py $1 $2 $3 $4  
# fi

if [ $1 == 1 ]
then
	python q1.py $1 $2 $3 $4 
fi

# export "libsvm-3.23/python:${PYTHONPATH}"

if [ $1 == 2 ]
then
	export PYTHONPATH="libsvm-3.23/python:${PYTHONPATH}"

	python svm1.py $2 $3 $4 $5
fi





