# $2 train 
# $3 test
# $4 vali


if [ $1 == 1 ]
then
	python preprocess.py $2 $4 $3 $1
fi

if [ $1 == 2 ]
then
	python preprocess.py $2 $4 $3 $1
fi

if [ $1 == 3 ]
then
	python preprocess_c.py $2 $4 $3 $1
fi

 


if [ $1 == 4 ]
then
	python decisionTreeb.py $2 $4 $3 $1
fi

if [ $1 == 5 ]
then
	python decisionTreeb.py $2 $4 $3 $1
fi

if [ $1 == 6 ]
then
	python decisionTreeb.py $2 $4 $3 $1
fi
