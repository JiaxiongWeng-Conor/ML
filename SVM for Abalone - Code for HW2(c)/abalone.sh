#!/bin/bash

#---arguments---#
LIBSVM_PATH=libsvm
TRAIN_SIZE=3133
TEST_SIZE=1044
DATAPATH=abalone/abalone.data
TRAIN_DATA_PATH=abalone/abalone.train
TEST_DATA_PATH=abalone/abalone.test

#---alias---#
alias svm-scale=$LIBSVM_PATH/svm-scale
alias svm-train=$LIBSVM_PATH/svm-train
alias svm-predict=$LIBSVM_PATH/svm-predict


##-----------------##
##---Problem C.2---##
echo "-----------------"
echo "-- Problem C.2 --"
echo

# setup
rm -r 2*
mkdir 2.log
RANGE_PATH=2.log/range
SCL_TRAIN=2.log/train
SCL_TEST=2.log/test

# Preprocessing data into libsvm format
python3 preprocess.py --data $DATAPATH --trainsize $TRAIN_SIZE --testsize $TEST_SIZE --trainfile $TRAIN_DATA_PATH --testfile $TEST_DATA_PATH

# Scaling
svm-scale -l -1 -u 1 -s $RANGE_PATH $TRAIN_DATA_PATH > $SCL_TRAIN
svm-scale -r $RANGE_PATH $TEST_DATA_PATH > $SCL_TEST

echo "Scaled data saved in $SCL_TRAIN and $SCL_TEST. Range parameters are saved in $RANGE_PATH."
echo


##-----------------##
##---Problem C.3---##
# Finding d* and C* via 5-fold cross-validation
echo "-----------------"
echo "-- Problem C.3 --"
echo

# setup
rm -r 3*
mkdir 3.log
TMP=3.log/tmp
REC_FILE=3.log/accuracy
FIGPATH=3.log/cv.png

# cross validation
for d in 1 2 3 4 5
do
for k in {-6..6}
do
C=$(bc -l <<< "3 ^($k)")

echo "Doing 5-fold cross-validation for d=${d}, power on C=${k} ..."
svm-train -t 1 -d $d -c $C -v 5 -h 0 $SCL_TRAIN > $TMP
python3 extract.py --src $TMP --save $REC_FILE -d $d -C $C

done
done

# plot
python3 plotaccu.py --src $REC_FILE --save $FIGPATH -n $TRAIN_SIZE
echo "Figure saved in $FIGPATH."
echo



##-----------------##
##---Problem C.4---##
echo "-----------------"
echo "-- Problem C.4 --"
echo

rm -r 4*
mkdir 4.log
OUTFILE=4.log/out
TMP=4.log/tmp
REC_CV=4.log/accuracy_cv
REC_TEST=4.log/accuracy_te
REC_CNT=4.log/cnt
MODELPATH=4.log/model
FIG_ERR=4.log/errors.png
FIG_SV=4.log/supvec.png

# fix the best C
C=243

for d in 1 2 3 4 5
do
# cross validation
echo "Doing 5-fold cross-validation for d=${d} (C=${C}) ..."
svm-train -t 1 -d $d -c $C -v 5 -h 0 $SCL_TRAIN > $TMP
python3 extract.py --src $TMP --save $REC_CV -d $d -C $C
# training
echo "Training ..."
svm-train -t 1 -d $d -c $C -h 0 $SCL_TRAIN $MODELPATH > $TMP
python3 count.py --src $MODELPATH --save $REC_CNT -C $C
# testing
echo "Testing ..."
svm-predict $SCL_TEST $MODELPATH $OUTFILE > $TMP
python3 extract.py --src $TMP --save $REC_TEST -d $d -C $C
done

# plot error figure
python3 plot4d.py --src $REC_CV --src2 $REC_TEST --save $FIG_ERR --mode error -C $C
# plot number of support vectors
python3 plot4d.py --src $REC_CNT --save $FIG_SV --mode supvec -C $C
echo "Figure saved in $FIG_ERR and $FIG_SV."
echo

rm $TMP



##-----------------##
##---Problem C.5---##
echo "-----------------"
echo "-- Problem C.5 --"
echo

rm -r 5*
mkdir 5.log
TMP_TRAIN=5.log/train
TMP_TEST=5.log/test
TMP_RANGE=5.log/range
TMP_TRAIN_SCL=5.log/train.scale
TMP_TEST_SCL=5.log/test.scale
TMP=5.log/tmp
REC_TRAIN=5.log/accuracy_tr
REC_TEST=5.log/accuracy_te
OUTFILE=5.log/out

MODELPATH=5.log/model
FIG_PATH=5.log/err2sample.png

# fix the best C and d
C=243
d=3
let N=$TRAIN_SIZE+$TEST_SIZE

for s in {10..$N..100}
do
echo "Sample size $s/$N"

let ts=$TRAIN_SIZE+$TEST_SIZE-$s
python3 preprocess.py --data $DATAPATH --trainsize $s --testsize $ts --trainfile $TMP_TRAIN --testfile $TMP_TEST
# Scaling
svm-scale -l -1 -u 1 -s $TMP_RANGE $TMP_TRAIN > $TMP_TRAIN_SCL
svm-scale -r $TMP_RANGE $TMP_TEST > $TMP_TEST_SCL
# train
svm-train -t 1 -d $d -c $C -h 0 $TMP_TRAIN_SCL $MODELPATH > $TMP
svm-predict $TMP_TRAIN_SCL $MODELPATH $OUTFILE > $TMP
python3 extract.py --src $TMP --save $REC_TRAIN -d $d -C $C
# test
svm-predict $TMP_TEST_SCL $MODELPATH $OUTFILE > $TMP
python3 extract.py --src $TMP --save $REC_TEST -d $d -C $C
done

python3 plot4sample.py --sample_range "[10,${N},100]" --src $REC_TRAIN --src2 $REC_TEST --save $FIG_PATH
echo "Figure saved to $FIG_PATH."
echo

# rm $TMP
rm $TMP
rm $TMP_RANGE
rm $MODELPATH
rm $OUTFILE
rm $TMP_TRAIN
rm $TMP_TRAIN_SCL
rm $TMP_TEST
rm $TMP_TEST_SCL


# ##-----------------##
# ##---Problem C.6---##
# echo "-----------------"
# echo "-- Problem C.6 --"
# echo "(note this part may take very long time, can seperatedly run in 'abalone.sparseSVM.sh' file)"

# rm -r 6*
# mkdir 6.log
# python3 sparse_svm.py --train $SCL_TRAIN --test $SCL_TEST