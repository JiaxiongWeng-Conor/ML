SCL_TRAIN=2.log/train
SCL_TEST=2.log/test

##-----------------##
##---Problem C.6---##
echo "-----------------"
echo "-- Problem C.6 --"
echo "(note this part may take very long time)"
echo

rm -r 6*
mkdir 6.log
python3 sparse_svm.py --train $SCL_TRAIN --test $SCL_TEST