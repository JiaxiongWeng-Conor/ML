# SVM for Abalone
Implementation of SVM using libsvm on Abalone dataset. 

This is the part (c) of homework 2 for [XXX] course.

## Dataset
* [Abalone dataset](http://archive.ics.uci.edu/ml/datasets/Abalone)

## Environment
* < [libsvm 3.25](https://github.com/cjlin1/libsvm) >
* < pandas 1.3.1 >
* < numpy 1.19.5 >
* < sklearn 0.24.2 >

## Usage instructions
* Direct use `libsvm/` in the repo or download `libsvm.zip` and compile by yourself:

```shell
$ libsvm.zip
$ cd libsvm
$ make
$ cd ..
```
* Run SVM using libsvm on dataset (Problem C.2-C.6):

```shell
$ abalone.sh
```
* Problem C.6 can take very long time and hence is commented in the file by default. Alternatively you can directly see the directory `6.log/` for results. If you want to generate results by yourself, you can seperately run:

```shell
$ abalone.sparseSVM.sh
```