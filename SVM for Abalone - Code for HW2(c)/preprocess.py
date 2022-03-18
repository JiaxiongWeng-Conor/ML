import pandas as pd
import numpy as np
import argparse

# CONFIGURATIONS #
parser = argparse.ArgumentParser(description='extract accuracy args.')
parser.add_argument('--data', type=str, help='libsvm file to extract from')
parser.add_argument('--trainsize', type=int, help='info file to write in')
parser.add_argument('--testsize', type=int, help='info file to write in')
parser.add_argument('--trainfile', type=str, help='info file to write in')
parser.add_argument('--testfile', type=str, help='info file to write in')
args = parser.parse_args()

TRAIN=args.trainsize
TEST=args.testsize
DATAFILE=args.data
TRAINFILE=args.trainfile
TESTFILE=args.testfile


# FUNCTIONS #
def importdata(filename):
    data = pd.read_table(filename, sep=',', header=None)
    # process target
    data.iloc[:,-1] = np.where(data.iloc[:,-1] <= 9, 1, -1)
    # categorical variable: onehot encode
    data = pd.concat((pd.get_dummies(data.iloc[:,0]), data.drop(0, axis=1)), axis=1)

    X_train = data.iloc[:TRAIN, :-1]
    y_train = data.iloc[:TRAIN, -1]
    X_test = data.iloc[-TEST:, :-1]
    y_test = data.iloc[-TEST:, -1]

    return X_train, y_train, X_test, y_test

def exportdata(X, Y, filename):
    with open(filename, "w") as f:
        for i in range(X.shape[0]):
            x = [f'{k+1}:{X.iloc[i,k]}' for k in range(X.shape[1])]
            row = '\t'.join([str(Y.iloc[i])] + x)
            f.write(row + "\n")
    f.close()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = importdata(DATAFILE)
    exportdata(X_train, y_train, TRAINFILE)
    exportdata(X_test, y_test, TESTFILE)