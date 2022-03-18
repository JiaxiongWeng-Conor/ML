# sparse_svm.py
# choice of kernel: polynomial
import numpy as np
import re
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# >> CONFIGURATIONS << #
parser = argparse.ArgumentParser(description='arguments for self-defined sparse SVM.')
parser.add_argument('--train', type=str, help='train file after scaled')
parser.add_argument('--test', type=str, help='test file after scaled')
args = parser.parse_args()

# >> GLOBAL VAR << #
C = 81 # regularization strength
d = 3 # polynomial degree
epochs = 5
learning_rate = 1e-6

# >> MODEL TRAINING FUNCTIONS << #
def data_loader(file):
    data = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(' ')
            y = int(items[0])
            n_feature = 10; x = [0] * n_feature
            for i in range(n_feature):
                match = re.search(f"{i+1}:([\-.0-9]+)", line)
                if match:
                    x[i] = eval(match.group(1))
            data.append(x + [y])

    return np.array(data, dtype=np.float)


def kernel(va, vb, degree, gamma=.1, coef0=0):
    return (gamma * np.dot(va, vb) + coef0) ** degree

def cal_yhat(xi, Alpha, b, X, Y, K, d, intrain=False):
    M = X.shape[0]
    yhat = 0
    for j in range(M):
        if intrain:
            yhat += Alpha[j] * Y[j] * K[xi,j]
        else:
            yhat += Alpha[j] * Y[j] * kernel(xi, X[j,:], degree=d)
    yhat += b
    return yhat

def compute_cost(i, yhat, Alpha, Y, C):
    cost = 1/2 * Alpha[i] + C * max(1 - Y[i]*yhat, 0)
    return cost

def cal_gradient(i, yhat, Alpha, Y, K, C): 
    M = Y.shape[0]
    ga = np.zeros_like(Alpha)
    gb = 0

    if 1 - Y[i] * yhat <= 0:
        ga[i] = 1/2
    else:
        ga[i] = 1/2
        for j in range(M):
            ga[i] -= C * Y[i] * Y[j] * K[i,j]
            if j != i:
                ga[j] = - C * Y[i] * Y[j] * K[i,j]
        gb = C * Y[i]

    return ga, gb

def predict(Alpha, b, X, traindata, K, d, fit_train=False):
    pred = []
    for i in range(X.shape[0]):
        if fit_train:
            yhat = cal_yhat(i, Alpha, b, traindata[:,:-1], traindata[:,-1], K, d, intrain=True)
        else:
            yhat = cal_yhat(X[i], Alpha, b, traindata[:,:-1], traindata[:,-1], K, d, intrain=False)
        pred.append(np.sign(yhat))
    return np.array(pred)

def train(traindata, validdata, d, C):
    np.random.seed(101)

    M = traindata.shape[0]
    X_train, y_train = traindata[:,:-1], traindata[:,-1]
    X_valid, y_valid = validdata[:,:-1], validdata[:,-1]

    # parameter initialization
    Alpha = np.zeros(M)
    b = 0

    # Gram matrix
    K = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            K[i,j] = kernel(X_train[i], X_train[j], degree=d)

    # Training
    print(f"-- Sparse SVM Training begins (d={d}, C={C}) --")
    for epoch in range(epochs):
        cost = 0
        # X_train, y_train = shuffle(X_train, y_train)
        # # .. and also shuffle K
        for i in tqdm(range(M)):
        # for i in range(M):
            xi = X_train[i]
            import time
            t = time.time()
            yhat = cal_yhat(i, Alpha, b, X_train, y_train, K, d, intrain=True)
            # t2 = time.time(); print(t2-t)
            cost += compute_cost(i, yhat, Alpha, y_train, C) / M
            # t3 = time.time(); print(t3-t2)
            ga, gb = cal_gradient(i, yhat, Alpha, y_train, K, C)
            # t4 = time.time(); print(t4-t3); print()

            # sgd
            lr_decayed = learning_rate * 0.98 ** epoch # decay on epoch
            Alpha = np.maximum(0, Alpha - lr_decayed * ga)

            b -= lr_decayed * gb
        print(f"Epoch: {epoch+1}/{epochs}\tCost={cost:.6f}")

    train_pred = predict(Alpha, b, X_train, traindata, K, d, fit_train=True)
    valid_pred = predict(Alpha, b, X_valid, traindata, K, d, fit_train=False)
    train_accu = accuracy_score(y_train, train_pred)
    valid_accu = accuracy_score(y_valid, valid_pred)
    print(f"Training accuracy={train_accu*100:.3f}%\tValid accuracy={valid_accu*100:.3f}%")

    return train_accu, valid_accu


def main():
    traindata = data_loader(args.train)
    testdata = data_loader(args.test)

    dlist = np.array([1,2,3,4,5])
    cv_err_list = np.zeros_like(dlist)
    test_err_list = np.zeros_like(dlist)
    for d in dlist:
        
        # Cross-validation
        kfold = 5
        l = traindata.shape[0] // kfold
        cv_train_accu, cv_valid_accu = 0, 0
        print(f"\n== {kfold}-fold Cross-Validation ==")
        for k in range(kfold):
            print(f"\n-- CV: Fold {k+1}/{kfold}")
            valid_index = list(range(k*l, (k+1)*l))
            cv_train = np.delete(traindata, valid_index, axis=0)
            cv_valid = traindata[valid_index,:]

            tr_accu, vl_accu = train(cv_train, cv_valid, d, C)
            
            cv_train_accu += tr_accu / kfold
            cv_valid_accu += vl_accu / kfold

        cv_err_list[d-1] = (1 - cv_valid_accu) * 100

        # Full training
        print(f"\n== Full Training ==")
        tr_accu, vl_accu = train(traindata, testdata, d, C)
        test_err_list[d-1] = (1 - vl_accu) * 100

    # print(cv_err_list, test_err_list)

    # Plot
    plt.figure(figsize=(7.5,4.8))
    plt.plot(dlist, cv_err_list, label="5-fold cv error", marker='^')
    plt.plot(dlist, test_err_list, label="Test error", marker='o')
    plt.grid(True)
    plt.ylabel('Error (%) ')
    plt.xlabel("Polynomial Degree")
    plt.title(f'5-Fold CV and Test Error v.s. Polynomial Degree for Sparse SVM (C={C}, Epochs={epochs})')
    plt.legend(title="Error type")
    plt.savefig("6.log/errors.png")
    # plt.show()
    plt.close()


# >> MAIN << #
main()

