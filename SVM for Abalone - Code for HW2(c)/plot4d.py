import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# CONFIGURATIONS #
parser = argparse.ArgumentParser(description='extract accuracy args.')
parser.add_argument('--src', type=str, help='libsvm file to extract from')
parser.add_argument('--src2', type=str, help='libsvm file to extract from')
parser.add_argument('--save', type=str, help='info file to write in')
parser.add_argument('--mode', type=str, choices=['error','supvec'], help='plotting mode')
parser.add_argument('-C', type=float)
args = parser.parse_args()

from_file = args.src
from_file2 = args.src2
to_file = args.save
mode = args.mode
C = args.C


if mode == 'error':
    assert from_file and from_file2 # both src should not be empty
    # READ
    datacv = pd.read_table(from_file, sep='\t', header=None, dtype={1: np.int64})
    datatest = pd.read_table(from_file2, sep='\t', header=None, dtype={1: np.int64})

    dlist = datacv.iloc[:,1]
    cverror = 100 - datacv.iloc[:,0]
    testerror = 100 - datatest.iloc[:,0]

    # PLOT
    plt.figure()
    plt.plot(dlist, cverror, label="5-fold cv error", marker='^')
    plt.plot(dlist, testerror, label="Test error", marker='o')
    plt.grid(True)
    plt.ylabel('Error (%) ')
    plt.xlabel("Polynomial Degree")
    plt.title(f'5-Fold CV and Test Error v.s. Polynomial Degree (C={C})')
    plt.legend(title="Error type")
    plt.savefig(to_file)
    # plt.show()
    plt.close()

elif mode == "supvec":
    # READ
    data = pd.read_table(from_file, sep='\t', header=None)
    dlist = data.index + 1

    # PLOT
    plt.figure()
    plt.plot(dlist, data.iloc[:,0], label="Total # of SV", marker='^')
    plt.plot(dlist, data.iloc[:,1], label="# of on-margin", marker='o')
    plt.grid(True)
    plt.ylabel('Number of SV')
    plt.xlabel("Polynomial Degree")
    plt.title(f'Number of Support Vectors v.s. Polynomial Degree (C={243})')
    plt.legend()
    plt.savefig(to_file)
    # plt.show()
    plt.close()
    
