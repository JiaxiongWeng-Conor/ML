import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# CONFIGURATIONS #
parser = argparse.ArgumentParser(description='extract accuracy args.')
parser.add_argument('--src', type=str, help='libsvm file to extract from')
parser.add_argument('--save', type=str, help='info file to write in')
parser.add_argument('-n', type=int, help='training sample size')
args = parser.parse_args()

from_file = args.src
to_file = args.save
n = args.n

# READ
data = pd.read_table(from_file, sep='\t', header=None, dtype={1: np.int64})

# PLOT
plt.figure()
cmap = plt.cm.get_cmap('Paired')
for d in np.unique(data.iloc[:, 1]):
    clist = np.log(data.loc[data.iloc[:,1] == d, 2]) / np.log(3) # power list
    errorlist = 100 - data.loc[data.iloc[:,1] == d, 0]
    std = errorlist * (100-errorlist) / n
    upper = errorlist + std
    lower = errorlist - std

    plt.plot(clist, errorlist, c=cmap(d), label=f'd={d}', alpha=.8)
    plt.plot(clist, upper, c=cmap(d), linestyle=":", linewidth=.5)
    plt.plot(clist, lower, c=cmap(d), linestyle=":", linewidth=.5)
    plt.fill_between(clist, upper, lower, color=cmap(d), alpha=.15)


plt.grid(True)
plt.ylabel('Error (%) ')
plt.xlabel(r"$\log_3{C}$")
plt.title('5-Fold Cross-Validation Error with +/- 1 Standard Deviation')
plt.legend(title="Polynomial degree")
plt.savefig(to_file)
# plt.show()
plt.close()
