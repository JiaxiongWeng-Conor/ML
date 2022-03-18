import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# CONFIGURATIONS #
parser = argparse.ArgumentParser(description='extract accuracy args.')
parser.add_argument('--sample_range', type=str, help='range of sample size')
parser.add_argument('--src', type=str, help='libsvm file to extract from')
parser.add_argument('--src2', type=str, help='libsvm file to extract from')
parser.add_argument('--save', type=str, help='info file to write in')
args = parser.parse_args()

sample_range = eval(args.sample_range)
from_file = args.src
from_file2 = args.src2
to_file = args.save

# READ
datatrain = pd.read_table(from_file, sep='\t', header=None, dtype={1: np.int64})
datatest = pd.read_table(from_file2, sep='\t', header=None, dtype={1: np.int64})

d = int(datatrain.iloc[0,1])
C = int(datatrain.iloc[0,2])
trainerror = 100 - datatrain.iloc[:,0]
testerror = 100 - datatest.iloc[:,0]
samples = np.array(range(sample_range[0], sample_range[1], sample_range[2]))

# PLOT
plt.figure()
plt.plot(samples, trainerror, label="Training error")
plt.plot(samples, testerror, label="Test error")
plt.grid(True)
plt.ylabel('Error (%) ')
plt.xlabel("Sample size")
plt.title(f'Training and Testing Error v.s. Sample Size (d={d}, C={C})')
plt.legend(title="Error type")
plt.savefig(to_file)
# plt.show()
plt.close()
    
