import re
import argparse


# CONFIGURATIONS #
parser = argparse.ArgumentParser(description='extract accuracy args.')
parser.add_argument('--src', type=str, help='libsvm file to extract from')
parser.add_argument('--save', type=str, help='info file to write in')
parser.add_argument('-d', type=float, help='value of polynomial degree')
parser.add_argument('-C', type=float, help='value of cost')
args = parser.parse_args()

from_file = args.src
to_file = args.save
d = args.d
C = args.C

# READ
with open(from_file, "r") as f:
    line = f.readlines()[-1]
    accuracy = float(re.search('= (.+?)%', line).group(1))

# WRITE
with open(to_file, "a") as f:
    row = f"{accuracy:.4f}\t{d}\t{C:.4f}\n"
    f.write(row)