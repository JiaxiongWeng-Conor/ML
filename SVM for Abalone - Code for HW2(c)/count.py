import argparse
import re

# CONFIGURATIONS #
parser = argparse.ArgumentParser(description='count support vector args.')
parser.add_argument('--src', type=str, help='libsvm model file to extract from')
parser.add_argument('--save', type=str, help='info file to write in')
parser.add_argument('-C', type=float, help='info file to write in')
args = parser.parse_args()

from_file = args.src
to_file = args.save
C = args.C

# READ
on_margin, off_margin = 0, 0
with open(from_file, "r") as f:
    lines = f.readlines()
    # total SV
    total = int(re.search('([0-9]+)', lines[6]).group())
    # on-margin SV
    for line in lines[11:]:
        ci = float(line.split(' ')[0])
        if ci == C or ci == -C:
            on_margin += 1
        else: 
            off_margin += 1

# WRITE
with open(to_file, "a") as f:
    f.write(f"{total}\t{on_margin}\n")