from glob import glob
import os
import random
import numpy as np

# Script to create train, test, validation splits.
# Control parameters
SEED = 0
DATADIR = 'data_5_all'
TEST_RATIO = 0.2
VAL_RATIO = 0.1
TRAIN_RATIO = 0.7

# Create Split
flist = glob(os.path.join(DATADIR, "data-*.npz"))
flist = [os.path.basename(l) for l in flist]
if len(flist) == 0:
    print("[!] Wrong data directory.")
    assert(0)

random.Random(SEED).shuffle(flist)
n_test = int(np.round(TEST_RATIO*len(flist)))
n_val = int(np.round(VAL_RATIO*len(flist)))
n_train = len(flist) - n_test - n_val
flist_test = flist[:n_test]
flist_val = flist[n_test:(n_test+n_val)]
flist_train = flist[(n_test+n_val):]

with open('test_split.txt', 'w') as f:
    f.writelines("%s\n" % item for item in flist_test)

with open('val_split.txt', 'w') as f:
    f.writelines("%s\n" % item for item in flist_val)

with open('train_split.txt', 'w') as f:
    f.writelines("%s\n" % item for item in flist_train)