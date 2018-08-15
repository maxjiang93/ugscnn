import numpy as np
from glob import glob
import os
import random
from torch.utils.data import Dataset, DataLoader


class ClimateSegLoader(Dataset):
    """Data loader for Climate Segmentation dataset."""

    def __init__(self, data_dir, partition="train", test_size=0.1, random_seed=0):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
            test_size: test ratio [0, 1]
            random_seed: seed for randomly partitioning the set into test and train
        """
        assert(partition in ["train", "test"])
        assert(test_size >= 0 and test_size <= 1)
        flist = glob(os.path.join(data_dir, "data-*.npz"))
        if len(flist) == 0:
            print("[!] Wrong data directory.")
            assert(0)

        random.Random(random_seed).shuffle(flist)
        n_test = int(np.round(test_size*len(flist)))
        n_train = len(flist) - n_test
        self.flist_train = flist[:n_train]
        self.flist_test = flist[n_train:]
        if partition == "train":
            self.flist = self.flist_train
        else:
            self.flist = self.flist_test

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        # load files
        fname = self.flist[idx]
        data = np.load(fname)["data"].T
        labels = np.load(fname)["labels"]
        # one-hot to categorical labels
        labels = np.argmax(labels, axis=-1)
        return data, labels
