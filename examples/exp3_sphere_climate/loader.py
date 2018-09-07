import numpy as np
from glob import glob
import os
import random
from torch.utils.data import Dataset, DataLoader

# precomputed mean and std of the dataset
precomp_mean = [26.160023, 0.98314494, 0.116573125, -0.45998842, 0.1930554, 0.010749293, 98356.03, 100982.02, 216.13145, 258.9456, 3.765611e-08, 288.82578, 288.03925, 342.4827, 12031.449, 63.435772]
precomp_std =  [17.04294, 8.164175, 5.6868863, 6.4967732, 5.4465833, 0.006383436, 7778.5957, 3846.1863, 9.791707, 14.35133, 1.8771327e-07, 19.866386, 19.094095, 624.22406, 679.5602, 4.2283397]

class ClimateSegLoader(Dataset):
    """Data loader for Climate Segmentation dataset."""

    def __init__(self, data_dir, partition="train", normalize_mean=precomp_mean, normalize_std=precomp_std):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
        """
        assert(partition in ["train", "test", "val"])
        with open(partition+"_split.txt", "r") as f:
            lines = f.readlines()
        self.flist = [os.path.join(data_dir, l.replace('\n', '')) for l in lines]
        self.mean = np.expand_dims(precomp_mean, -1).astype(np.float32)
        self.std = np.expand_dims(precomp_std, -1).astype(np.float32)

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        # load files
        fname = self.flist[idx]
        data = (np.load(fname)["data"] - self.mean) / self.std
        labels = np.load(fname)["labels"].astype(np.int)
        # one-hot to categorical labels
        labels = np.argmax(labels, axis=0)
        return data, labels
