import numpy as np
from glob import glob
import os
import random
from torch.utils.data import Dataset, DataLoader

# precomputed mean and std of the dataset
precomp_mean = [0.4974898, 0.47918808, 0.42809588, 1.0961773]
precomp_std = [0.23762763, 0.23354423, 0.23272438, 0.75536704]

class ClimateSegLoader(Dataset):
    """Data loader for Climate Segmentation dataset."""

    def __init__(self, data_dir, partition="train", fold=0, normalize_mean=precomp_mean, normalize_std=precomp_std):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
            fold: 1, 2 or 3 (for 3-fold cross-validation)
            
        """
        assert(partition in ["train", "test"])
        assert(fold in [1, 2, 3])
        with open(partition+"_split.txt", "r") as f:
            lines = f.readlines()
        if fold == 1:
            train_areas = ['1', '2', '3', '4', '6']
            test_areas = ['5a', '5b']
        elif fold == 2:
            train_areas = ['1', '3', '5a', '5b', '6']
            test_areas = ['2', '4']
        elif fold == 3:
            train_areas = ['2', '4', '5a', '5b']
            test_areas = ['1', '3', '6']

        if partition == "train":
            self.areas = train_areas
        else:
            self.areas = test_areas

        self.flist = []
        for a in self.areas:
            area_dir = os.path.join(data_dir, "area_" + a)
            file_format = os.path.join(area_dir, "*.npz")
            self.flist += sorted(glob(file_format))

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
