from glob import glob
import os
from tqdm import tqdm
import numpy as np


class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

def compute_stats():
	data_dir = "data/train_normal"
	files = sorted(glob(os.path.join(data_dir, "*.npy")))
	sr = StatsRecorder()
	for f in tqdm(files):
		u = np.load(f)
		sr.update(u.T)
	print(sr.mean)
	print(sr.std)
	return sr.mean, sr.std

def normalize_data(mean, std):
	data_dir = "data/train_normal"
	files = sorted(glob(os.path.join(data_dir, "*.npy")))
	for f in tqdm(files):
		u = np.load(f)
		v = (u.T-mean)/std
		new_f = f.replace("b64", "sp5")
		np.save(new_f, v.T)

# mean, std = compute_stats()
compute_stats()
normalize_data(mean, std)
