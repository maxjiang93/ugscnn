from glob import glob
import os
from tqdm import tqdm
import numpy as np; np.set_printoptions(threshold=np.nan)


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

def compute_stats(data_dir):
    files = sorted(glob(os.path.join(data_dir, "*.npz")))
    rolling_sum = 0
    sr = StatsRecorder()
    for f in tqdm(files):
        u = np.load(f)
        sr.update(u['data'].T)
        labels = u['labels']
        count = np.sum(labels, axis=1)
        ratio = count / np.sum(count)
        rolling_sum += ratio
    label_ratio = rolling_sum / len(files)
    print("Data Mean: \n", '[' + ', '.join(list(sr.mean.astype(str))) + ']') 
    print("Data std: \n", '[' + ', '.join(list(sr.std.astype(str))) + ']') 
    print("Labels Ratio:", label_ratio)
    return sr.mean, sr.std, label_ratio

mean, std, label_ratio = compute_stats("data_5_all")