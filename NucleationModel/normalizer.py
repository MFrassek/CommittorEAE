import numpy as np
import numpy as np


class Normalizer():
    def __init__(self, base_snapshots):
        self._dimensions = len(base_snapshots[0])
        self._mean = np.mean(base_snapshots, axis=0)
        self._std = np.array([element if element != 0 else 1 for element
                             in np.std(base_snapshots, axis=0)])
        self._inv_std = np.ones(self._dimensions) / self.std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def inv_std(self):
        return self._inv_std

    def normalize_snapshots(self, snapshots):
        """Normalize the list by substracting the mean
        and multiplying with the inverse of the standard deviation.
        """
        return (snapshots - self.mean) * self.inv_std
