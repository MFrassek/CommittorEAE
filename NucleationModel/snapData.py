import numpy as np
from sklearn.utils import shuffle


class SnapData():
    def __init__(
            self, AA_snapshots, AB_snapshots,
            BA_snapshots, BB_snapshots,
            AA_labels, AB_labels,
            BA_labels, BB_labels,
            AA_weights, AB_weights,
            BA_weights, BB_weights):
        self._AA_snapshots = AA_snapshots
        self._AB_snapshots = AB_snapshots
        self._BA_snapshots = BA_snapshots
        self._BB_snapshots = BB_snapshots
        self._AA_labels = AA_labels
        self._AB_labels = AB_labels
        self._BA_labels = BA_labels
        self._BB_labels = BB_labels
        self._AA_weights = AA_weights
        self._AB_weights = AB_weights
        self._BA_weights = BA_weights
        self._BB_weights = BB_weights

    @property
    def snapshot_cnt(self):
        return len(self._AA_snapshots) + len(self._AB_snapshots) + \
                len(self._BA_snapshots) + len(self._BB_snapshots)

    @property
    def AA_snapshot_cnt(self):
        return len(self._AA_snapshots)

    @property
    def AB_snapshot_cnt(self):
        return len(self._AB_snapshots)

    @property
    def BA_snapshot_cnt(self):
        return len(self._BA_snapshots)

    @property
    def BB_snapshot_cnt(self):
        return len(self._BB_snapshots)

    @property
    def snapshots(self):
        """returns list of all snapshots in order AA, AB, BA, BB"""
        return np.array([snapshot for paths
                        in [self._AA_snapshots, self._AB_snapshots,
                            self._BA_snapshots, self._BB_snapshots]
                        for snapshot in paths])

    @property
    def AA_snapshots(self):
        return self._AA_snapshots

    @property
    def AB_snapshots(self):
        return self._AB_snapshots

    @property
    def BA_snapshots(self):
        return self._BA_snapshots

    @property
    def BB_snapshots(self):
        return self._BB_snapshots

    @property
    def labels(self):
        """returns list of all labels in order AA, AB, BA, BB"""
        return np.array([snapshot for paths
                        in [self._AA_labels, self._AB_labels,
                            self._BA_labels, self._BB_labels]
                        for snapshot in paths])

    @property
    def AA_labels(self):
        return self._AA_labels

    @property
    def AB_labels(self):
        return self._AB_labels

    @property
    def BA_labels(self):
        return self._BA_labels

    @property
    def BB_labels(self):
        return self._BB_labels

    @property
    def weights(self):
        """returns list of all weights in order AA, AB, BA, BB"""
        return np.array([snapshot for paths
                         in [self._AA_weights, self._AB_weights,
                             self._BA_weights, self._BB_weights]
                         for snapshot in paths])

    @property
    def AA_weights(self):
        return self._AA_weights

    @property
    def AB_weights(self):
        return self._AB_weights

    @property
    def BA_weights(self):
        return self._BA_weights

    @property
    def BB_weights(self):
        return self._BB_weights

    def shuffle_lists(self):
        return shuffle(
            self.snapshots,
            self.labels,
            self.weights,
            random_state=42)

    def split_lists(self, train_ratio, val_ratio):
        assert isinstance(train_ratio, float) \
            and train_ratio > 0.0, \
            "train_ratio needs to be a float higher than 0.0"
        assert isinstance(val_ratio, float) \
            and val_ratio > 0.0, \
            "val_ratio needs to be a float higher than 0.0"
        assert train_ratio + val_ratio < 1.0, \
            "Sum of train_ratio and val_ratio must be lower than 1.0"
        train_end = int(self.snapshot_cnt * train_ratio)
        val_end = train_end + int(self.snapshot_cnt * val_ratio)
        snapshots, labels, weights \
            = self.shuffle_lists()
        return np.array([*snapshots[:train_end]]), \
            np.array([*labels[:train_end]]), \
            np.array([*weights[:train_end]]), \
            np.array([*snapshots[train_end:val_end]]), \
            np.array([*labels[train_end:val_end]]), \
            np.array([*weights[train_end:val_end]]), \
            np.array([*snapshots[val_end:]]), \
            np.array([*labels[val_end:]]), \
            np.array([*weights[val_end:]])

    def __str__(self):
        return "Snapshots (cnt): {} \n{} in AA\n{} in AB\n{} in BA\n{} in BB"\
            .format(
                self.snapshot_cnt,
                self.AA_snapshot_cnt,
                self.AB_snapshot_cnt,
                self.BA_snapshot_cnt,
                self.BB_snapshot_cnt)
