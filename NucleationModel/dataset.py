class Dataset():
    def __init__(self, snapshots, labels, weights, flag):
        self._snapshots = snapshots
        self._labels = labels
        self._weights = weights
        self._flag = flag
        self._size = len(labels)
        self._dimensions = len(snapshots[0])

    def __str__(self):
        return "{} set has a size of {} snapshots along {} input dimensions"\
            .format(self._flag, self._size, self._dimensions)

    @staticmethod
    def initialize_train_val_test_datasets(
            train_snapshots, train_labels, train_weights,
            val_snapshots, val_labels, val_weights,
            test_snapshots, test_labels, test_weights):
        return Dataset(
                train_snapshots, train_labels, train_weights, "Training"), \
            Dataset(
                val_snapshots, val_labels, val_weights, "Validation"), \
            Dataset(
                test_snapshots, test_labels, test_weights, "Testing")

    @property
    def size(self):
        return self._size

    @property
    def flag(self):
        return self._flag

    @property
    def snapshots(self):
        return self._snapshots

    @property
    def labels(self):
        return self._labels

    @property
    def weights(self):
        return self._weights
