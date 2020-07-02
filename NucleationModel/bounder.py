import numpy as np


class Bounder():
    def __init__(self, base_snapshots, outlier_cutoff):
        self._dimensions = len(base_snapshots[0])
        self._lower_bound = np.percentile(
            base_snapshots,
            100 * outlier_cutoff,
            axis=0)
        self._upper_bound = np.percentile(
            base_snapshots,
            100 * (1 - outlier_cutoff),
            axis=0)

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    def bound_snapshots(self, snapshots):
        """Set the values of a snapshopt that lie outside of the bounds to that
        bound while leaving the other values unchanged.
        Initially transpose the snapshots to a column list
        For each column, iterate over all entries and compare them to the
        lower or upper bound of that column. If they are lower or higher,
        change to the value of that bound.
        Return the transpose of the column list, thereby yielding
        the cleaned snapshots.
        """
        column_list = np.transpose(snapshots)
        column_list = [[min(self.upper_bound[col_nr],
                        max(self.lower_bound[col_nr], entry))
                        for entry in column_list[col_nr]]
                       for col_nr in range(self._dimensions)]
        return np.transpose(column_list)
