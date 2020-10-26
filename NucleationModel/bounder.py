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
        """Set the values of a snapshot that lie outside of the bounds to that
        bound while leaving the other values unchanged.
        Initially transpose the snapshots to columns.
        Bound all entries betwwen their respective upper and lower bound.
        Transpose the columns back into snapshots and return.
        """
        return np.transpose(self.bound_all_columns(np.transpose(snapshots)))

    def bound_all_columns(self, columns):
        return [self.bound_one_dimensional_column(
                    self.upper_bound[col_nr],
                    self.lower_bound[col_nr],
                    columns[col_nr])
                for col_nr in range(self._dimensions)]

    def bound_one_dimensional_column(self, upper_bound, lower_bound, column):
        return [self.bound_one_dimensional_entry(
                    upper_bound, lower_bound, entry)
                for entry in column]

    def bound_one_dimensional_entry(self, upper_bound, lower_bound, entry):
        return min(upper_bound,
                   max(lower_bound, entry))
