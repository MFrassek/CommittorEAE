import numpy as np


class ImportanceData():
    def __init__(
            self, snapshots, labels, weights,
            corr_thresholds):
        self._snapshots = snapshots
        self._snapshot_cnt = len(snapshots)
        self._columns = np.transpose(snapshots)
        self._labels = labels
        self._weights = weights
        self._strong_corr_threshold = corr_thresholds[0]
        self._weak_corr_threshold = corr_thresholds[1]

    def measure_correlation(self):
        covar_matrix = np.cov(self._columns)
        strong_corr_inputs = []
        weak_corr_inputs = []
        for row_nr, row in enumerate(covar_matrix):
            for entry_nr, entry in enumerate(row):
                if row_nr > entry_nr:
                    if abs(entry) >= self._strong_corr_threshold:
                        strong_corr_inputs.append(
                            [[str(row_nr), str(entry_nr)],
                             "{:.3f}".format(entry)])
                    elif abs(entry) >= self._weak_corr_threshold:
                        weak_corr_inputs.append(
                            [[str(row_nr), str(entry_nr)],
                             "{:.3f}".format(entry)])
        if len(strong_corr_inputs) > 0 or len(weak_corr_inputs) > 0:
            print(("Caution!\nCorrelation between input data can affect the "
                  + "reliability of the importance measure.\n"
                  + "Strong correlations of more than {} "
                  + "were found between {} pair(s) of input variables:\n\t{}\n"
                  + "Additionally, weak correlations of more than "
                  + "{} were found between {} pair(s) of input variables:\n\t")
                  .format(
                  self._strong_corr_threshold),
                  len(strong_corr_inputs),
                  self._weak_corr_threshold,
                  len(weak_corr_inputs),
                  "\n\t".join([": ".join([",".join(subentry)
                               if isinstance(subentry, list) else subentry
                               for subentry in entry])
                               for entry in strong_corr_inputs]),
                  "\n\t".join([": ".join([",".join(subentry)
                               if isinstance(subentry, list) else subentry
                               for subentry in entry])
                               for entry in weak_corr_inputs]))
        else:
            print("No correlation above {} found between the inputs."
                  .format(weak_corr_threshold))
        return strong_corr_inputs, weak_corr_inputs
