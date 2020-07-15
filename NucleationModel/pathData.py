import numpy as np


class PathData:
    def __init__(
            self, paths, path_labels, path_weights,
            path_names, path_type_labels: list, path_type_weights: list):
        self._paths = paths
        self._path_labels = path_labels
        self._path_weights = path_weights
        self._path_names = path_names
        # sets the labels for each type of path.
        # AB and BA path labels will be overwritten if progress labels are use
        self._AA_label = path_type_labels[0]
        self._AB_label = path_type_labels[1]
        self._BA_label = path_type_labels[2]
        self._BB_label = path_type_labels[3]
        # determines the weight each type of path should
        # have on the total contribution to the model
        self._AA_weight = path_type_weights[0] / sum(path_type_weights)
        self._AB_weight = path_type_weights[1] / sum(path_type_weights)
        self._BA_weight = path_type_weights[2] / sum(path_type_weights)
        self._BB_weight = path_type_weights[3] / sum(path_type_weights)

    def __str__(self):
        return "Paths (cnt): {}".format(len(self._paths))

    @property
    def path_snapshot_cnt(self):
        return len(self._paths)

    @property
    def paths(self):
        return self._paths

    @property
    def path_labels(self):
        return self._path_labels

    @property
    def path_weights(self):
        return self._path_weights

    def snapshots_labels_weights(
            self, progress,
            transitioned, turnedback):
        """Generates snapshots, labels and weights
        for each path type (AA, AB, BA, BB).
        params:
            progress: bool
                Defines whether the preset labels self._AB_label and
                self._BA_label are used, or whether
                the labels are calculated as a progress along the
                transition path. If progress == True
                the first snapshot of a trajectory will be assigned the
                label of the state the path start from
                and the last one will be assigned the label of the states
                it ends in, with the labels of
                all other snapshots being mapped in between linearly
            transitioned: bool
                Determines whether transition paths (AB and BA)
                should be considered
            turnedback: bool
                Determines whether paths returning to their starting
                state (AA and BB) should be considered
        """

        AA_snapshots = []
        BB_snapshots = []
        AA_labels = []
        BB_labels = []
        AA_weights = []
        BB_weights = []

        AB_snapshots = []
        BA_snapshots = []
        AB_labels = []
        BA_labels = []
        AB_weights = []
        BA_weights = []

        for path_nr, path in enumerate(self._paths):
            # iterates over all indices within paths and uses the index to
            # assign the current path, path_label and path_weight
            path_label = self._path_labels[path_nr]
            path_weight = self._path_weights[path_nr]
            for snapshot_nr in range(len(path)):
                # iterates over all indices within each path and appends
                # accordingly the snapshot as well as label
                # and weight
                if turnedback:
                    # AA and BB paths are only filled if turnedback == True.
                    # Allows the generation of a dataset consisting only
                    # of (AA and BB) or (AB and BA) paths
                    if path_label == "AA":
                        AA_snapshots.append(path[snapshot_nr])
                        AA_labels.append(self._AA_label)
                        AA_weights.append(path_weight)
                    if path_label == "BB":
                        BB_snapshots.append(path[snapshot_nr])
                        BB_labels.append(self._BB_label)
                        BB_weights.append(path_weight)
                if transitioned:
                    if path_label == "AB":
                        AB_snapshots.append(path[snapshot_nr])
                        if progress:
                            # Calculate the progess label in such a way,
                            # that the first snapshot of the current path
                            # is assigned the same label as AA paths,
                            # the last snapshot is assigned the same label as
                            # BB paths and all other snapshot labels are
                            # mapped linearly in between.
                            AB_labels.append(
                                ((self._BB_label - self._AA_label)
                                 * (snapshot_nr) / (len(path) - 1.0))
                                + self._AA_label)
                        else:
                            AB_labels.append(self._AB_label)
                        AB_weights.append(path_weight)
                    if path_label == "BA":
                        BA_snapshots.append(path[snapshot_nr])
                        if progress:
                            # corresponding to labels of AB paths, but
                            # starting with the label ob BB paths and
                            # ending with the label of AA paths
                            BA_labels.append(
                                ((self._BB_label - self._AA_label)
                                 * (len(trajectory) - (snapshot_nr + 1))
                                 / (len(trajectory) - 1)) + self._AA_label)
                        else:
                            BA_labels.append(self._BA_label)
                        BA_weights.append(path_weight)

        all_snapshot_cnt = len(AA_snapshots) + len(AB_snapshots) \
            + len(BA_snapshots) + len(BA_snapshots)

        all_weight_mean = np.mean(
            AA_weights
            + AB_weights
            + BA_weights
            + BB_weights)

        print("Mean weights: {}".format(all_weight_mean))
        print("Sum weights AA: {}\t Sum weights AB: {}"
              .format(sum(AA_weights), sum(AB_weights)))
        print("Sum weights AA after: {}\t Sum weights AB after: {}"
              .format(sum(AA_weights), sum(AB_weights)))

        return np.array(AA_snapshots), np.array(AB_snapshots), \
            np.array(BA_snapshots), np.array(BB_snapshots), \
            np.array(AA_labels), np.array(AB_labels), \
            np.array(BA_labels), np.array(BB_labels), \
            np.array(AA_weights), np.array(AB_weights), \
            np.array(BA_weights), np.array(BB_weights)
