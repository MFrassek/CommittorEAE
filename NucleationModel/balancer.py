import numpy as np
from collections import Counter
from gridifier import Gridifier


class Balancer():
    @staticmethod
    def hypercube_balance(snapshots, bins):
        gridified_snapshots = Balancer.gridify_snapshots(snapshots, bins)
        # Turn into tuples to be hashable
        tuple_round_snapshots = list(map(tuple, gridified_snapshots))
        snapshot_len = len(snapshots)
        balanced_counter = \
            Balancer.get_balanced_counter(snapshot_len, tuple_round_snapshots)
        return Balancer.get_weights_from_balanced_counter(
            balanced_counter, tuple_round_snapshots)

    @staticmethod
    def multidim_balance(snapshots, bins):
        gridified_snapshots = Balancer.gridify_snapshots(snapshots, bins)
        snapshot_len = len(snapshots)
        md_balanced_weights = np.ones(snapshot_len)
        gridified_columns = np.transpose(gridified_snapshots)
        for column in gridified_columns:
            balanced_counter = \
                Balancer.get_balanced_counter(snapshot_len, column)
            md_balanced_weights *= Balancer.get_weights_from_balanced_counter(
                    balanced_counter, column)
        md_balanced_weights /= np.mean(md_balanced_weights)
        return md_balanced_weights

    def gridify_snapshots(snapshots, bins):
        gridifier = Gridifier(snapshots, bins)
        return gridifier.gridify_snapshots(snapshots)

    def get_balanced_counter(snapshot_len, list_in_need_of_weights):
        """Balance weights such that all weight together sum to snapshot_len
        and the weights for each key sum together to snapshot_len / counter_len
        """
        counter = Counter(list_in_need_of_weights)
        counter_len = len(counter)
        return {key: snapshot_len / (label * counter_len)
                for key, label in counter.items()}

    def get_weights_from_balanced_counter(counter, list_in_need_of_weights):
        return np.array([counter[i] for i in list_in_need_of_weights])
