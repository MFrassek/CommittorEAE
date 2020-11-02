import numpy as np
from collections import Counter
from gridifier import Gridifier


class Balancer():
    @staticmethod
    def hypercube_balance(snapshots, bins):
        gridified_snapshots = Balancer.gridify_snapshots(snapshots, bins)
        # Turn into tuples to be hashable
        tuple_round_snapshots = list(map(tuple, gridified_snapshots))
        counter = Counter(tuple_round_snapshots)
        counter_len = len(counter)
        # Balance weights such that all weight together sum to
        # snapshot_len and the weights for each key sum together to
        # snapshot_len/counter_len
        snapshot_len = len(snapshots)
        balanced_counter = {key: snapshot_len / (label * counter_len)
                            for key, label in counter.items()}
        hc_balanced_weights = np.array(
            [balanced_counter[i] for i in tuple_round_snapshots])
        return hc_balanced_weights

    @staticmethod
    def multidim_balance(snapshots, bins):
        gridified_snapshots = Balancer.gridify_snapshots(snapshots, bins)
        snapshot_len = len(snapshots)
        hc_balanced_weights = np.ones(snapshot_len)
        round_columns = np.transpose(gridified_snapshots)
        for column in round_columns:
            counter = Counter(column)
            counter_len = len(counter)
            # Balance weights such that all weight together sum to
            # snapshot_len and the weights for each key sum together to
            # snapshot_len / counter_len
            balanced_counter = {key: snapshot_len / (label * counter_len)
                                for key, label in counter.items()}
            col_balanced_weights = np.array(
                [balanced_counter[i] for i in column])
            hc_balanced_weights *= col_balanced_weights
        hc_balanced_weights /= np.mean(hc_balanced_weights)
        return hc_balanced_weights

    def gridify_snapshots(snapshots, bins):
        gridifier = Gridifier(snapshots, bins)
        return gridifier.gridify_snapshots(snapshots)
