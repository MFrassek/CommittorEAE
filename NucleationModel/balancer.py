import numpy as np
from collections import Counter
from gridifier import Gridifier


class Balancer():
    @staticmethod
    def hypercube_balance(snapshots, bins):
        gridified_snapshots = gridify_snapshots(snapshots, bins)
        tuple_gridified_snapshots = list(map(tuple, gridified_snapshots))
        snapshot_len = len(snapshots)
        return get_balanced_weights_from_list(
            snapshot_len, tuple_gridified_snapshots)

    @staticmethod
    def multidim_balance(snapshots, bins):
        gridified_snapshots = Balancer.gridify_snapshots(snapshots, bins)
        gridified_columns = np.transpose(gridified_snapshots)
        snapshot_len = len(snapshots)
        md_balanced_weights = np.ones(snapshot_len)
        for column in gridified_columns:
            Balancer.update_balanced_weights_based_on_column(
                md_balanced_weights, snapshot_len, column)
        md_balanced_weights /= np.mean(md_balanced_weights)
        return md_balanced_weights

    @staticmethod
    def pB_balance(pBs, bins):
        pBs_len = len(pBs)
        round_pBs = np.ceil((np.array(pBs) * (bins + 1)))
        return get_balanced_weights_from_list(pBs_len, round_pBs)


def gridify_snapshots(snapshots, bins):
    gridifier = Gridifier(snapshots, bins)
    return gridifier.gridify_snapshots(snapshots)


def update_balanced_weights_based_on_column(
        balanced_weights, snapshot_len, column):
    balanced_weights *= get_balanced_weights_from_list(
            snapshot_len, column)


def get_balanced_weights_from_list(snapshot_len, list_in_need_of_weights):
    return get_weights_from_balanced_counter(
        get_balanced_counter(
            snapshot_len, list_in_need_of_weights),
        list_in_need_of_weights)


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
