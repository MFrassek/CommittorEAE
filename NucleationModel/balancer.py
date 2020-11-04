import numpy as np
from collections import Counter
from gridifier import Gridifier


class Balancer():
    @staticmethod
    def hypercube_balance(snapshots, bins):
        gridified_snapshots = gridify_list_entries(snapshots, bins)
        tuple_gridified_snapshots = list(map(tuple, gridified_snapshots))
        return get_balanced_weights_from_list(tuple_gridified_snapshots)

    @staticmethod
    def multidim_balance(snapshots, bins):
        gridified_snapshots = gridify_list_entries(snapshots, bins)
        gridified_columns = np.transpose(gridified_snapshots)
        md_balanced_weights = np.ones(len(snapshots))
        for column in gridified_columns:
            update_balanced_weights_based_on_column(
                md_balanced_weights, column)
        md_balanced_weights /= np.mean(md_balanced_weights)
        return md_balanced_weights

    @staticmethod
    def pB_balance(pBs, bins):
        gridified_pBs = gridify_list_entries(pBs, bins)
        return get_balanced_weights_from_list(gridified_pBs)


def gridify_list_entries(lst, bins):
    gridifier = Gridifier(lst, bins)
    return gridifier.gridify_snapshots(lst)


def update_balanced_weights_based_on_column(
        balanced_weights, column):
    balanced_weights *= get_balanced_weights_from_list(column)


def get_balanced_weights_from_list(list_in_need_of_weights):
    return get_weights_from_balanced_counter(
        get_balanced_counter(list_in_need_of_weights),
        list_in_need_of_weights)


def get_balanced_counter(list_in_need_of_weights):
    """Balance weights such that all weight together sum to list_len
    and the weights for each key sum together to list_len / counter_len
    """
    list_len = len(list_in_need_of_weights)
    counter = Counter(list_in_need_of_weights)
    counter_len = len(counter)
    return {key: list_len / (label * counter_len)
            for key, label in counter.items()}


def get_weights_from_balanced_counter(counter, list_in_need_of_weights):
    return np.array([counter[i] for i in list_in_need_of_weights])
