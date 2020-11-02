import numpy as np


class pB_Approximator():
    @staticmethod
    def approximate_pBs(grid_snapshots, labels, weights):
        weighted_label_dict, weight_dict = \
            get_weighted_label_dict_and_weight_dict(
                grid_snapshots, labels, weights)
        pB_dict = get_pB_dict(weighted_label_dict, weight_dict)
        pBs = get_pBs_list(pB_dict, grid_snapshots)
        pB_weights = get_pB_weight_list(weight_dict, grid_snapshots)
        return pB_dict, pBs, pB_weights


def get_weighted_label_dict_and_weight_dict(grid_snapshots, labels, weights):
    weighted_label_dict = {}
    weight_dict = {}
    for snapshot, label, weight in zip(grid_snapshots, labels, weights):
        gridpoint_tuple = tuple(snapshot)
        if gridpoint_tuple in weighted_label_dict:
            update_dictionary_entries(
                weighted_label_dict, weight_dict,
                gridpoint_tuple, weight, label)
        else:
            add_dictionary_entries(
                weighted_label_dict, weight_dict,
                gridpoint_tuple, weight, label)
    return weighted_label_dict, weight_dict


def update_dictionary_entries(
        weighted_label_dict, weight_dict, gridpoint_tuple, weight, label):
    weighted_label_dict[gridpoint_tuple] += weight * label
    weight_dict[gridpoint_tuple] += weight


def add_dictionary_entries(
        weighted_label_dict, weight_dict, gridpoint_tuple, weight, label):
    weighted_label_dict[gridpoint_tuple] = weight * label
    weight_dict[gridpoint_tuple] = weight


def get_pB_dict(weighted_label_dict, weight_dict):
    return {key: weighted_label_dict[key] / weight_dict[key]
            for key in weight_dict}


def get_pBs_list(pB_dict, grid_snapshots):
    return np.array([pB_dict[tuple(key)] for key in grid_snapshots])


def get_pB_weight_list(weight_dict, grid_snapshots):
    return [weight_dict[tuple(key)] for key in grid_snapshots]
