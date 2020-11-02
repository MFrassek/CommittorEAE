import numpy as np


class pB_Approximator():
    @staticmethod
    def approximate_pBs(grid_snapshots, labels, weights):
        weighted_label_dict, weight_dict = \
            get_weighted_label_dict_and_weight_dict(
                grid_snapshots, labels, weights)
        pB_dict = {key: weighted_label_dict[key] / weight_dict[key]
                   for key in weight_dict}
        pBs = [pB_dict[tuple(key)] for key in grid_snapshots]
        pB_weights = [weight_dict[tuple(key)] for key in grid_snapshots]
        return pB_dict, np.array(pBs), pB_weights


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
