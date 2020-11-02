import numpy as np


class pB_Approximator():
    @staticmethod
    def approximate_pBs(grid_snapshots, labels, weights):
        weighted_label_dict = {}
        weight_dict = {}
        for snapshot_nr, snapshot in enumerate(grid_snapshots):
            gridpoint_tuple = tuple(snapshot)
            weight = weights[snapshot_nr]
            label = labels[snapshot_nr]
            if gridpoint_tuple in weighted_label_dict:
                update_dictionary_entries(
                    weighted_label_dict, weight_dict,
                    gridpoint_tuple, weight, label)
            else:
                add_dictionary_entries(
                    weighted_label_dict, weight_dict,
                    gridpoint_tuple, weight, label)
        pB_dict = {key: weighted_label_dict[key] / weight_dict[key]
                   for key in weight_dict}
        pBs = [pB_dict[tuple(key)] for key in grid_snapshots]
        pB_weights = [weight_dict[tuple(key)] for key in grid_snapshots]
        return pB_dict, np.array(pBs), pB_weights


def update_dictionary_entries(
        weighted_label_dict, weight_dict, gridpoint_tuple, weight, label):
    weighted_label_dict[gridpoint_tuple] += weight * label
    weight_dict[gridpoint_tuple] += weight


def add_dictionary_entries(
        weighted_label_dict, weight_dict, gridpoint_tuple, weight, label):
    weighted_label_dict[gridpoint_tuple] = weight * label
    weight_dict[gridpoint_tuple] = weight
