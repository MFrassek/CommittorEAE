import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import listdir


def merge_all_OPS_simulation_pickle_files(folder_name):
    file_names = listdir(folder_name)
    all_paths = []
    all_labels = []
    for file_name in file_names:
        if file_name.endswith("paths.p"):
            paths = np.array(
                pickle.load(open("{}/{}".format(
                    folder_name, file_name), "rb")))
            all_paths = np.append(all_paths, paths)
        if file_name.endswith("labels.p"):
            labels = np.array(
                pickle.load(open("{}/{}".format(
                    folder_name, file_name), "rb")))
            all_labels = np.append(all_labels, labels)
    pickle.dump(all_paths, open(
        "{}/all/paths.p".format(folder_name), "wb"))
    pickle.dump(all_labels, open(
        "{}/all/labels.p".format(folder_name), "wb"))
    return all_paths, all_labels


def discard_unwanted_dimensions_from_pickle_files(folder_name, keep_first_n):
    paths = np.array(
        pickle.load(open("{}/paths.p".format(folder_name), "rb")))
    truncated_paths = [np.transpose(np.transpose(path)[:keep_first_n])
                       for path in paths]
    pickle.dump(
        truncated_paths, open("{}/trunc_paths.p".format(folder_name), "wb"))
