import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import listdir


def get_all_ranges(datasets: list):
    ranges = [
        [np.float("inf"), np.float("-inf")]
        for i in datasets[0].past_snapshots[0]]
    dimensions = len(ranges)
    for dataset in datasets:
        for snapshot in dataset.past_snapshots:
            for dim in range(dimensions):
                if snapshot[dim] < ranges[dim][0]:
                    ranges[dim][0] = snapshot[dim]
                if snapshot[dim] > ranges[dim][1]:
                    ranges[dim][1] = snapshot[dim]
    return ranges


def store_model_weights(
        path, autoencoder, autoencoder_1, autoencoder_2,
        encoder, decoder_1, decoder_2):
    autoencoder.save_weights("{}/autoencoder".format(path))
    autoencoder_1.save_weights("{}/autoencoder_1".format(path))
    autoencoder_2.save_weights("{}/autoencoder_2".format(path))
    encoder.save_weights("{}/encoder".format(path))
    decoder_1.save_weights("{}/decoder_1".format(path))
    decoder_2.save_weights("{}/decoder_2".format(path))


def load_model_weights(
        path, autoencoder, autoencoder_1, autoencoder_2,
        encoder, decoder_1, decoder_2):
    autoencoder.load_weights("{}/autoencoder".format(path))
    autoencoder_1.load_weights("{}/autoencoder_1".format(path))
    autoencoder_2.load_weights("{}/autoencoder_2".format(path))
    encoder.load_weights("{}/encoder".format(path))
    decoder_1.load_weights("{}/decoder_1".format(path))
    decoder_2.load_weights("{}/decoder_2".format(path))
    return autoencoder, autoencoder_1, autoencoder_2, \
        encoder, decoder_1, decoder_2


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


def measure_correlation(snapshots, correlation_threshold):
    correlated_inputs = get_list_of_correlated_inputs(
        snapshots, correlation_threshold)
    if len(correlated_inputs) > 0:
        print(("Caution!\nCorrelation between input data can affect the "
              + "reliability of the importance measure.\n"
              + "Correlations of more than {} "
              + "were found between {} pair(s) of input variables:\n\t{}\n")
              .format(correlation_threshold,
              len(correlated_inputs),
              "\n\t".join([convert_correlation_list_entry_to_string(entry)
                           for entry in correlated_inputs])))
    else:
        print("No correlation above {} was found between the inputs."
              .format(correlation_threshold))
    return correlated_inputs


def get_list_of_correlated_inputs(snapshots, correlation_threshold):
    return [make_correlation_list_entry(row_nr, col_nr, entry)
            for row_nr, row in enumerate(get_covariance_matrix(snapshots))
            for col_nr, entry in enumerate(row)
            if row_nr > col_nr and abs(entry) >= correlation_threshold]


def get_covariance_matrix(snapshots):
    return np.cov(np.transpose(snapshots))


def make_correlation_list_entry(row_nr, col_nr, entry):
    return [str(row_nr), str(col_nr), "{:.3f}".format(entry)]


def convert_correlation_list_entry_to_string(entry):
    return "{},{}: {}".format(*entry)
