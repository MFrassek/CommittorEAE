import sys
import math
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import copy
import pickle
from os import listdir


def get_size(obj, seen=None):
    """Recursively finds size of objects
    Copied from https://goshippo.com/blog/measure-real-size-any-python-object/
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') \
            and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def make_halfpoint_divided_colormap(logvmin):
    resolution = 1001
    bandwidth = 0.1
    lower_bound_halfpoint = math.log(0.5-bandwidth/2, 10)/math.log(logvmin, 10)
    lower_bound_halfpoint_int = round(lower_bound_halfpoint*resolution)
    upper_bound_halfpoint = math.log(0.5+bandwidth/2, 10)/math.log(logvmin, 10)
    upper_bound_halfpoint_int = round(upper_bound_halfpoint*resolution)
    bottom = cm.get_cmap("summer", resolution)
    middle = cm.get_cmap("Greys", 10)
    top = cm.get_cmap("summer", resolution)
    c_map = ListedColormap(np.vstack((
        bottom(np.linspace(
            0,
            1 - lower_bound_halfpoint,
            resolution - lower_bound_halfpoint_int)),
        middle(np.linspace(
            0.9,
            1.0,
            lower_bound_halfpoint_int - upper_bound_halfpoint_int)),
        top(np.linspace(
            1 - upper_bound_halfpoint,
            1,
            upper_bound_halfpoint_int)))), "SplitSummer")
    c_map.set_under([0.2, 0.9, 0.5, 0.9])
    c_map.set_over([1, 0.7, 0.1, 0.9])
    return c_map


def make_png_with_bad_as_transparent_colormap():
    c_map = copy.copy(plt.cm.get_cmap('gray'))
    c_map.set_bad(alpha=0)
    return c_map


def function_to_str(function):
    return str(function).split(" ")[1]


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


def print_coverage(list_var_names, dataset):
    AAs = [snapshot for i, snapshot in enumerate(dataset.snapshots)
           if dataset.labels[i] == 0]
    ABs = [snapshot for i, snapshot in enumerate(dataset.snapshots)
           if dataset.labels[i] == 1]
    AACoverageOfABRange = np.amax(AAs, axis=0) / np.amax(ABs, axis=0) * 100
    plt.bar(range(len(list_var_names)), AACoverageOfABRange)
    plt.ylim(0,)
    plt.xticks(range(len(list_var_names)), list_var_names, rotation=60)
    plt.ylabel("AA/AB path range coverage [%]")
    plt.title("Fraction of AB path range also covered by AA paths ")
    plt.tight_layout()
    plt.savefig("results/AA-AB_coverage.png")
    plt.show()
    for i, _ in enumerate(AACoverageOfABRange):
        print("{}: {}".format(list_var_names[i], AACoverageOfABRange[i]))


def get_relative_encoder_importances(encoder, reduced_list_var_names):
    """Return the relative importances of inputs in a linear encoder."""
    formula_components = \
        get_encoder_formula_components(encoder, reduced_list_var_names)
    absolute_formula_components = \
        make_components_absolute(formula_components)
    normalized_absolute_formula_component = \
        make_components_normalized(absolute_formula_components)
    return normalized_absolute_formula_component


def get_encoder_formula_components(encoder, reduced_list_var_names):
    """Calculates the linear formula represented by the encoder."""
    bn_size = encoder.layers[-1].output_shape[1]
    formula_components = [[] for _ in range(bn_size)]
    in_size = encoder.layers[0].output_shape[0][1]
    base_predictions = encoder.predict([[np.zeros(in_size)]])[0]
    for dimension in range(in_size):
        mod_list = np.zeros(in_size)
        mod_list[dimension] = 1
        mod_predictions = encoder.predict([[mod_list]])[0]
        for i, _ in enumerate(mod_predictions):
            formula_components[i].append(
                mod_predictions[i] - base_predictions[i])
    return list(zip(reduced_list_var_names, *formula_components))


def make_components_absolute(formula_components):
    absolute_components = []
    for component in formula_components:
        absolute_components.append((
            component[0],
            *list(map(abs, component[1:]))))
    return absolute_components


def make_components_normalized(formula_components):
    normalized_components = []
    sum_list = []
    for column in np.transpose(formula_components)[1:]:
        sum_list.append(sum(list(map(float, column))))
    for component in formula_components:
        normalized_components.append((
            component[0],
            *list(map(lambda x: (x / sum_list)[0], component[1:]))))
    return normalized_components


def flatten_list_of_lists(list_of_lists):
    return [y for x in list_of_lists for y in x]


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


def calculate_slope_MCG_BigCage(x):
    return 68.14 - 0.4286*x


def calculate_slope_now_BigCage(x):
    return 30 - 0.0553*x


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
