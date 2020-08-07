import sys
import math
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats


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


def get_all_1D_correlated_representations(grid_snapshots, function):
    all_representations = {}
    for x in range(len(grid_snapshots[0])):
        print(x)
        x_representation = get_1D_correlated_representation(
            grid_snapshots=grid_snapshots,
            x_int=x,
            function=function)
        x_representation = np.array(
            [value for value in x_representation.values()])
        all_representations[x] = x_representation
    return all_representations


def get_1D_correlated_representation(grid_snapshots, x_int, function):
    snapshots_per_x = {}
    for snapshot in grid_snapshots:
        # Append to key associated with this snapshots value for x
        try:
            snapshots_per_x[snapshot[x_int]].append(tuple(snapshot))
        except Exception:
            snapshots_per_x[snapshot[x_int]] = [tuple(snapshot)]
    means_per_x = {}
    for x in snapshots_per_x:
        means_per_x[x] = function(snapshots_per_x[x])
    return means_per_x


def get_all_2D_correlated_representations(grid_snapshots, function):
    all_representations = {}
    for x in range(len(grid_snapshots[0])):
        for y in range(0, x):
            print(x, y)
            xy_representation = get_2D_correlated_representation(
                grid_snapshots=grid_snapshots,
                x_int=x,
                y_int=y,
                function=function)
            xy_representation = np.array(
                [value for value in xy_representation.values()])
            all_representations[(x, y)] = xy_representation
    return all_representations


def get_2D_correlated_representation(grid_snapshots, x_int, y_int, function):
    snapshots_per_xy = {}
    for snapshot in grid_snapshots:
        # Append to key associated with this snapshots values for x and y
        try:
            snapshots_per_xy[(snapshot[x_int], snapshot[y_int])] \
                .append(tuple(snapshot))
        except Exception:
            snapshots_per_xy[(snapshot[x_int], snapshot[y_int])] \
                = [tuple(snapshot)]
    means_per_xy = {}
    for xy in snapshots_per_xy:
        means_per_xy[xy] = function(snapshots_per_xy[xy])
    return means_per_xy


def get_means_from_tuples(tuples):
    return np.mean(list(zip(*tuples)), axis=1)


def get_modes_from_tuples(tuples):
    return stats.mode(tuples)[0][0]
