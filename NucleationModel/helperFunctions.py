import sys
import math
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


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
    c_map.set_under([0.9, 0.9, 0.9, 1])
    c_map.set_over([0.9, 0.9, 0.9, 1])
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
