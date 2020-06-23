import sys
import math
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def get_size(obj, seen = None):
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
	elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
		size += sum([get_size(i, seen) for i in obj])
	return size

def make_halfpoint_divided_colormap(logvmin):
	resolution = 100
	halfpoint = math.log(0.5,10)/math.log(logvmin,10)
	halfpoint_int = round(halfpoint*resolution)
	bottom = cm.get_cmap("summer", resolution)
	middle = cm.get_cmap("Greys", 10)
	top = cm.get_cmap("summer", resolution)
	c_map = ListedColormap(np.vstack((
		bottom(np.linspace(
			0, 
			1 - halfpoint - (4 / resolution),
			resolution - halfpoint_int)),
		middle(np.linspace(
			0.9, 
			1.0, 
			4)),
		top(np.linspace(
			1 - halfpoint + (4 / resolution),
			1,
			halfpoint_int)))), "SplitSummer")
	return c_map

