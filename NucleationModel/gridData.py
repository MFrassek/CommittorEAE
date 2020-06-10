import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class GridData():
	def __init__(self, resolution):
		self._resolution = resolution
	# 	self._minima = minima
	# 	self._maxima = maxima
	# 	self._snapshots = snapshots
	# 	self._labels = labels
	# 	self._weights = weights
	# 	self._dimensions = len(self._minima)
	# 	self._snapshot_cnt = len(self.snapshots)

	# def __str__(self):
	# 	return "{}-dimensional {}-grid for {} snapshots."\
	# 		.format(self._dimensions, self._resolution, self._snapshot_cnt)
	# 	#"{}".format(self.resolution) \
	# 	#	+(len(self.minima)-1)*"x{}".format(self.resolution) \
	# 	#	+ " grid for {} snapshots"\
	# 	#		.format(self.resolution, self.resolution, len(self.snapshots))

	@property
	def resolution(self):
		return self._resolution
	# @property
	# def minima(self):
	# 	return self._minima
	# @property
	# def maxima(self):
	# 	return self._maxima
	# @property
	# def snapshots(self):
	# 	return self._snapshots
	# @property
	# def labels(self):
	# 	return self._labels
	# @property
	# def weights(self):
	# 	return self._weights
	# @property
	# def dimensions(self):
	# 	return self._dimensions
	# @property
	# def snapshot_cnt(self):
	# 	return self._snapshot_cnt

	# def gridify(self, snapshots, minima, maxima):
	# 	"""Take a list of snapshots and round all entries
	# 	to the closest gridpoint.
	# 	The positions of the gridpoints are
	# 	determined by the chosen resolution as well as the minima and
	# 	maxima of the snapshots.
	# 	"""
	# 	dimensions = len(minima)
	# 	# Calculate the inverse of the spans to allow for 
	# 	# multiplication instead of division. 
	# 	# Then multiply with the resolution, to reduce the 
	# 	# number of multiplications later.		
	# 	inv_spans_res = np.array([1 / (maxima[i] - minima[i]) * \
	# 		(self._resolution - 1) for i in range(dimensions)])
	# 	# Use broadcasting to shift (- np.array(minima)) and rescale
	# 	# (* inv_spans_res) the entries in snapshots.
	# 	# Then increase all values by 0.5 and floor them to have an
	# 	# efficient way of rounding.
	# 	grid_snaps = np.floor((snapshots - np.array(minima)) \
	# 		* inv_spans_res + 0.5)
	# 	return grid_snaps

	

	def plot_distribution(
			self, grid_snapshots, max_row_len, 
			subfig_size, var_names, file_name):

		cols = np.transpose(grid_snapshots)
		dimensions = len(grid_snapshots[0])
		# for col in cols:
		# 	plt.plot()
		# 	plt.hist(col, self._resolution)

		# 	plt.show()
		suptitle = "Distribution of input"
		row_cnt = ((dimensions-1)//max_row_len)+1
		fig, axs = plt.subplots(
			row_cnt, max_row_len,
			figsize=(
				subfig_size*max_row_len,
				subfig_size*row_cnt*1.3))
		fig.suptitle(
			suptitle, 
			fontsize=subfig_size*max_row_len*2, 
			y=1.04 - 0.04*row_cnt)				

		for i in range(dimensions):   
			if row_cnt > 1:
				new_axs = axs[i//max_row_len]
			else:
				new_axs = axs
			new_axs[i%max_row_len].tick_params(
				axis='both',
				which='both',
				top=False,
				bottom=False,
				labelbottom=False,
				left = False,
				labelleft= False)	
			im = new_axs[i%max_row_len]\
				.hist(cols[i], self._resolution)
			new_axs[i%max_row_len]\
				.set_xlabel("${}$".format(var_names[i]),fontsize=subfig_size*10)
		# if not all rows are filled 
		# remove the remaining empty subplots in the last row
		if dimensions%max_row_len != 0:
			for i in range(dimensions%max_row_len, max_row_len):
				new_axs[i].axis("off")
		plt.tight_layout(rect = [0, 0, 1, 0.8])
		plt.savefig("hist_{}_{}.png".format(file_name, self._resolution)) 
		plt.show()


	def approximate_pB(self, grid_snapshots, labels, weights):
		weighted_label_dict = {}
		weight_dict = {}
		print("Fill hash maps")
		for snapshot_nr in range(len(grid_snapshots)):
			gridpoint_tuple = tuple(grid_snapshots[snapshot_nr])
			try:
				weighted_label_dict[gridpoint_tuple] \
					+= weights[snapshot_nr] * labels[snapshot_nr]
				weight_dict[gridpoint_tuple] += weights[snapshot_nr]
			except:
				weighted_label_dict[gridpoint_tuple] \
					= weights[snapshot_nr] * labels[snapshot_nr]
				weight_dict[gridpoint_tuple] = weights[snapshot_nr]
		print("Rescale")
		pB_dict = {key: weighted_label_dict[key] / weight_dict[key] \
			for key in weight_dict}
		pBs = [pB_dict[tuple(key)] for key in grid_snapshots]
		# return pB_dict, weighted_label_dict, weight_dict
		return pB_dict, np.array(pBs)
	