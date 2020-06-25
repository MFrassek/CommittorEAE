import numpy as np

class pB_Approximator():
	@staticmethod
	def approximate_pBs(grid_snapshots, labels, weights):
		weighted_label_dict = {}
		weight_dict = {}
		for snapshot_nr in range(len(grid_snapshots)):
			gridpoint_tuple = tuple(grid_snapshots[snapshot_nr])
			if gridpoint_tuple in weighted_label_dict:
				weighted_label_dict[gridpoint_tuple] \
					+= weights[snapshot_nr] * labels[snapshot_nr]
				weight_dict[gridpoint_tuple] += weights[snapshot_nr]
			else:
				weighted_label_dict[gridpoint_tuple] \
					= weights[snapshot_nr] * labels[snapshot_nr]
				weight_dict[gridpoint_tuple] = weights[snapshot_nr]
		pB_dict = {key: weighted_label_dict[key] / weight_dict[key] \
			for key in weight_dict}
		pBs = [pB_dict[tuple(key)] for key in grid_snapshots]
		pB_weights = [weight_dict[tuple(key)] for key in grid_snapshots]
		return pB_dict, np.array(pBs), pB_weights