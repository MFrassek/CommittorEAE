import numpy as np

class PathData:
	def __init__(
			self, paths, path_labels, path_weights, 
			path_names, path_type_labels: list, path_type_weights: list):
		self._paths = paths
		self._path_labels = path_labels
		self._path_weights = path_weights
		self._path_names = path_names
		# sets the labels for each type of path. 
		# AB and BA path labels will be overwritten if progress labels are use
		self._AA_label = path_type_labels[0]
		self._AB_label = path_type_labels[1]
		self._BA_label = path_type_labels[2]
		self._BB_label = path_type_labels[3]
		# determines the weight each type of path should 
		# have on the total contribution to the model
		self._AA_weight = path_type_weights[0] / sum(path_type_weights)
		self._AB_weight = path_type_weights[1] / sum(path_type_weights)
		self._BA_weight = path_type_weights[2] / sum(path_type_weights)
		self._BB_weight = path_type_weights[3] / sum(path_type_weights)

	def __str__(self):
		return "Paths (cnt): {}".format(len(self._paths))

	@property
	def path_cnt(self):
		return len(self._paths)

	@property
	def paths(self):
		return self._paths
	
	@property
	def path_labels(self):
		return self._path_labels

	@property
	def path_weights(self):
		return self._path_weights
	
	def offset_paths(self, offset):
		"""Generates two sets of truncated paths that are offset 
		by 'offset' from each other.
		The paths of the first set get truncated at their ends 
		by n = 'offset' snapshots, while the paths of the second 
		set are truncated in their beginnings."""
		assert offset > 0 and isinstance(offset, int), \
			"offset needs to be chosen as a positive integer"
		return np.array([path[:-offset] for path in self._paths]), \
			np.array([path[offset:] for path in self._paths])
	
	def snapshots_labels_weights(
			self, offset, progress, 
			transitioned, turnedback):
		"""Generates snapshots, snapshot_labels and snapshot_weights 
		for each path type (AA, AB, BA, BB).
		params:
			offset: int
				Used to generate two snapshots that are offset 
				by n = 'offset' snapshots. 
				snapshot_labels and snapshot_weights are defined 
				based on the "later" snapshots
			progress: bool
				Defines whether the preset labels self._AB_label and 
				self._BA_label are used, or whether
				the labels are calculated as a progress along the 
				transition path. If progress == True
				the first snapshot of a trajectory will be assigned the 
				label of the state the path start from
				and the last one will be assigned the label of the states 
				it ends in, with the labels of
				all other snapshots being mapped in between linearly
			transitioned: bool
				Determines whether transition paths (AB and BA) 
				should be considered
			turnedback: bool
				Determines whether paths returning to their starting 
				state (AA and BB) should be considered
		"""

		AA_past_snapshots = []
		BB_past_snapshots = []
		AA_snapshots = []
		BB_snapshots = []
		AA_snapshot_labels = []
		BB_snapshot_labels = []
		AA_snapshot_weights = []
		BB_snapshot_weights = []

		AB_past_snapshots = []
		BA_past_snapshots = []
		AB_snapshots = []
		BA_snapshots = []
		AB_snapshot_labels = []
		BA_snapshot_labels = []
		AB_snapshot_weights = []
		BA_snapshot_weights = []
		
		for path_nr in range(len(self._paths)):
			# iterates over all indices within paths and uses the index to 
			# assign the current path, path_label and path_weight
			path = self._paths[path_nr]
			path_label = self._path_labels[path_nr]
			path_weight = self._path_weights[path_nr]
			for snapshot_nr in range(offset, len(path)):
				# iterates over all indices within each path and appends 
				# accordingly the snapshot as well as snapshot_label 
				# and snapshot_weight
				if turnedback:
					# AA and BB paths are only filled if turnedback == True. 
					# Allows the generation of a dataset consisting only 
					# of (AA and BB) or (AB and BA) paths
					if path_label == "AA":
						AA_snapshots.append(path[snapshot_nr])
						AA_snapshot_labels.append(self._AA_label)
						AA_snapshot_weights.append(path_weight)
					if path_label == "BB":
						BB_snapshots.append(path[snapshot_nr])
						BB_snapshot_labels.append(self._BB_label)
						BB_snapshot_weights.append(path_weight)
				if transitioned:
					if path_label == "AB":
						AB_snapshots.append(path[snapshot_nr])
						if progress:
							# Calculate the progess label in such a way, 
							# that the first snapshot of the current path 
							# is assigned the same label as AA paths,
							# the last snapshot is assigned the same label as 
							# BB paths and all other snapshot labels are 
							# mapped linearly in between.
							AB_snapshot_labels.append(
								((self._BB_label - self._AA_label) \
								* (snapshot_nr + offset) \
								/ (len(path) - 1.0 + offset)) \
								+ self._AA_label)
						else:
							AB_snapshot_labels.append(self._AB_label)
						AB_snapshot_weights.append(path_weight)
					if path_label == "BA":
						BA_snapshots.append(path[snapshot_nr])
						if progress:
							# corresponding to labels of AB paths, but 
							# starting with the label ob BB paths and 
							# ending with the label of AA paths
							BA_snapshot_labels.append(((self._BB_label \
								- self._AA_label) * (len(trajectory) \
								- (snapshot_nr + offset + 1)) \
								/ (len(trajectory) + offset - 1)) \
								+ self._AA_label)
						else:
							BA_snapshot_labels.append(self._BA_label)
						BA_snapshot_weights.append(path_weight)

			if offset > 0:
				# generates the past_snapshots only is offset is > 0, 
				# by truncating each path by the last 'offset' snapshots 
				# and subsequently appending the remainin snapshots 
				# to the past_snapshots
				for snapshot_nr in range(len(path)-offset):
					if turnedback:
						if path_label == "AA":
							AA_past_snapshots.append(path[snapshot_nr])
						if path_label == "BB":
							BB_past_snapshots.append(path[snapshot_nr])
					if transitioned:
						if path_label == "AB":
							AB_past_snapshots.append(path[snapshot_nr])
						if path_label == "BA":
							BA_past_snapshots.append(path[snapshot_nr])

		if offset == 0:
			# If the offset is 0, past_snapshots and regular snapshots 
			# are identical. A copying process is therefore sufficient.
			AA_past_snapshots = [*AA_snapshots]
			BA_past_snapshots = [*AB_snapshots]
			AB_past_snapshots = [*BA_snapshots]
			BB_past_snapshots = [*BB_snapshots]

		all_snapshot_cnt = len(AA_snapshots) + len(AB_snapshots) \
			+ len(BA_snapshots) + len(BA_snapshots)
		
		all_snapshot_weight_mean = np.mean(
			AA_snapshot_weights \
			+ AB_snapshot_weights \
			+ BA_snapshot_weights \
			+ BB_snapshot_weights)

		print("Mean weights: {}".format(all_snapshot_weight_mean))
		print("Sum weights AA: {}\t Sum weights AB: {}"\
			.format(sum(AA_snapshot_weights), sum(AB_snapshot_weights)))
		# # Normalize all weights of considered paths.
		# # Assign all snapshots of one path type equal weight, and modify
		# # that by the corresponding path type weight
		# # then normalize with the mean of all snapshots weights.
		# if len(AA_snapshots) > 0:
		# 	AA_snapshot_weights = np.array(AA_snapshot_weights) \
		# 		* (all_snapshot_cnt/len(AA_snapshots)) * self._AA_weight \
		# 		/ all_snapshot_weight_mean 
		# if len(AB_snapshots) > 0:
		# 	AB_snapshot_weights = np.array(AB_snapshot_weights) \
		# 		* (all_snapshot_cnt/len(AB_snapshots)) * self._AB_weight \
		# 		/ all_snapshot_weight_mean 
		# if len(BA_snapshots) > 0:
		# 	BA_snapshot_weights = np.array(BA_snapshot_weights) \
		# 		* (all_snapshot_cnt/len(BA_snapshots)) * self._BA_weight \
		# 		/ all_snapshot_weight_mean 
		# if len(BB_snapshots) > 0:
		# 	BB_snapshot_weights = np.array(BB_snapshot_weights) \
		# 		* (all_snapshot_cnt/len(BB_snapshots)) * self._BB_weight \
		# 		/ all_snapshot_weight_mean 

		print("Sum weights AA after: {}\t Sum weights AB after: {}"\
			.format(sum(AA_snapshot_weights), sum(AB_snapshot_weights)))

		return np.array(AA_past_snapshots), np.array(AB_past_snapshots), \
				np.array(BA_past_snapshots), np.array(BB_past_snapshots), \
				np.array(AA_snapshots), np.array(AB_snapshots), \
				np.array(BA_snapshots), np.array(BB_snapshots), \
				np.array(AA_snapshot_labels), np.array(AB_snapshot_labels), \
				np.array(BA_snapshot_labels), np.array(BB_snapshot_labels), \
				np.array(AA_snapshot_weights), np.array(AB_snapshot_weights), \
				np.array(BA_snapshot_weights), np.array(BB_snapshot_weights)

		# return np.array(AA_past_snapshots), np.array(AB_past_snapshots), \
		# 		np.array(BA_past_snapshots), np.array(BB_past_snapshots), \
		# 		np.array(AA_snapshots), np.array(AB_snapshots), \
		# 		np.array(BA_snapshots), np.array(BB_snapshots), \
		# 		np.array(AA_snapshot_labels), np.array(AB_snapshot_labels), \
		# 		np.array(BA_snapshot_labels), np.array(BB_snapshot_labels), \
		# 		AA_snapshot_weights, AB_snapshot_weights, \
		# 		BA_snapshot_weights, BB_snapshot_weights
