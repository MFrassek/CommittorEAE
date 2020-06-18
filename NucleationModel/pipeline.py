from reducer import Reducer
from bounder import Bounder
from normalizer import Normalizer
from gridifier import Gridifier

import numpy as np

class Pipeline():
	def __init__(self, const, reduced_list_var_names, base_snapshots):
		self._const = const
		self._reduced_list_var_names = reduced_list_var_names
		self._dimensions = len(reduced_list_var_names)
		self._reducer = Reducer(
			self._reduced_list_var_names, 
			self._const.name_to_list_position)
		base_snapshots = self._reducer.reduce_snapshots(base_snapshots)
		self._bounder = Bounder(base_snapshots, self._const.outlier_cutoff)
		base_snapshots = self._bounder.bound_snapshots(base_snapshots)
		self._normalizer = Normalizer(base_snapshots)
		base_snapshots = self._normalizer.normalize_snapshots(base_snapshots)
		self._gridifier = Gridifier(base_snapshots, self._const.resolution)
		#base_snapshots = self._gridifier.gridify_snapshots(base_snapshots)

	def rbn(self, snapshots):
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		return snapshots

	def rbng(self, snapshots):
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		snapshots = self._gridifier.gridify_snapshots(snapshots)		
		return snapshots

	def plot_data(self, dataset):
		return self.rbng(dataset.train_past_snapshots), \
			np.zeros(self._dimensions), \
			np.ones(self._dimensions)*(self._const.resolution - 1), \
			dataset.train_labels, \
			dataset.train_weights

	def importance_data(self, dataset): 
		return self.rbn(dataset.val_past_snapshots), \
			self.rbn(dataset.val_snapshots), \
			dataset.val_labels, \
			dataset.val_weights

	def stepwise_data(self, dataset):
		return self.rbn(dataset.train_past_snapshots), \
			self.rbn(dataset.train_snapshots), \
			dataset.train_labels, \
			dataset.train_weights, \
			self.rbn(dataset.val_past_snapshots), \
			self.rbn(dataset.val_snapshots), \
			dataset.val_labels, \
			dataset.val_weights