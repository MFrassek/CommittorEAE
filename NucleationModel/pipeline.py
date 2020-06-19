from reducer import Reducer
from bounder import Bounder
from normalizer import Normalizer
from gridifier import Gridifier
from pB_approximator import pB_Approximator
from trimmer import Trimmer
from pB_balancer import pB_Balancer

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

	@property
	def mean(self):
		return self._normalizer.mean
	@property
	def std(self):
		return self._normalizer.std
	
	@property
	def minima(self):
		return self._gridifier.minima
	@property
	def maxima(self):
		return self._gridifier.maxima

	def rbn(self, snapshots):
		"""Reduce, bound and normalize snapshots."""
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		return snapshots

	def rbng(self, snapshots):
		"""Reduce, bound, normalize and gridify snapshots."""
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		snapshots = self._gridifier.gridify_snapshots(snapshots)		
		return snapshots

	def rbnga(self, snapshots, labels, weights):
		"""Reduce, bound, normalize and gridify snapshots 
		and approximate the pBs.
		"""
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		snapshots = self._gridifier.gridify_snapshots(snapshots)		
		pB_dict, pBs = pB_Approximator.approximate_pBs(
			snapshots,
			labels,
			weights)
		return snapshots, pB_dict, pBs

	def rbngat(self, snapshots, labels, weights):
		"""Reduce, bound, normalize and gridify snapshots,
		approximate the pBs
		and trimm the snapshots, labels and weights.
		"""
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		snapshots = self._gridifier.gridify_snapshots(snapshots)		
		pB_dict, pBs = pB_Approximator.approximate_pBs(
			snapshots,
			labels,
			weights)
		trimmer = Trimmer(pBs)
		snapshots = trimmer.trim_snapshots(snapshots)
		label = trimmer.trim_snapshots(labels)
		weights = trimmer.trim_snapshots(weights)
		pB_dict = trimmer.trim_dict(pB_dict)
		pBs = trimmer.trim_snapshots(pBs)
		return snapshots, labels, weights, pB_dict, pBs

	def rbngatb(self, snapshots, labels, weights):
		"""Reduce, bound, normalize and gridify snapshots,
		approximate the pBs,
		trimm the snapshots, labels and weights
		and generate balanced weights for the pBs.
		"""
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		snapshots = self._gridifier.gridify_snapshots(snapshots)		
		pB_dict, pBs = pB_Approximator.approximate_pBs(
			snapshots,
			labels,
			weights)
		trimmer = Trimmer(pBs)
		snapshots = trimmer.trim_snapshots(snapshots)
		label = trimmer.trim_snapshots(labels)
		weights = trimmer.trim_snapshots(weights)
		pB_dict = trimmer.trim_dict(pB_dict)
		pBs = trimmer.trim_snapshots(pBs)
		pB_weights = pB_Balancer.balance(pBs, self._const.balance_bins)
		return snapshots, labels, weights, pB_dict, pBs, pB_weights

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