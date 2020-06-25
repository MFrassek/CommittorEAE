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
	def lower_bound(self):
		return self._bounder.lower_bound
	@property
	def upper_bound(self):
		return self._bounder.upper_bound

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
		snapshots = self.rbn(snapshots)
		grid_snapshots = self._gridifier.gridify_snapshots(snapshots)		
		return grid_snapshots, snapshots

	def rbnga(self, dataset):
		"""Reduce, bound, normalize and gridify snapshots 
		and approximate the pBs.
		"""
		grid_snapshots, snapshots = self.rbng(dataset.past_snapshots)		
		pB_dict, pBs, pB_weights = pB_Approximator.approximate_pBs(
			grid_snapshots,
			dataset.labels,
			dataset.weights)
		return grid_snapshots, snapshots, pB_dict, pBs, pB_weights

	def rbngat(self, dataset):
		"""Reduce, bound, normalize and gridify snapshots,
		approximate the pBs
		and trimm the snapshots, labels and weights.
		"""
		grid_snapshots, snapshots, \
			pB_dict, pBs, pB_weights = self.rbnga(dataset)
		trimmer = Trimmer(pBs)
		grid_snapshots = trimmer.trim_snapshots(grid_snapshots)
		snapshots = trimmer.trim_snapshots(snapshots)
		labels = trimmer.trim_snapshots(dataset.labels)
		weights = trimmer.trim_snapshots(dataset.weights)
		pB_dict = trimmer.trim_dict(pB_dict)
		pBs = trimmer.trim_snapshots(pBs)
		pB_weights = trimmer.trim_snapshots(pB_weights)
		return grid_snapshots, snapshots, labels, weights, \
			pB_dict, pBs, pB_weights

	def rbngatb(self, dataset):
		"""Reduce, bound, normalize and gridify snapshots,
		approximate the pBs,
		trimm the snapshots, labels and weights
		and generate balanced weights for the pBs.
		"""
		grid_snapshots, snapshots, labels, weights, \
			pB_dict, pBs, pB_weights = self.rbngat(dataset)
		pBb_weights = pB_Balancer.balance(pBs, self._const.balance_bins)
		return grid_snapshots, snapshots, labels, weights, \
			pB_dict, pBs, pB_weights, pBb_weights

	def importance_data(self, valDataset): 
		assert valDataset.flag == "Validation", \
			"valDataset needs to be a validation set."
		return self.rbn(valDataset.past_snapshots), \
			self.rbn(valDataset.snapshots), \
			valDataset.labels, \
			valDataset.weights

	def stepwise_data(self, trainDataset, valDataset):
		assert trainDataset.flag == "Training", \
			"trainDataset needs to be a training set."
		assert valDataset.flag == "Validation", \
			"valDataset needs to be a validation set."
		return self.rbn(trainDataset.past_snapshots), \
			self.rbn(trainDataset.snapshots), \
			trainDataset.labels, \
			trainDataset.weights, \
			self.rbn(valDataset.past_snapshots), \
			self.rbn(valDataset.snapshots), \
			valDataset.labels, \
			valDataset.weights