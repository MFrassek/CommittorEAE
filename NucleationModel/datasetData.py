import numpy as np
import tensorflow as tf

class DatasetData():
	def __init__(
			self, 
			train_past_snapshots,
			val_past_snapshots,
			test_past_snapshots,
			train_snapshots,
			val_snapshots,
			test_snapshots,
			train_labels,
			val_labels,
			test_labels,
			train_weights,
			val_weights,
			test_weights):
		self._train_past_snapshots = train_past_snapshots
		self._val_past_snapshots = val_past_snapshots
		self._test_past_snapshots = test_past_snapshots
		self._train_snapshots = train_snapshots
		self._val_snapshots = val_snapshots
		self._test_snapshots = test_snapshots			
		self._train_labels = train_labels
		self._val_labels = val_labels
		self._test_labels = test_labels
		self._train_weights = train_weights
		self._val_weights = val_weights
		self._test_weights = test_weights
		self._train_size = len(train_past_snapshots)
		self._val_size = len(val_past_snapshots)
		self._test_size = len(test_past_snapshots)

	def __str__(self):
		return "{} training snapshots\n{} testing snapshots\n\
		{} input dimensions"\
			.format(self.train_size, self.test_size, self._dimensions)

	@property
	def train_size(self):
		return self._train_size
	@property
	def val_size(self):
		return self._val_size
	@property
	def test_size(self):
		return self._test_size

	@property
	def train_past_snapshots(self):
		return self._train_past_snapshots
	@property
	def val_past_snapshots(self):
		return self._val_past_snapshots
	@property
	def test_past_snapshots(self):
		return self._test_past_snapshots
	@property
	def train_snapshots(self):
		return self._train_snapshots
	@property
	def val_snapshots(self):
		return self._val_snapshots
	@property
	def test_snapshots(self):
		return self._test_snapshots
	@property
	def train_labels(self):
		return self._train_labels
	@property
	def val_labels(self):
		return self._val_labels
	@property
	def test_labels(self):
		return self._test_labels
	@property
	def train_weights(self):
		return self._train_weights
	@property
	def val_weights(self):
		return self._val_weights
	@property
	def test_weights(self):
		return self._test_weights