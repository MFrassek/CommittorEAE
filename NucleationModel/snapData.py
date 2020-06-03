import numpy as np
from sklearn.utils import shuffle

class SnapData():
	def __init__(
			self, AA_past_snapshots, AB_past_snapshots, 
			BA_past_snapshots, BB_past_snapshots, 
			AA_snapshots, AB_snapshots, 
			BA_snapshots, BB_snapshots, 
			AA_snapshot_labels, AB_snapshot_labels, 
			BA_snapshot_labels, BB_snapshot_labels, 
			AA_snapshot_weights, AB_snapshot_weights, 
			BA_snapshot_weights, BB_snapshot_weights):
		self._AA_past_snapshots = AA_past_snapshots
		self._AB_past_snapshots = AB_past_snapshots
		self._BA_past_snapshots = BA_past_snapshots
		self._BB_past_snapshots = BB_past_snapshots
		self._AA_snapshots = AA_snapshots
		self._AB_snapshots = AB_snapshots
		self._BA_snapshots = BA_snapshots
		self._BB_snapshots = BB_snapshots
		self._AA_snapshot_labels = AA_snapshot_labels
		self._AB_snapshot_labels = AB_snapshot_labels
		self._BA_snapshot_labels = BA_snapshot_labels
		self._BB_snapshot_labels = BB_snapshot_labels
		self._AA_snapshot_weights = AA_snapshot_weights
		self._AB_snapshot_weights = AB_snapshot_weights
		self._BA_snapshot_weights = BA_snapshot_weights
		self._BB_snapshot_weights = BB_snapshot_weights


	@property
	def snapshot_cnt(self):
		return len(self._AA_snapshots) + len(self._AB_snapshots) + \
				len(self._BA_snapshots) + len(self._BB_snapshots)
	@property
	def AA_snapshot_cnt(self):
		return len(self._AA_snapshots)
	@property
	def AB_snapshot_cnt(self):
		return len(self._AB_snapshots)
	@property
	def BA_snapshot_cnt(self):
		return len(self._BA_snapshots)
	@property
	def BB_snapshot_cnt(self):
		return len(self._BB_snapshots)

	@property
	def past_snapshots(self):
		"""returns list of all past_snapshots in order AA, AB, BA, BB"""
		return np.array([snapshot for paths \
				in [self._AA_past_snapshots, self._AB_past_snapshots, \
				self._BA_past_snapshots, self._BB_past_snapshots] \
				for snapshot in paths])
	@property
	def AA_past_snapshots(self):
		return self._AA_past_snapshots
	@property
	def AB_past_snapshots(self):
		return self._AB_past_snapshots
	@property
	def BA_past_snapshots(self):
		return self._BA_past_snapshots
	@property
	def BB_past_snapshots(self):
		return self._BB_snapshots

	@property
	def snapshots(self):
		"""returns list of all snapshots in order AA, AB, BA, BB"""
		return np.array([snapshot for paths \
				in [self._AA_snapshots, self._AB_snapshots, \
				self._BA_snapshots, self._BB_snapshots] \
				for snapshot in paths])
	@property
	def AA_snapshots(self):
		return self._AA_snapshots
	@property
	def AB_snapshots(self):
		return self._AB_snapshots
	@property
	def BA_snapshots(self):
		return self._BA_snapshots
	@property
	def BB_snapshots(self):
		return self._BB_snapshots
	
	@property
	def snapshot_labels(self):
		"""returns list of all snapshot_labels in order AA, AB, BA, BB"""
		return np.array([snapshot for paths \
				in [self._AA_snapshot_labels, self._AB_snapshot_labels, \
				self._BA_snapshot_labels, self._BB_snapshot_labels] \
				for snapshot in paths])
	@property
	def AA_snapshot_labels(self):
		return self._AA_snapshot_labels
	@property
	def AB_snapshot_labels(self):
		return self._AB_snapshot_labels
	@property
	def BA_snapshot_labels(self):
		return self._BA_snapshot_labels
	@property
	def BB_snapshot_labels(self):
		return self._BB_snapshot_labels

	@property
	def snapshot_weights(self):
		"""returns list of all snapshot_weights in order AA, AB, BA, BB"""
		return np.array([snapshot for paths \
				in [self._AA_snapshot_weights, self._AB_snapshot_weights, \
				self._BA_snapshot_weights, self._BB_snapshot_weights] \
				for snapshot in paths])
	@property
	def AA_snapshot_weights(self):
		return self._AA_snapshot_weights
	@property
	def AB_snapshot_weights(self):
		return self._AB_snapshot_weights
	@property
	def BA_snapshot_weights(self):
		return self._BA_snapshot_weights
	@property
	def BB_snapshot_weights(self):
		return self._BB_snapshot_weights
	
	def shuffle_lists(self):
		return shuffle(self.past_snapshots, self.snapshots, \
				self.snapshot_labels, self.snapshot_weights, random_state = 42)

	def split_lists(self, train_ratio, val_ratio):
		assert isinstance(train_ratio, float) \
			and train_ratio > 0.0, \
			"train_ratio needs to be a float higher than 0.0"
		assert isinstance(val_ratio, float) \
			and val_ratio > 0.0, \
			"val_ratio needs to be a float higher than 0.0"
		assert train_ratio + val_ratio < 1.0, \
			"Sum of train_ratio and val_ratio must be lower than 1.0"
		train_end = int(self.snapshot_cnt * train_ratio)
		val_end = train_end + int(self.snapshot_cnt * val_ratio) 
		past_snapshots, snapshots, snapshot_labels, snapshot_weights \
			= self.shuffle_lists()
		return np.array([*past_snapshots[:train_end]]), \
				np.array([*past_snapshots[train_end:val_end]]), \
				np.array([*past_snapshots[val_end:]]), \
				np.array([*snapshots[:train_end]]), \
				np.array([*snapshots[train_end:val_end]]), \
				np.array([*snapshots[val_end:]]), \
				np.array([*snapshot_labels[:train_end]]), \
				np.array([*snapshot_labels[train_end:val_end]]), \
				np.array([*snapshot_labels[val_end:]]), \
				np.array([*snapshot_weights[:train_end]]), \
				np.array([*snapshot_weights[train_end:val_end]]), \
				np.array([*snapshot_weights[val_end:]])

	def __str__(self):
		return "Snapshots (cnt): {} \n{} in AA\n{} in AB\n{} in BA\n{} in BB"\
			.format(self.snapshot_cnt,self.AA_snapshot_cnt,self,\
			self.AB_snapshot_cnt, self.BA_snapshot_cnt, self.BB_snapshot_cnt)
