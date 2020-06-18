from reducer import Reducer
from bounder import Bounder
from normalizer import Normalizer
from gridifier import Gridifier

class Pipeline():
	def __init__(self, const, reduced_list_var_names, base_snapshots):
		self._const = const
		self._reduced_list_var_names = reduced_list_var_names
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

	def process(self, snapshots):
		snapshots = self._reducer.reduce_snapshots(snapshots)
		snapshots = self._bounder.bound_snapshots(snapshots)
		snapshots = self._normalizer.normalize_snapshots(snapshots)
		snapshots = self._gridifier.gridify_snapshots(snapshots)

		return snapshots