import numpy as np
class Reducer():
	def __init__(
			self,
			reduced_list_of_variable_names,
			name_to_list_position):
		self._reduced_list_of_variable_names = reduced_list_of_variable_names
		self._name_to_list_position = name_to_list_position

	def reduce_snapshots(self, snapshots):
		columns = np.transpose(snapshots)
		return np.transpose([columns[self._name_to_list_position[name]] \
			for name in self._reduced_list_of_variable_names])