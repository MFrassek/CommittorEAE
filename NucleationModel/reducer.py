import numpy as np


class Reducer():
    def __init__(self, const):
        self._used_variable_names = const._used_variable_names
        self._name_to_list_position = const.name_to_list_position

    def reduce_snapshots(self, snapshots):
        columns = np.transpose(snapshots)
        return np.transpose([columns[self._name_to_list_position[name]]
                            for name in self._used_variable_names])
