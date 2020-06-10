import numpy as np

class Trimmer():
	def __init__(self, pBs):
		self._pBs = pBs
		self._keep_elements = [ele == 0 or ele == 1 for ele in pBs]

	def trim(self, list_to_trim):
		return np.array(list_to_trim)[self._keep_elements]

	@staticmethod
	def trim_dict(dict_to_trim):
		return {key: label for key, label in dict_to_trim.items() \
			if label != 0 and label != 1}