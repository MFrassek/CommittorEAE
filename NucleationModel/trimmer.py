

class Trimmer():
	def __init__(self, pBs):
		self._pBs = pBs
		self._keep_elements = [ele == 0 or ele == 1 for ele in pBs]

	def trim(self, list_to_trim):
		return list_to_trim[self._keep_elements]