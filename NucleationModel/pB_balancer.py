import numpy as np
from collections import Counter

class pB_Balancer():
	@staticmethod
	def balance(pBs, bins):
		pBs_len = len(pBs)
		round_pBs = np.ceil((np.array(pBs) * (bins + 1)))
		counter = Counter(round_pBs)
		balanced_counter = {key: pBs_len/label for key, label \
			in counter.items()}
		pB_balanced_weights = np.array([balanced_counter[i] for i in round_pBs])
		return pB_balanced_weights

	@staticmethod
	def balance_and_trim(pBs, bins):
		pBs_len = len(pBs)
		round_pBs = np.ceil((np.array(pBs) * (bins + 1)))
		counter = Counter(round_pBs)
		trimmed_balanced_counter = {key: pBs_len/label if key != 0 and \
			key != bins + 1 else 0 for key, label in counter.items()}
		trimmed_pB_balanced_weights = np.array([trimmed_balanced_counter[i] \
			for i in round_pBs])
		return trimmed_pB_balanced_weights