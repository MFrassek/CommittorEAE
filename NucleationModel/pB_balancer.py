import numpy as np
from collections import Counter


class pB_Balancer():
    @staticmethod
    def balance(pBs, bins):
        pBs_len = len(pBs)
        round_pBs = np.ceil((np.array(pBs) * (bins + 1)))
        counter = Counter(round_pBs)
        balanced_counter = {key: pBs_len/label for key, label
                            in counter.items()}
        pB_balanced_weights = np.array(
            [balanced_counter[i] for i in round_pBs])
        return pB_balanced_weights

    @staticmethod
    def balance_and_mask(pBs, bins):
        pBs_len = len(pBs)
        round_pBs = np.ceil((np.array(pBs) * (bins + 1)))
        counter = Counter(round_pBs)
        masked_balanced_counter = {key: pBs_len/label
                                   if key != 0 and key != bins + 1 else 0
                                   for key, label
                                   in counter.items()}
        masked_pB_balanced_weights = np.array(
            [masked_balanced_counter[i] for i in round_pBs])
        return masked_pB_balanced_weights

    @staticmethod
    def trim_and_balance(pBs, bins):
        trimmed_pBs = [ele for ele in pBs if ele != 0 and ele != 1]
        trimmed_pBs_len = len(trimmed_pBs)
        trimmed_round_pBs = np.ceil((np.array(trimmed_pBs) * (bins + 1)))
        counter = Counter(trimmed_round_pBs)
        trimmed_balanced_counter = {key: trimmed_pBs_len/label
                                    for key, label
                                    in counter.items()}
        trimmed_pB_balanced_weights = np.array(
            [trimmed_balanced_counter[i] for i in trimmed_round_pBs])
        return trimmed_pB_balanced_weights
