import numpy as np


class Squeezer():
    @staticmethod
    def squeeze_pBs(pBs, const):
        return np.array([const.logvmin if ele == 0
                         else 1 - const.logvmin if ele == 1
                         else ele
                         for ele in pBs])

    @staticmethod
    def squeeze_pB_dict(pB_dict, const):
        return {key: const.logvmin if value == 0
                else 1 - const.logvmin if value == 1
                else value
                for key, value in pB_dict.items()}
