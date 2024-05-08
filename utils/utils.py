import numpy as np


def get_one_hot(k, n):
    ret = np.zeros(n)
    ret[k] = 1
    return ret
