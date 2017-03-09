from math import *

import numpy as np


def cart_dir(lvect, carts):
    """
    Transforms cartesian coordinates to fractional
    """

    m = np.linalg.inv(lvect)
    direct = np.dot(carts, m)
    for i in range(len(direct)):
        for j in range(3):
            while direct[i][j] > 1:
                direct[i][j] = direct[i][j] - 1
            while direct[i][j] < 0:
                direct[i][j] = direct[i][j] + 1
    idirect = []
    for i in range(len(direct)):
        idirect.append(direct[i])

    return idirect
