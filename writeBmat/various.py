import numpy as np


def give_dist(A, B):

    return (sum((A - B) ** 2)) ** 0.5


def shortest_dist(cartesians, lattmat, atom1, atom2):
    """
    finds the shortest distance between two atoms
    """

    cartesians = np.array(cartesians).reshape(len(cartesians) / 3, 3)
    cart1 = cartesians[atom1]
    cart2 = cartesians[atom2]
    dists = []
    what = []

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                trans = i * lattmat[0] + j * lattmat[1] + k * lattmat[2]
                point2 = cart2 + trans
                dist = give_dist(cart1, point2)
                dists.append(dist)
                what.append([i, j, k])

    dists = np.array(dists)
    dummy = np.argmin(dists)
    return [[0, 0, 0], [what[dummy][0], what[dummy][1], what[dummy][2]]]
