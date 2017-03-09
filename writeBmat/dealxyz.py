from math import *
import mymath
import numpy as np


class Dealxyz:
    """
    from cartesians and given reciept (intwhat, intwhere) and
    lattice vectors (lattmat) calculates values of internal
    coordinates
    """

    def __init__(self, cart, coords, lattmat):

        # cart must be an array!!!
        cart = np.array(cart).reshape(len(cart) / 3, 3)
        self.internals = []

        for i in range(len(coords)):
            if coords[i].dtyp == 'simple':
                if coords[i].tag == 'X':
                    internal = self.set_singles(cart, 0, coords[i].what)
                elif coords[i].tag == 'Y':
                    internal = self.set_singles(cart, 1, coords[i].what)
                elif coords[i].tag == 'Z':
                    internal = self.set_singles(cart, 2, coords[i].what)
                elif coords[i].tag == 'fX':
                    internal = self.set_fsingles(cart, 0, coords[i].what, lattmat)
                elif coords[i].tag == 'fY':
                    internal = self.set_fsingles(cart, 1, coords[i].what, lattmat)
                elif coords[i].tag == 'fZ':
                    internal = self.set_fsingles(cart, 2, coords[i].what, lattmat)
                elif coords[i].tag == 'hX':
                    internal = lattmat[coords[i].what[0]][0]
                elif coords[i].tag == 'hY':
                    internal = lattmat[coords[i].what[0]][1]
                elif coords[i].tag == 'hZ':
                    internal = lattmat[coords[i].what[0]][2]
                elif coords[i].tag == 'R':
                    internal = self.set_lengths(cart, coords[i].what, coords[i].where, lattmat)
                elif coords[i].tag == 'M':
                    internal = self.set_midlengths(cart, coords[i].what, coords[i].where, lattmat)
                elif coords[i].tag == 'RatioR':
                    internal = self.set_ratior(cart, coords[i].what, coords[i].where, lattmat)
                elif coords[i].tag == 'A':
                    internal = self.set_angles(cart, coords[i].what, coords[i].where, lattmat)
                elif coords[i].tag == 'T':
                    internal = self.set_dihs(cart, coords[i].what, coords[i].where, lattmat)
                elif coords[i].tag == 'tV':
                    internal = self.set_tetrahedralVol(cart, coords[i].what, coords[i].where, lattmat)
                elif coords[i].tag == 'IR1':
                    internal = self.set_lengths(cart, coords[i].what, coords[i].where, lattmat)
                    internal = 5 / internal
                elif coords[i].tag == 'IR6':
                    internal = self.set_lengths(cart, coords[i].what, coords[i].where, lattmat)
                    internal = 2000 / internal**6
                elif coords[i].tag == 'LR':
                    internal = self.set_llength(coords[i].what, lattmat)
                elif coords[i].tag == 'LA':
                    internal = self.set_langle(coords[i].what, lattmat)
                elif coords[i].tag == 'LB':
                    internal = self.set_lbngle(coords[i].what, lattmat)
                elif coords[i].tag == 'LV':
                    internal = self.set_lvolume(lattmat)
                elif coords[i].tag == 'RatioLR':
                    internal = self.set_ratiollength(coords[i].what, lattmat)
            elif coords[i].dtyp == 'sum':
                internal = self.set_sum(cart, coords[i], lattmat)
            elif coords[i].dtyp == 'norm':
                internal = self.set_norm(cart, coords[i], lattmat)
            elif coords[i].dtyp == 'cn':
                internal = self.set_cnum(cart, coords[i], lattmat)
            self.internals.append(internal)

        self.internals = np.array(self.internals)

    def set_lengths(self, cart, what, where, lattmat):
        'Calculates bonds.'

        index1 = what[0]
        index2 = what[1]
        a = cart[index1] + np.dot(where[0], lattmat)
        b = cart[index2] + np.dot(where[1], lattmat)
        bond = (sum((a - b)**2))**0.5
        return bond

    def set_midlengths(self, cart, what, where, lattmat):
        'Calculates distance from an atom to midpoint between two atoms.'

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]

        a = cart[index1] + np.dot(where[0], lattmat)
        b = cart[index2] + np.dot(where[1], lattmat)
        # c=cart[index3]+np.dot(where[1],lattmat)+np.dot(where[2],lattmat)
        c = cart[index3] + np.dot(where[2], lattmat)
        m = (b + c) / 2
        bond = (sum((a - m)**2))**0.5
        return bond

    def set_ratior(self, cart, what, where, lattmat):

        r1 = self.set_lengths(cart, what[:2], where[:2], lattmat)
        r2 = self.set_lengths(cart, what[2:], where[2:], lattmat)
        return r1 / r2

    def set_angles(self, cart, what, where, lattmat):
        'Calculates angles.'

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]

        a = cart[index1] + np.dot(where[0], lattmat)
        b = cart[index2]
        c = cart[index3] + np.dot(where[2], lattmat)

        vector1 = a - b
        vector2 = c - b

        size1 = mymath.vector_size(vector1)
        size2 = mymath.vector_size(vector2)
        angle = sum(vector1 * vector2) / (size1 * size2)
        if angle > 1:
            angle = 1
        elif angle < -1:
            angle = -1
        angle = abs(acos(angle))
        return angle

    def set_dihs(self, cart, what, where, lattmat):
        'Calculates torsions.'

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]
        index4 = what[3]
        a = cart[index1] + np.dot(where[0], lattmat)
        b = cart[index2]
        c = cart[index3] + np.dot(where[2], lattmat)
        d = cart[index4] + np.dot(where[3], lattmat)
        vector1 = a - b
        vector2 = b - c
        vector3 = c - d
        cross1 = mymath.cross_product(vector1, vector2)
        cross2 = mymath.cross_product(vector2, vector3)
        cross1_size = mymath.vector_size(cross1)
        cross2_size = mymath.vector_size(cross2)
        fuck = sum(cross1 * cross2) / (cross1_size * cross2_size)
        if fuck > 1:
            fuck = 1.0
        if fuck < -1:
            fuck = -1.0
        dangle = acos(fuck)
        if sum(cross1 * vector3) >= 0:
            dangle = -dangle
        return dangle

    def set_tetrahedralVol(self, cart, what, where, lattmat):
        'Calculates torsions.'

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]
        index4 = what[3]

        a = cart[index1] + np.dot(where[0], lattmat)
        b = cart[index2] + np.dot(where[1], lattmat)
        c = cart[index3] + np.dot(where[2], lattmat)
        d = cart[index4] + np.dot(where[3], lattmat)

        vector1 = b - a
        vector2 = c - a
        vector3 = d - a

        cross1 = mymath.cross_product(vector1, vector2)
        tv = abs(sum(cross1 * vector3) / 6)
        return tv

    def set_singles(self, cart, xyz, what):

        index = what[0]
        internal = cart[index][xyz]
        return internal

    def set_fsingles(self, cart, xyz, what, lattmat):

        index = what[0]
        x = cart[index]
        x = np.dot(x, np.linalg.inv(lattmat))
        internal = x[xyz]
        return internal

    def set_llength(self, intwhat, lattmat):

        index1 = intwhat[0]
        llength = (sum(lattmat[index1] * lattmat[index1]))**0.5
        return llength

    def set_langle(self, intwhat, lattmat):

        a = intwhat[0]
        b = intwhat[1]
        diffav = lattmat[a]
        diffbv = lattmat[b]
        d1 = mymath.vector_size(diffav)
        d2 = mymath.vector_size(diffbv)
        cosalpha = (sum(diffav * diffbv) / (d1 * d2))
        alpha = acos(cosalpha)
        return alpha

    def set_lbngle(self, intwhat, lattmat):

        a = intwhat[0]
        b = intwhat[1]
        c = intwhat[2]
        av = lattmat[a]
        bv = lattmat[b]
        cv = lattmat[c]

        v1 = mymath.cross_product(av, bv)
        v2 = mymath.cross_product(av, cv)

        v1 = v1 / sum(v1**2)**0.5
        v2 = v2 / sum(v2**2)**0.5
        alpha = sum(v1 * v2)
        alpha = acos(alpha)
        return alpha

    def set_lvolume(self, lattmat):

        l1 = lattmat[0]
        l2 = lattmat[1]
        l3 = lattmat[2]
        volume = mymath.cross_product(l1, l2)
        volume = sum(volume * l3)
        return volume

    def set_ratiollength(self, intwhat, lattmat):

        index1 = intwhat[0]
        llengtha = (sum(lattmat[index1] * lattmat[index1]))**0.5
        index2 = intwhat[1]
        llengthb = (sum(lattmat[index2] * lattmat[index2]))**0.5
        ratio = llengtha / llengthb
        return ratio

    def set_sum(self, cart, coord, lattmat):

        complexcoord = 0.0

        for i in range(len(coord.tag)):
            if coord.tag[i] == 'X':
                dist = self.set_singles(cart, 0, coord.what[i])
            if coord.tag[i] == 'Y':
                dist = self.set_singles(cart, 1, coord.what[i])
            if coord.tag[i] == 'Z':
                dist = self.set_singles(cart, 2, coord.what[i])
            if coord.tag[i] == 'fX':
                dist = self.set_fsingles(cart, 0, coord.what[i], lattmat)
            if coord.tag[i] == 'fY':
                dist = self.set_fsingles(cart, 1, coord.what[i], lattmat)
            if coord.tag[i] == 'fZ':
                dist = self.set_fsingles(cart, 2, coord.what[i], lattmat)
            if coord.tag[i] == 'R':
                dist = self.set_lengths(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'M':
                dist = self.set_midlengths(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'A':
                dist = self.set_angles(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'T':
                dist = self.set_dihs(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'tV':
                dist = self.set_tetrahedralVol(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'LR':
                dist = self.set_llength(coord.what[i], lattmat)
            if coord.tag[i] == 'LA':
                dist = self.set_langle(coord.what[i], lattmat)
            if coord.tag[i] == 'RatioR':
                dist = self.set_ratior(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'LR':
                dist = self.set_llength(coord.what[i], lattmat)
            if coord.tag[i] == 'LA':
                dist = self.set_langle(coord.what[i], lattmat)
            if coord.tag[i] == 'LV':
                dist = self.set_lvolume(lattmat)
            if coord.tag[i] == 'RatioLR':
                dist = self.set_ratiollength(coord.what[i], lattmat)
            if coord.tag[i] == 'hX':
                dist = lattmat[coord.what[i][0]][0]
            if coord.tag[i] == 'hY':
                dist = lattmat[coord.what[i][0]][1]
            if coord.tag[i] == 'hZ':
                dist = lattmat[coord.what[i][0]][2]

            complexcoord += coord.coefs[i] * dist
        return complexcoord

    def set_norm(self, cart, coord, lattmat):

        complexcoord = 0.0

        for i in range(len(coord.tag)):
            if coord.tag[i] == 'X':
                dist = self.set_singles(cart, 0, coord.what[i])
            if coord.tag[i] == 'Y':
                dist = self.set_singles(cart, 1, coord.what[i])
            if coord.tag[i] == 'Z':
                dist = self.set_singles(cart, 2, coord.what[i])
            if coord.tag[i] == 'R':
                dist = self.set_lengths(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'M':
                dist = self.set_midlengths(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'A':
                dist = self.set_angles(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'T':
                dist = self.set_dihs(cart, coord.what[i], coord.where[i], lattmat)
            if coord.tag[i] == 'LR':
                dist = self.set_llength(coord.what[i], lattmat)
            if coord.tag[i] == 'LA':
                dist = self.set_langle(coord.what[i], lattmat)
            if coord.tag[i] == 'RatioR':
                dist = self.set_ratior(cart, coord.what[i], coord.where[i], lattmat)
            complexcoord += (coord.coefs[i] * dist)**2

        complexcoord = complexcoord**0.5
        return complexcoord

    def set_cnum(self, cart, coord, lattmat):

        complexcoord = 0.0

        for i in range(len(coord.tag)):
            if coord.tag[i] == 'R':
                if abs(coord.coefs[i]) > 1e-4:
                    # dist=coord.value
                    dist = self.set_lengths(cart,coord.what[i], coord.where[i], lattmat)
                    dummyq = dist / coord.coefs[i]
                    if abs(dummyq - 1.0) < 1.0e-4:
                        dummyq = 1.0001
                    complexcoord = complexcoord + (1.0 - dummyq**9.) / (1.0 - dummyq**14.0)

        return complexcoord
