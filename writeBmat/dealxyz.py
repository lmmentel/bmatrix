from math import *
import numpy as np

from .physconstants import ANGS2BOHR


class Dealxyz:
    """
    From cartesians and given reciept (intwhat, intwhere) and
    lattice vectors (cell) calculates values of internal
    coordinates

    Args:
        atoms (ase.Atoms) :
            Atoms object from ASE package

        coords () :
            Internal coordinates

        cell (array_like) :
            Lattice vectors in atomic units (3 x 3)

    """

    def __init__(self, atoms, coords):

        self.natoms = len(atoms)
        self.cell = atoms.get_cell() * ANGS2BOHR
        self.fractional = atoms.get_scaled_positions()
        self.cartesian = atoms.get_positions() * ANGS2BOHR
        self.symbols = atoms.get_chemical_symbols()

        self.internals = []

        for i in range(len(coords)):
            if coords[i].dtyp == "simple":
                if coords[i].tag == "X":
                    internal = self.set_singles(0, coords[i].what)
                elif coords[i].tag == "Y":
                    internal = self.set_singles(1, coords[i].what)
                elif coords[i].tag == "Z":
                    internal = self.set_singles(2, coords[i].what)
                elif coords[i].tag == "fX":
                    internal = self.set_fsingles(0, coords[i].what)
                elif coords[i].tag == "fY":
                    internal = self.set_fsingles(1, coords[i].what)
                elif coords[i].tag == "fZ":
                    internal = self.set_fsingles(2, coords[i].what)
                elif coords[i].tag == "hX":
                    internal = self.cell[coords[i].what[0]][0]
                elif coords[i].tag == "hY":
                    internal = self.cell[coords[i].what[0]][1]
                elif coords[i].tag == "hZ":
                    internal = self.cell[coords[i].what[0]][2]
                elif coords[i].tag == "R":
                    internal = self.set_lengths(coords[i].what, coords[i].where)
                elif coords[i].tag == "M":
                    internal = self.set_midlengths(coords[i].what, coords[i].where)
                elif coords[i].tag == "RatioR":
                    internal = self.set_ratior(coords[i].what, coords[i].where)
                elif coords[i].tag == "A":
                    internal = self.set_angles(coords[i].what, coords[i].where)
                elif coords[i].tag == "T":
                    internal = self.set_dihs(coords[i].what, coords[i].where)
                elif coords[i].tag == "tV":
                    internal = self.set_tetrahedralVol(coords[i].what, coords[i].where)
                elif coords[i].tag == "IR1":
                    internal = self.set_lengths(coords[i].what, coords[i].where)
                    internal = 5 / internal
                elif coords[i].tag == "IR6":
                    internal = self.set_lengths(coords[i].what, coords[i].where)
                    internal = 2000 / internal ** 6
                elif coords[i].tag == "LR":
                    internal = self.set_llength(coords[i].what)
                elif coords[i].tag == "LA":
                    internal = self.set_langle(coords[i].what)
                elif coords[i].tag == "LB":
                    internal = self.set_lbngle(coords[i].what)
                elif coords[i].tag == "LV":
                    internal = self.set_lvolume()
                elif coords[i].tag == "RatioLR":
                    internal = self.set_ratiollength(coords[i].what)
            elif coords[i].dtyp == "sum":
                internal = self.set_sum(coords[i])
            elif coords[i].dtyp == "norm":
                internal = self.set_norm(coords[i])
            elif coords[i].dtyp == "cn":
                internal = self.set_cnum(coords[i])
            self.internals.append(internal)

        self.internals = np.array(self.internals)

    def set_lengths(self, what, where):
        "Calculates bonds."

        index1 = what[0]
        index2 = what[1]
        a = self.cartesian[index1] + np.dot(where[0], self.cell)
        b = self.cartesian[index2] + np.dot(where[1], self.cell)
        return (sum((a - b) ** 2)) ** 0.5

    def set_midlengths(self, what, where):
        "Calculates distance from an atom to midpoint between two atoms."

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]

        a = self.cartesian[index1] + np.dot(where[0], self.cell)
        b = self.cartesian[index2] + np.dot(where[1], self.cell)
        # c=cart[index3]+np.dot(where[1],cell)+np.dot(where[2],cell)
        c = self.cartesian[index3] + np.dot(where[2], self.cell)
        m = (b + c) / 2
        return (sum((a - m) ** 2)) ** 0.5

    def set_ratior(self, what, where):

        r1 = self.set_lengths(self.cartesian, what[:2], where[:2])
        r2 = self.set_lengths(self.cartesian, what[2:], where[2:])
        return r1 / r2

    def set_angles(self, what, where):
        "Calculates angles."

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]

        a = self.cartesian[index1] + np.dot(where[0], self.cell)
        b = self.cartesian[index2]
        c = self.cartesian[index3] + np.dot(where[2], self.cell)

        vector1 = a - b
        vector2 = c - b

        size1 = np.linalg.norm(vector1)
        size2 = np.linalg.norm(vector2)
        angle = sum(vector1 * vector2) / (size1 * size2)
        if angle > 1:
            angle = 1
        elif angle < -1:
            angle = -1
        angle = abs(acos(angle))
        return angle

    def set_dihs(self, what, where):
        "Calculates torsions."

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]
        index4 = what[3]
        a = self.cartesian[index1] + np.dot(where[0], self.cell)
        b = self.cartesian[index2]
        c = self.cartesian[index3] + np.dot(where[2], self.cell)
        d = self.cartesian[index4] + np.dot(where[3], self.cell)
        vector1 = a - b
        vector2 = b - c
        vector3 = c - d
        cross1 = np.cross(vector1, vector2)
        cross2 = np.cross(vector2, vector3)
        cross1_size = np.linalg.norm(cross1)
        cross2_size = np.linalg.norm(cross2)
        fuck = sum(cross1 * cross2) / (cross1_size * cross2_size)
        if fuck > 1:
            fuck = 1.0
        if fuck < -1:
            fuck = -1.0
        dangle = acos(fuck)
        if sum(cross1 * vector3) >= 0:
            dangle = -dangle
        return dangle

    def set_tetrahedralVol(self, what, where):
        "Calculates torsions."

        index1 = what[0]
        index2 = what[1]
        index3 = what[2]
        index4 = what[3]

        a = self.cartesian[index1] + np.dot(where[0], self.cell)
        b = self.cartesian[index2] + np.dot(where[1], self.cell)
        c = self.cartesian[index3] + np.dot(where[2], self.cell)
        d = self.cartesian[index4] + np.dot(where[3], self.cell)

        vector1 = b - a
        vector2 = c - a
        vector3 = d - a

        cross1 = np.cross(vector1, vector2)
        return abs(sum(cross1 * vector3) / 6)

    def set_singles(self, xyz, what):

        index = what[0]
        return self.cartesian[index][xyz]

    def set_fsingles(self, xyz, what):

        index = what[0]
        x = self.cartesian[index]
        x = np.dot(x, np.linalg.inv(self.cell))
        return x[xyz]

    def set_llength(self, intwhat):

        index1 = intwhat[0]
        return (sum(self.cell[index1] * self.cell[index1])) ** 0.5

    def set_langle(self, intwhat):

        a = intwhat[0]
        b = intwhat[1]
        diffav = self.cell[a]
        diffbv = self.cell[b]
        d1 = np.linalg.norm(diffav)
        d2 = np.linalg.norm(diffbv)
        cosalpha = sum(diffav * diffbv) / (d1 * d2)
        return acos(cosalpha)

    def set_lbngle(self, intwhat):

        a = intwhat[0]
        b = intwhat[1]
        c = intwhat[2]
        av = self.cell[a]
        bv = self.cell[b]
        cv = self.cell[c]

        v1 = np.cross(av, bv)
        v2 = np.cross(av, cv)

        v1 = v1 / sum(v1 ** 2) ** 0.5
        v2 = v2 / sum(v2 ** 2) ** 0.5
        alpha = sum(v1 * v2)
        alpha = acos(alpha)
        return alpha

    def set_lvolume(self):

        l1 = self.cell[0]
        l2 = self.cell[1]
        l3 = self.cell[2]
        volume = np.cross(l1, l2)
        volume = sum(volume * l3)
        return volume

    def set_ratiollength(self, intwhat):

        index1 = intwhat[0]
        llengtha = (sum(self.cell[index1] * self.cell[index1])) ** 0.5
        index2 = intwhat[1]
        llengthb = (sum(self.cell[index2] * self.cell[index2])) ** 0.5
        return llengtha / llengthb

    def set_sum(self, coord):

        complexcoord = 0.0

        for i in range(len(coord.tag)):
            if coord.tag[i] == "X":
                dist = self.set_singles(0, coord.what[i])
            if coord.tag[i] == "Y":
                dist = self.set_singles(1, coord.what[i])
            if coord.tag[i] == "Z":
                dist = self.set_singles(2, coord.what[i])
            if coord.tag[i] == "fX":
                dist = self.set_fsingles(0, coord.what[i])
            if coord.tag[i] == "fY":
                dist = self.set_fsingles(1, coord.what[i])
            if coord.tag[i] == "fZ":
                dist = self.set_fsingles(2, coord.what[i])
            if coord.tag[i] == "R":
                dist = self.set_lengths(coord.what[i], coord.where[i])
            if coord.tag[i] == "M":
                dist = self.set_midlengths(coord.what[i], coord.where[i])
            if coord.tag[i] == "A":
                dist = self.set_angles(coord.what[i], coord.where[i])
            if coord.tag[i] == "T":
                dist = self.set_dihs(coord.what[i], coord.where[i])
            if coord.tag[i] == "tV":
                dist = self.set_tetrahedralVol(coord.what[i], coord.where[i])
            if coord.tag[i] == "LR":
                dist = self.set_llength(coord.what[i])
            if coord.tag[i] == "LA":
                dist = self.set_langle(coord.what[i])
            if coord.tag[i] == "RatioR":
                dist = self.set_ratior(coord.what[i], coord.where[i])
            if coord.tag[i] == "LR":
                dist = self.set_llength(coord.what[i])
            if coord.tag[i] == "LA":
                dist = self.set_langle(coord.what[i])
            if coord.tag[i] == "LV":
                dist = self.set_lvolume()
            if coord.tag[i] == "RatioLR":
                dist = self.set_ratiollength(coord.what[i])
            if coord.tag[i] == "hX":
                dist = self.cell[coord.what[i][0]][0]
            if coord.tag[i] == "hY":
                dist = self.cell[coord.what[i][0]][1]
            if coord.tag[i] == "hZ":
                dist = self.cell[coord.what[i][0]][2]

            complexcoord += coord.coefs[i] * dist
        return complexcoord

    def set_norm(self, coord):

        complexcoord = 0.0

        for i in range(len(coord.tag)):
            if coord.tag[i] == "X":
                dist = self.set_singles(0, coord.what[i])
            if coord.tag[i] == "Y":
                dist = self.set_singles(1, coord.what[i])
            if coord.tag[i] == "Z":
                dist = self.set_singles(2, coord.what[i])
            if coord.tag[i] == "R":
                dist = self.set_lengths(coord.what[i], coord.where[i])
            if coord.tag[i] == "M":
                dist = self.set_midlengths(coord.what[i], coord.where[i])
            if coord.tag[i] == "A":
                dist = self.set_angles(coord.what[i], coord.where[i])
            if coord.tag[i] == "T":
                dist = self.set_dihs(coord.what[i], coord.where[i])
            if coord.tag[i] == "LR":
                dist = self.set_llength(coord.what[i])
            if coord.tag[i] == "LA":
                dist = self.set_langle(coord.what[i])
            if coord.tag[i] == "RatioR":
                dist = self.set_ratior(coord.what[i], coord.where[i])
            complexcoord += (coord.coefs[i] * dist) ** 2

        complexcoord **= 0.5
        return complexcoord

    def set_cnum(self, coord):

        complexcoord = 0.0

        for i in range(len(coord.tag)):
            if coord.tag[i] == "R" and abs(coord.coefs[i]) > 1e-4:
                # dist=coord.value
                dist = self.set_lengths(coord.what[i], coord.where[i])
                dummyq = dist / coord.coefs[i]
                if abs(dummyq - 1.0) < 1.0e-4:
                    dummyq = 1.0001
                complexcoord += (1.0 - dummyq ** 9.0) / (1.0 - dummyq ** 14.0)

        return complexcoord
