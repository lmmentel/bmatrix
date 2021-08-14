from math import *
import numpy as np

from . import datastruct
from .physconstants import ANGS2BOHR


def get_bmatrix(atoms, internals, coordinates="cartesian"):
    """
    Calculate the B matrix

    Args:
        atoms : ase.Atoms
            Atoms must be sorted by species

        internals (list) :
            Internal coordinates as a list of ``Complextype`` objects
    """

    relax = False

    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    cartesian = atoms.get_positions() * ANGS2BOHR
    # convert and assign the cell and cartesian coordiantes
    cell = atoms.get_cell() * ANGS2BOHR
    cell_inv = np.linalg.inv(cell)

    # compute the Bmatrix (wrt. fractional coordinates!)
    b = Bmatrix(cartesian, internals, natoms, cell, relax)
    Bmat = b.Bmatrix

    if relax:
        transmat = np.zeros((ndof + 9, ndof + 9), dtype=float)
        for i in range(ndof + 9):
            transmat[i, i] = 1.0
        transmat[0:ndof, 0:ndof] = np.kron(np.eye(natoms), cell_inv.T)
    else:
        transmat = np.kron(np.eye(natoms), cell_inv.T)

    Bmat_c2 = np.dot(Bmat, transmat)

    if coordinates == "cartesian":
        Bmat = Bmat_c2

    return Bmat


class Bmatrix:
    """
    Jacobi matrix for x->q conversion

    Args:
        cartesian (array_like) :
            Cartesian coordinates (N x 3)

        coords () :
            Internal coordinates

        cell (array_like) :
            Lattice vectors in atomic units (3 x 3)

        natoms (int) :
            Number of atoms
    """

    def __init__(self, cartesian, coords, natoms, cell, relax):

        self.cartesian = cartesian
        self.cell = cell
        self.invcell = np.linalg.inv(cell)
        self.natoms = natoms

        dimdim1 = len(coords)

        if relax:
            dimdim2 = self.natoms * 3 + 9
        else:
            dimdim2 = self.natoms * 3

        self.Bmatrix = np.zeros((dimdim1, dimdim2), dtype=float)

        # derstep = 0.00001
        # self.prepare_lnum(cartesian,lengths,intwhat,intwhere,cell,derstep)
        # self.prepare_anum(cartesian,angles,lengths,intwhat,intwhere,cell,derstep)
        # self.prepare_dhnum(cartesian,dihs,angles,intwhat,intwhere,cell,derstep)

        for i in range(len(coords)):
            BROW = np.zeros((dimdim2), dtype=float)
            if coords[i].dtyp == "simple":
                if coords[i].tag == "X":
                    self.Bmatrix[i] = self.prepare_sing(coords[i], 0, BROW)
                elif coords[i].tag == "Y":
                    self.Bmatrix[i] = self.prepare_sing(coords[i], 1, BROW)
                elif coords[i].tag == "Z":
                    self.Bmatrix[i] = self.prepare_sing(coords[i], 2, BROW)
                elif coords[i].tag == "fX":
                    self.Bmatrix[i] = self.prepare_fsing(coords[i], 0, BROW)
                elif coords[i].tag == "fY":
                    self.Bmatrix[i] = self.prepare_fsing(coords[i], 1, BROW)
                elif coords[i].tag == "fZ":
                    self.Bmatrix[i] = self.prepare_fsing(coords[i], 2, BROW)
                elif coords[i].tag == "hX":
                    self.Bmatrix[i] = self.prepare_latsing(coords[i], 0, BROW)
                elif coords[i].tag == "hY":
                    self.Bmatrix[i] = self.prepare_latsing(coords[i], 1, BROW)
                elif coords[i].tag == "hZ":
                    self.Bmatrix[i] = self.prepare_latsing(coords[i], 2, BROW)
                elif coords[i].tag == "R":
                    self.Bmatrix[i] = self.prepare_l(coords[i], BROW)
                elif coords[i].tag == "M":
                    self.Bmatrix[i] = self.prepare_m(coords[i], BROW)
                elif coords[i].tag == "RatioR":
                    self.Bmatrix[i] = self.prepare_ratior(coords[i], BROW)
                elif coords[i].tag == "A":
                    self.Bmatrix[i] = self.prepare_a(coords[i], BROW)
                elif coords[i].tag == "T":
                    self.Bmatrix[i] = self.prepare_dh(coords[i], BROW)
                elif coords[i].tag == "tV":
                    self.Bmatrix[i] = self.prepare_tv(coords[i], BROW)
                elif coords[i].tag == "IR1":
                    self.Bmatrix[i] = self.prepare_ir1(coords[i], BROW)
                elif coords[i].tag == "IR6":
                    self.Bmatrix[i] = self.prepare_ir6(coords[i], BROW)

                if relax:
                    if coords[i].tag == "LR":
                        self.Bmatrix[i] = self.prepare_lr(coords[i], BROW)
                    elif coords[i].tag == "LA":
                        self.Bmatrix[i] = self.prepare_la(coords[i], BROW)
                    elif coords[i].tag == "LB":
                        self.Bmatrix[i] = self.prepare_lb(coords[i], BROW)
                    # elif inttags[i]=='LS':
                    elif coords[i].tag == "LV":
                        self.Bmatrix[i] = self.prepare_lv(BROW)
                    elif coords[i].tag == "RatioLR":
                        self.Bmatrix[i] = self.prepare_ratiolr(coords[i], BROW)

            if coords[i].dtyp == "sum":
                self.Bmatrix[i] = self.prepare_sum(coords[i], BROW)
            if coords[i].dtyp == "norm":
                self.Bmatrix[i] = self.prepare_norm(coords[i], BROW)
            if coords[i].dtyp == "cn":
                self.Bmatrix[i] = self.prepare_cnum(coords[i], BROW)

    def prepare_l(self, coord, BROW):
        """
        Calculates length components of the B matrix.
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        b = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        vector = a - b
        dist = np.linalg.norm(vector)
        dl_c = (a - b) / dist
        dl = np.dot(dl_c, np.transpose(self.cell))

        if coord.what[0] != coord.what[1]:
            column = coord.what[0] * 3
            BROW[column] = dl[0]
            BROW[column + 1] = dl[1]
            BROW[column + 2] = dl[2]
            column = coord.what[1] * 3
            BROW[column] = -dl[0]
            BROW[column + 1] = -dl[1]
            BROW[column + 2] = -dl[2]

        if len(BROW) == 3 * len(self.cartesian) + 9:
            cmat = np.zeros((2, 3), dtype=float)
            cmat[0] = a
            cmat[1] = b
            dmat = np.dot(cmat, self.invcell)  # -0.5
            for i in range(3):
                if dmat[0][i] > 1 or dmat[0][i] < 0.0:
                    dmat[:, i] -= dmat[0][i] - dmat[0][i] % 1
            dmat -= 0.5
            deriv = np.zeros((2, 3), dtype=float)
            deriv[0] = dl_c
            deriv[1] = -dl_c
            latderiv = np.dot(np.transpose(dmat), deriv)
            BROW[-9:-6] = latderiv[0]
            BROW[-6:-3] = latderiv[1]
            BROW[-3:] = latderiv[2]
        return BROW

    def prepare_m(self, coord, BROW):
        """
        Calculates length components of the B matrix.
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        b = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        # c=cartesian[coord.what[2]]+np.dot(coord.where[1],cell)+np.dot(coord.where[2],cell)
        c = self.cartesian[coord.what[2]] + np.dot(coord.where[2], self.cell)

        vector = a - (b + c) / 2
        m = (sum((vector) ** 2)) ** 0.5

        vector = (b + c) - 2 * a
        vector = vector / m

        dala_c = -vector / 2
        dalb_c = vector / 4
        dalc_c = vector / 4

        dala = np.dot(dala_c, np.transpose(self.cell))
        dalb = np.dot(dalb_c, np.transpose(self.cell))
        dalc = np.dot(dalc_c, np.transpose(self.cell))

        for i in range(3):
            BROW[(coord.what[1]) * 3 + i] = dalb[i]
            BROW[(coord.what[0]) * 3 + i] += dala[i]
            BROW[(coord.what[2]) * 3 + i] += dalc[i]

        if len(BROW) == 3 * len(self.cartesian) + 9:
            cmat = np.zeros((3, 3), dtype=float)
            cmat[0] = a
            cmat[1] = b
            cmat[2] = c
            dmat = np.dot(cmat, self.invcell)  # -0.5
            for i in range(3):
                if dmat[1][i] > 1 or dmat[1][i] < 0.0:
                    dmat[:, i] -= dmat[1][i] - dmat[1][i] % 1
            dmat -= 0.5
            # dmat=self.put_intocell(dmat)-0.5
            deriv = np.zeros((3, 3), dtype=float)
            deriv[0] = dala_c
            deriv[1] = dalb_c
            deriv[2] = dalc_c
            latderiv = np.dot(np.transpose(dmat), deriv)
            BROW[-9:-6] = latderiv[0]
            BROW[-6:-3] = latderiv[1]
            BROW[-3:] = latderiv[2]

        return BROW

    def prepare_ratior(self, coord, BROW):
        """
        Ratio between two bond lengths
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        b = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        c = self.cartesian[coord.what[2]] + np.dot(coord.where[2], self.cell)
        d = self.cartesian[coord.what[3]] + np.dot(coord.where[3], self.cell)
        vector1 = a - b
        vector2 = c - d
        dist1 = np.linalg.norm(vector1)
        dist2 = np.linalg.norm(vector2)
        tmp1 = coord.what
        tmp2 = coord.where
        coord_ = coord
        coord_.what = tmp1[:2]
        coord_.where = tmp2[:2]
        dr1 = self.prepare_l(coord_, np.zeros(len(BROW), dtype=float))
        coord_.what = tmp1[2:]
        coord_.where = tmp2[2:]
        dr2 = self.prepare_l(coord_, np.zeros(len(BROW), dtype=float))
        coord.what = tmp1
        coord.where = tmp2
        BROW = dr1 / dist2 - (dist1 / dist2 ** 2) * dr2

        return BROW

    def prepare_ir1(self, coord, BROW):
        """
        Inverse power coordinate
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        b = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        vector = a - b
        dist = np.linalg.norm(vector)
        BROW = self.prepare_l(coord, BROW)
        BROW = -5 * BROW / dist ** 2
        return BROW

    def prepare_ir6(self, coord, BROW):
        """
        Another inverse power coordinate
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        b = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        vector = a - b
        dist = np.linalg.norm(vector)
        BROW = self.prepare_l(coord, BROW)
        BROW = -12000 * BROW / dist ** 7
        return BROW

    def prepare_a(self, coord, BROW):
        """
        Calculates angular components of the B matrix.
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        v = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        b = self.cartesian[coord.what[2]] + np.dot(coord.where[2], self.cell)
        diffav = a - v
        diffbv = b - v

        d1 = np.linalg.norm(diffav)
        d2 = np.linalg.norm(diffbv)
        cosalpha = sum(diffav * diffbv) / (d1 * d2)
        alpha = acos(cosalpha)

        sinalpha = sin(alpha)
        dala_c = -(diffbv / (d1 * d2) - diffav * cosalpha / (d1 ** 2)) / sinalpha
        dalc_c = -(diffav / (d1 * d2) - diffbv * cosalpha / (d2 ** 2)) / sinalpha
        dalb_c = -dala_c - dalc_c

        dala = np.dot(dala_c, np.transpose(self.cell))
        dalb = np.dot(dalb_c, np.transpose(self.cell))
        dalc = np.dot(dalc_c, np.transpose(self.cell))

        for i in range(3):
            BROW[(coord.what[1]) * 3 + i] = dalb[i]
            BROW[(coord.what[0]) * 3 + i] += dala[i]
            BROW[(coord.what[2]) * 3 + i] += dalc[i]

            # if coord.what[0]==coord.what[2]:
            #  BROW[(coord.what[0])*3+i]=dala[i]+dalc[i]
            # else:
            #  BROW[(coord.what[0])*3+i]=dala[i]
            #  BROW[(coord.what[2])*3+i]=dalc[i]

        if len(BROW) == 3 * len(self.cartesian) + 9:
            cmat = np.zeros((3, 3), dtype=float)
            cmat[0] = a
            cmat[1] = v
            cmat[2] = b

            dmat = np.dot(cmat, self.invcell)  # -0.5
            for i in range(3):
                if dmat[1][i] > 1 or dmat[1][i] < 0.0:
                    dmat[:, i] -= dmat[1][i] - dmat[1][i] % 1
            dmat -= 0.5
            # dmat=self.put_intocell(dmat)-0.5
            deriv = np.zeros((3, 3), dtype=float)
            deriv[0] = dala_c
            deriv[1] = dalb_c
            deriv[2] = dalc_c
            latderiv = np.dot(np.transpose(dmat), deriv)
            BROW[-9:-6] = latderiv[0]
            BROW[-6:-3] = latderiv[1]
            BROW[-3:] = latderiv[2]
        return BROW

    def prepare_dh(self, coord, BROW):
        """
        Calculates dihedral components of the B matrix.
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        b = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        c = self.cartesian[coord.what[2]] + np.dot(coord.where[2], self.cell)
        d = self.cartesian[coord.what[3]] + np.dot(coord.where[3], self.cell)
        r12 = a - b
        r23 = b - c
        r34 = c - d
        dr12 = np.linalg.norm(r12)
        dr23 = np.linalg.norm(r23)
        dr34 = np.linalg.norm(r34)
        cospsi2 = sum((r12) * (r23)) / (dr12 * dr23)
        sinpsi2 = sin(acos(cospsi2))
        cospsi3 = sum((r23) * (r34)) / (dr23 * dr34)
        sinpsi3 = sin(acos(cospsi3))
        e12 = r12 / dr12  # unit vector r12
        e23 = r23 / dr23  # unit vector r23
        e34 = r34 / dr34  # unit vector r34
        e12xe23 = np.cross(e12, e23)
        e23xe34 = np.cross(e23, e34)
        st1_c = -e12xe23 / (dr12 * sinpsi2 ** 2)
        st4_c = e23xe34 / (dr34 * sinpsi3 ** 2)
        part1 = (dr23 + dr12 * cospsi2) / (dr12 * dr23 * sinpsi2)
        part2 = e12xe23 / sinpsi2
        part3 = cospsi3 / (dr23 * sinpsi3)
        part4 = e23xe34 / sinpsi3
        st2_c = part1 * part2 + part3 * part4
        part1 = (dr23 + dr34 * cospsi3) / (dr34 * dr23 * sinpsi3)
        part2 = e23xe34 / sinpsi3
        part3 = cospsi2 / (dr23 * sinpsi2)
        part4 = e12xe23 / sinpsi2
        st3_c = -part1 * part2 - part3 * part4

        st1 = np.dot(st1_c, np.transpose(self.cell))
        st2 = np.dot(st2_c, np.transpose(self.cell))
        st3 = np.dot(st3_c, np.transpose(self.cell))
        st4 = np.dot(st4_c, np.transpose(self.cell))

        for i in range(3):
            BROW[(coord.what[0] * 3) + i] = st1[i]
            BROW[(coord.what[1] * 3) + i] = st2[i]
            BROW[(coord.what[2] * 3) + i] = st3[i]
            BROW[(coord.what[3] * 3) + i] = st4[i]
        if len(BROW) == 3 * len(self.cartesian) + 9:
            cmat = np.zeros((4, 3), dtype=float)
            cmat[0] = a
            cmat[1] = b
            cmat[2] = c
            cmat[3] = d
            dmat = np.dot(cmat, self.invcell)  # -0.5
            for i in range(3):
                if dmat[1][i] > 1 or dmat[1][i] < 0.0:
                    dmat[:, i] -= dmat[1][i] - dmat[1][i] % 1
            dmat -= 0.5

            # dmat=self.put_intocell(dmat)-0.5
            deriv = np.zeros((4, 3), dtype=float)
            deriv[0] = st1_c
            deriv[1] = st2_c
            deriv[2] = st3_c
            deriv[3] = st4_c
            latderiv = np.dot(np.transpose(dmat), deriv)
            BROW[-9:-6] = latderiv[0]
            BROW[-6:-3] = latderiv[1]
            BROW[-3:] = latderiv[2]
        return BROW

    def prepare_tv(self, coord, BROW):
        """
        Calculates volume of tetrahedron components of the B matrix.
        """

        a = self.cartesian[coord.what[0]] + np.dot(coord.where[0], self.cell)
        b = self.cartesian[coord.what[1]] + np.dot(coord.where[1], self.cell)
        c = self.cartesian[coord.what[2]] + np.dot(coord.where[2], self.cell)
        d = self.cartesian[coord.what[3]] + np.dot(coord.where[3], self.cell)
        vector1 = b - a
        vector2 = c - a
        vector3 = d - a

        st2_c = np.cross(vector2, vector3) / 6
        st3_c = np.cross(vector3, vector1) / 6
        st4_c = np.cross(vector1, vector2) / 6
        st1_c = -(st2_c + st3_c + st4_c)

        # volume must be positive - change sign if needed
        tv = sum(st4_c * vector3)
        if tv < 0.0:
            st1_c = -st1_c
            st2_c = -st2_c
            st3_c = -st3_c
            st4_c = -st4_c

        st1 = np.dot(st1_c, np.transpose(self.cell))
        st2 = np.dot(st2_c, np.transpose(self.cell))
        st3 = np.dot(st3_c, np.transpose(self.cell))
        st4 = np.dot(st4_c, np.transpose(self.cell))

        for i in range(3):
            BROW[(coord.what[0] * 3) + i] = st1[i]
            BROW[(coord.what[1] * 3) + i] = st2[i]
            BROW[(coord.what[2] * 3) + i] = st3[i]
            BROW[(coord.what[3] * 3) + i] = st4[i]
        if len(BROW) == 3 * len(self.cartesian) + 9:
            cmat = np.zeros((4, 3), dtype=float)
            cmat[0] = a
            cmat[1] = b
            cmat[2] = c
            cmat[3] = d
            dmat = np.dot(cmat, self.invcell)  # -0.5
            for i in range(3):
                if dmat[1][i] > 1 or dmat[1][i] < 0.0:
                    dmat[:, i] -= dmat[1][i] - dmat[1][i] % 1
            dmat -= 0.5

            # dmat=self.put_intocell(dmat)-0.5
            deriv = np.zeros((4, 3), dtype=float)
            deriv[0] = st1_c
            deriv[1] = st2_c
            deriv[2] = st3_c
            deriv[3] = st4_c
            latderiv = np.dot(np.transpose(dmat), deriv)
            BROW[-9:-6] = latderiv[0]
            BROW[-6:-3] = latderiv[1]
            BROW[-3:] = latderiv[2]
        return BROW

    def prepare_lr(self, coord, BROW):
        """
        internal coordinate: Length of the lattice vector
        """

        a = coord.what[0]
        dist = np.linalg.norm(self.cell[a])
        dl = self.cell[a] / dist
        if a == 0:
            BROW[-9:-6] = dl
        if a == 1:
            BROW[-6:-3] = dl
        if a == 2:
            BROW[-3:] = dl
        return BROW

    def prepare_la(self, coord, BROW):
        """
        internal coordinate: Angel between two lattice vectors
        """

        a = coord.what[0]
        b = coord.what[1]
        diffav = self.cell[a]
        diffbv = self.cell[b]
        d1 = np.linalg.norm(diffav)
        d2 = np.linalg.norm(diffbv)
        cosalpha = sum(diffav * diffbv) / (d1 * d2)
        alpha = acos(cosalpha)
        sinalpha = sin(alpha)
        dala = -(diffbv / (d1 * d2) - diffav * cosalpha / (d1 ** 2)) / sinalpha
        dalc = -(diffav / (d1 * d2) - diffbv * cosalpha / (d2 ** 2)) / sinalpha
        if a == 0:
            BROW[-9:-6] = dala
        elif a == 1:
            BROW[-6:-3] = dala
        elif a == 2:
            BROW[-3:] = dala
        if b == 0:
            BROW[-9:-6] = dalc
        elif b == 1:
            BROW[-6:-3] = dalc
        elif b == 2:
            BROW[-3:] = dalc
        return BROW

    def prepare_lb(self, coord, BROW):
        """
        internal coordinate: Angel betweentwo lattice vectors
        """

        ia = coord.what[0]
        ib = coord.what[1]
        ic = coord.what[2]
        a = self.cell[ia]
        b = self.cell[ib]
        c = self.cell[ic]

        aa = sum(a ** 2)
        ba = sum(b * a)
        bc = sum(b * c)
        ac = sum(a * c)

        b1 = 2 * a * bc - c * ba - b * ac
        b2 = c * aa - a * ac
        b3 = b * aa - a * ba

        axb = np.cross(a, b)
        axc = np.cross(a, c)

        naxb = sum(axb ** 2) ** 0.5
        naxc = sum(axc ** 2) ** 0.5

        calpha = sum(axb * axc) / naxb / naxc
        alpha = acos(calpha)
        salpha = sin(alpha)

        b1 /= naxb * naxc
        b2 /= naxb * naxc
        b3 /= naxb * naxc

        daxb_da = np.transpose(
            np.array([[0.0, -b[2], b[1]], [b[2], 0.0, -b[0]], [-b[1], b[0], 0.0]])
        )
        daxb_da = np.dot(axb, daxb_da) / naxb
        c1 = daxb_da * naxc * sum(axb * axc) / (naxb * naxc) ** 2

        daxb_db = np.transpose(
            np.array([[0.0, a[2], -a[1]], [-a[2], 0.0, a[0]], [a[1], -a[0], 0.0]])
        )
        daxb_db = np.dot(axb, daxb_db) / naxb
        c2 = daxb_db * naxc * sum(axb * axc) / (naxb * naxc) ** 2

        c3 = np.zeros(3, dtype=float)

        daxc_da = np.transpose(
            np.array([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
        )
        daxc_da = np.dot(axc, daxc_da) / naxc
        d1 = daxc_da * naxb * sum(axb * axc) / (naxb * naxc) ** 2

        d2 = np.zeros(3, dtype=float)

        daxc_dc = np.transpose(
            np.array([[0.0, a[2], -a[1]], [-a[2], 0.0, a[0]], [a[1], -a[0], 0.0]])
        )
        daxc_dc = np.dot(axc, daxc_dc) / naxc
        d3 = daxc_dc * naxb * sum(axb * axc) / (naxb * naxc) ** 2

        ind1 = -9
        ind2 = -6
        if ia < 2:
            BROW[ind1 + 3 * ia : ind2 + 3 * ia] = b1 - c1 - d1
        else:
            BROW[-3:] = b1 - c1 - d1
        if ib < 2:
            BROW[ind1 + 3 * ib : ind2 + 3 * ib] = b2 - c2 - d2
        else:
            BROW[-3:] = b2 - c2 - d2
        if ic < 2:
            BROW[ind1 + 3 * ic : ind2 + 3 * ic] = b3 - c3 - d3
        else:
            BROW[-3:] = b3 - c3 - d3
        BROW /= -salpha
        return BROW

    def prepare_lv(self, BROW):
        """
        internal coordinate: Volume of cell
        """
        l1 = self.cell[0]
        l2 = self.cell[1]
        l3 = self.cell[2]
        dl1 = np.cross(l2, l3)
        dl2 = np.cross(l3, l1)
        dl3 = np.cross(l1, l2)
        BROW[-9:-6] = dl1
        BROW[-6:-3] = dl2
        BROW[-3:] = dl3
        return BROW

    def prepare_ratiolr(self, coord, BROW):
        """
        internal coordinate: Length of the lattice vector
        """

        a = coord.what[0]
        b = coord.what[1]
        dista = np.linalg.norm(self.cell[a])
        distb = np.linalg.norm(self.cell[b])
        dla = self.cell[a] / (dista * distb)
        dlb = -self.cell[b] * (dista / distb ** 3)
        if a == 0:
            BROW[-9:-6] = dla
        if a == 1:
            BROW[-6:-3] = dla
        if a == 2:
            BROW[-3:] = dla
        if b == 0:
            BROW[-9:-6] = dlb
        if b == 1:
            BROW[-6:-3] = dlb
        if b == 2:
            BROW[-3:] = dlb
        return BROW

    def prepare_lnum(self, lengths, intwhat, intwhere, derstep):
        """
        Calculates length components of the B matrix numericaly.
        """

        for i in range(lengths):
            a = self.cartesian[intwhat[i][0]]
            b = self.cartesian[intwhat[i][1]] + np.dot(intwhere[i][1], self.cell)
            distnul = self.calculate_le(a, b)
            for j in range(3):
                dershift = np.array([0.0, 0.0, 0.0])
                dershift[j] = derstep
                a1 = a + dershift
                dist1 = self.calculate_le(a1, b)
                column = intwhat[i][0] * 3
                self.Bmatrix[i][column + j] = (dist1 - distnul) / derstep
                b1 = b + dershift
                dist1 = self.calculate_le(a, b1)
                column = intwhat[i][1] * 3
                self.Bmatrix[i][column + j] = (dist1 - distnul) / derstep

    def prepare_anum(self, what, where, BROW, derstep):
        """
        Calculates angular components of the B matrix numericaly.
        """

        a = self.cartesian[what[0]] + np.dot(where[0], self.cell)
        v = self.cartesian[what[1]]  # apex atom, allways in [0,0,0]
        b = self.cartesian[what[2]] + np.dot(where[2], self.cell)
        alphanul = self.calculate_an(a, v, b)
        for j in range(3):
            dershift = np.array([0.0, 0.0, 0.0])
            dershift[j] = derstep
            a1 = a + dershift
            alpha1 = self.calculate_an(a1, v, b)
            BROW[(what[0] * 3) + j] = -(alphanul - alpha1) / (derstep)
            v1 = v + dershift
            alpha1 = self.calculate_an(a, v1, b)
            BROW[(what[1] * 3) + j] = -(alphanul - alpha1) / (derstep)
            b1 = b + dershift
            alpha1 = self.calculate_an(a, v, b1)
            BROW[(what[2] * 3) + j] = -(alphanul - alpha1) / (derstep)
        return BROW

    def prepare_dhnum(self, what, where, BROW, derstep):
        """
        Calculates dihedral components of the B matrix numericaly.
        """

        a = self.cartesian[what[0]] + np.dot(where[0], self.cell)
        b = self.cartesian[what[1]]  # allways in [0,0,0]
        c = self.cartesian[what[2]] + np.dot(where[2], self.cell)
        d = self.cartesian[what[3]] + np.dot(where[3], self.cell)
        dihnul = self.calculate_da(a, b, c, d)

        for j in range(3):
            dershift = np.array([0.00, 0.0, 0.0])
            dershift[j] = derstep
            a1 = a + dershift
            a2 = a - dershift
            dih1 = self.calculate_da(a1, b, c, d)
            dih2 = self.calculate_da(a2, b, c, d)
            # BROW[(what[0]*3)+j]=(dih1-dih2)/(2*derstep)
            BROW[(what[0] * 3) + j] = -(dihnul - dih1) / (derstep)
            b1 = b + dershift
            b2 = b - dershift
            dih1 = self.calculate_da(a, b1, c, d)
            dih2 = self.calculate_da(a, b2, c, d)
            # BROW[(what[1]*3)+j]=(dih1-dih2)/(2*derstep)
            BROW[(what[1] * 3) + j] = -(dihnul - dih1) / (derstep)
            c1 = c + dershift
            c2 = c - dershift
            dih1 = self.calculate_da(a, b, c1, d)
            dih2 = self.calculate_da(a, b, c2, d)
            # BROW[(what[2]*3)+j]=(dih1-dih2)/(2*derstep)
            BROW[(what[2] * 3) + j] = -(dihnul - dih1) / (derstep)
            d1 = d + dershift
            d2 = d - dershift
            dih1 = self.calculate_da(a, b, c, d1)
            dih2 = self.calculate_da(a, b, c, d2)
            # BROW[(what[3]*3)+j]=(dih1-dih2)/(2*derstep)
            BROW[(what[3] * 3) + j] = -(dihnul - dih1) / (derstep)
        return BROW

    def calculate_le(self, a, b):
        vector = a - b
        distance = np.linalg.norm(vector)
        return distance

    def calculate_an(self, a, v, b):
        diffav = a - v
        diffbv = b - v
        d1 = np.linalg.norm(diffav)
        d2 = np.linalg.norm(diffbv)
        cosalpha = sum(diffav * diffbv) / (d1 * d2)
        alpha = acos(cosalpha)
        return alpha

    def calculate_da(self, a, b, c, d):
        r12 = a - b
        r23 = b - c
        r34 = c - d
        dr12 = np.linalg.norm(r12)
        dr23 = np.linalg.norm(r23)
        dr34 = np.linalg.norm(r34)
        e12 = r12 / dr12  # unit vector r12
        e23 = r23 / dr23  # unit vector r23
        e34 = r34 / dr34  # unit vector r34
        e12xe23 = np.cross(e12, e23)
        e23xe34 = np.cross(e23, e34)
        # if e12xe23!=0 and e23xe34!=0:
        fuck = sum(e12xe23 * e23xe34) / (
            np.linalg.norm(e12xe23) * np.linalg.norm(e23xe34)
        )
        if fuck > 1:
            fuck = 1.0
        if fuck < -1:
            fuck = -1.0
        dangle = acos(fuck)
        if sum(e12xe23 * e34) >= 0:
            dangle = -dangle
        return dangle

    def prepare_sing(self, coord, xyz, BROW):
        """
        Adds elements corresponding to pure cartesian coordinates.
        """

        ddd_c = np.zeros((1, 3), dtype=float)
        ddd_c[0][xyz] = 1.0
        ddd = np.dot(ddd_c, np.transpose(self.cell))
        indx = coord.what[0] * 3
        a = self.cartesian[coord.what[0], xyz]
        BROW[indx : indx + 3] = ddd[0]
        if len(BROW) == 3 * len(self.cartesian) + 9:
            cmat = np.zeros((1, 3), dtype=float)
            cmat[0][xyz] = a
            dmat = np.dot(cmat, self.invcell)  # -0.5
            # dmat=self.put_intocell(dmat)-0.5
            deriv = np.zeros((1, 3), dtype=float)
            deriv[0][xyz] = 1.0
            latderiv = np.dot(np.transpose(dmat), deriv)  # *0
            BROW[-9:-6] = latderiv[0]
            BROW[-6:-3] = latderiv[1]
            BROW[-3:] = latderiv[2]
        return BROW

    def prepare_fsing(self, coord, xyz, BROW):
        """
        Adds elements corresponding to pure fractional coordinates.
        """

        indx = coord.what[0] * 3
        BROW[indx + xyz] = 1.0
        return BROW

    def put_intocell(self, dmat):
        for i in range(3):
            if dmat[0][i] < 0.0:
                dmat[0][i] += 1
            if dmat[0][i] >= 1.0:
                dmat[0][i] -= 1
        return dmat

    def prepare_latsing(self, coord, xyz, BROW):
        """
        Adds elements corresponding to pure cartesian coordinates.
        """

        numb = 9 - (3 * coord.what[0] + xyz)
        BROW[-numb] = 1.0
        return BROW

    def prepare_sum(self, coord, BROW):
        # XROW=np.zeros(len(BROW),dtype=float)
        for i in range(len(coord.tag)):
            XROW = np.zeros(len(BROW), dtype=float)
            tcoord = datastruct.Complextype(
                "simple",
                [1],
                coord.tag[i],
                coord.what[i],
                [None],
                coord.where[i],
                0.0,
                "free",
            )
            if coord.tag[i] == "X":
                BROW = BROW + coord.coefs[i] * self.prepare_sing(tcoord, 0, XROW)
            if coord.tag[i] == "Y":
                BROW = BROW + coord.coefs[i] * self.prepare_sing(tcoord, 1, XROW)
            if coord.tag[i] == "Z":
                BROW = BROW + coord.coefs[i] * self.prepare_sing(tcoord, 2, XROW)
            if coord.tag[i] == "fX":
                BROW = BROW + coord.coefs[i] * self.prepare_fsing(tcoord, 0, XROW)
            if coord.tag[i] == "fY":
                BROW = BROW + coord.coefs[i] * self.prepare_fsing(tcoord, 1, XROW)
            if coord.tag[i] == "fZ":
                BROW = BROW + coord.coefs[i] * self.prepare_fsing(tcoord, 2, XROW)
            if coord.tag[i] == "R":
                BROW = BROW + coord.coefs[i] * self.prepare_l(tcoord, XROW)
            if coord.tag[i] == "M":
                BROW = BROW + coord.coefs[i] * self.prepare_m(tcoord, XROW)
            if coord.tag[i] == "A":
                BROW = BROW + coord.coefs[i] * self.prepare_a(tcoord, XROW)
            if coord.tag[i] == "T":
                BROW = BROW + coord.coefs[i] * self.prepare_dh(tcoord, XROW)
            if coord.tag[i] == "tV":
                BROW = BROW + coord.coefs[i] * self.prepare_tv(tcoord, XROW)
            if coord.tag[i] == "RatioR":
                # BROW=BROW+coord.coefs[i]*self.prepare_ratior(cartesian,tcoord,cell,invcell,BROW)
                BROW = BROW + coord.coefs[i] * self.prepare_ratior(tcoord, XROW)
            if coord.tag[i] == "LR":
                BROW = BROW + coord.coefs[i] * self.prepare_lr(tcoord, XROW)
            if coord.tag[i] == "LA":
                BROW = BROW + coord.coefs[i] * self.prepare_la(tcoord, XROW)
            if coord.tag[i] == "LV":
                BROW = BROW + coord.coefs[i] * self.prepare_lv(XROW)
            if coord.tag[i] == "RatioLR":
                BROW = BROW + coord.coefs[i] * self.prepare_ratiolr(tcoord, XROW)
            if coord.tag[i] == "hX":
                BROW = BROW + coord.coefs[i] * self.prepare_latsing(tcoord, 0, XROW)
            if coord.tag[i] == "hY":
                BROW = BROW + coord.coefs[i] * self.prepare_latsing(tcoord, 1, XROW)
            if coord.tag[i] == "hZ":
                BROW = BROW + coord.coefs[i] * self.prepare_latsing(tcoord, 2, XROW)
        return BROW

    def prepare_norm(self, coord, BROW):

        complexcoord = 0.0
        for i in range(len(coord.tag)):
            tcoord = datastruct.Complextype(
                "simple",
                [1],
                coord.tag[i],
                coord.what[i],
                [None],
                coord.where[i],
                0.0,
                "free",
            )
            if coord.tag[i] == "X":
                dist = self.cartesian[tcoord.what[0]][0]
                complexcoord += (coord.coefs[i] ** 2) * dist
                BROW = BROW + (coord.coefs[i] ** 2) * dist * self.prepare_sing(
                    tcoord, 0, BROW
                )
            if coord.tag[i] == "Y":
                dist = self.cartesian[tcoord.what[0]][1]
                complexcoord += (coord.coefs[i] ** 2) * dist
                BROW = BROW + (coord.coefs[i] ** 2) * dist * self.prepare_sing(
                    tcoord, 1, BROW
                )
            if coord.tag[i] == "Z":
                dist = self.cartesian[tcoord.what[0]][2]
                complexcoord += (coord.coefs[i] ** 2) * dist
                BROW = BROW + (coord.coefs[i] ** 2) * dist * self.prepare_sing(
                    tcoord, 2, BROW
                )
            if coord.tag[i] == "R":
                a = self.cartesian[tcoord.what[0]] + np.dot(tcoord.where[0], self.cell)
                b = self.cartesian[tcoord.what[1]] + np.dot(tcoord.where[1], self.cell)
                vector = a - b
                dist = np.linalg.norm(vector)
                complexcoord += (coord.coefs[i] ** 2) * dist
                BROW = BROW + (coord.coefs[i] ** 2) * dist * self.prepare_l(
                    tcoord, np.zeros(len(BROW), dtype=float)
                )
            if coord.tag[i] == "A":
                a = self.cartesian[tcoord.what[0]] + np.dot(tcoord.where[0], self.cell)
                v = self.cartesian[tcoord.what[1]] + np.dot(tcoord.where[1], self.cell)
                b = self.cartesian[tcoord.what[2]] + np.dot(tcoord.where[2], self.cell)
                diffav = a - v
                diffbv = b - v
                d1 = np.linalg.norm(diffav)
                d2 = np.linalg.norm(diffbv)
                cosalpha = sum(diffav * diffbv) / (d1 * d2)
                alpha = acos(cosalpha)
                complexcoord += (coord.coefs[i] ** 2) * alpha
                BROW = BROW + (coord.coefs[i] ** 2) * alpha * self.prepare_a(
                    tcoord, np.zeros(len(BROW), dtype=float)
                )
            if coord.tag[i] == "T":
                a = self.cartesian[tcoord.what[0]] + np.dot(tcoord.where[0], self.cell)
                b = self.cartesian[tcoord.what[1]] + np.dot(tcoord.where[1], self.cell)
                c = self.cartesian[tcoord.what[2]] + np.dot(tcoord.where[2], self.cell)
                d = self.cartesian[tcoord.what[3]] + np.dot(tcoord.where[3], self.cell)
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
                complexcoord += (coord.coefs[i] ** 2) * dangle
                BROW = BROW + (coord.coefs[i] ** 2) * dangle * self.prepare_dh(
                    tcoord, np.zeros(len(BROW), dtype=float)
                )
        BROW = BROW / complexcoord ** (0.5)
        return BROW

    def prepare_cnum(self, coord, BROW):

        for i in range(len(coord.tag)):
            if coord.tag[i] == "R":
                if abs(coord.coefs[i]) > 1e-4:
                    tcoord = datastruct.Complextype(
                        "simple",
                        [1],
                        coord.tag[i],
                        coord.what[i],
                        [None],
                        coord.where[i],
                        0.0,
                        "free",
                    )
                    a = self.cartesian[tcoord.what[0]] + np.dot(
                        tcoord.where[0], self.cell
                    )
                    b = self.cartesian[tcoord.what[1]] + np.dot(
                        tcoord.where[1], self.cell
                    )
                    vector = a - b
                    dist = np.linalg.norm(vector)
                    dummyA = dist / coord.coefs[i]
                    if abs(dummyA - 1.0) < 1e-4:
                        dummyA = 1.0001
                    dummyC = 1.0 - dummyA ** 9
                    dummyD = 1.0 - dummyA ** 14
                    BROW_ = self.prepare_l(tcoord, np.zeros(len(BROW), dtype=float))
                    BROW = BROW - 9.0 * BROW_ * (dummyA ** 9.0 / dist) / dummyD
                    BROW = (
                        BROW
                        + 14.0 * dummyC * BROW_ * (dummyA ** 14.0 / dist) / dummyD ** 2
                    )
        return BROW
