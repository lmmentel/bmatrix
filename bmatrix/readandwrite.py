from __future__ import print_function

import string
import pickle

import numpy as np

from .physconstants import physical_constants as pC


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
    return [direct[i] for i in range(len(direct))]


def write_whatwhere(coords, fname):
    with open(fname, "w") as wfile:
        for i in range(len(coords)):
            toadd = coords[i].tag + chr(58)
            for j in range(len(coords[i].what)):
                toadd = toadd + chr(32) + str(coords[i].what[j])
            toadd = toadd + chr(58)
            for j in range(len(coords[i].where)):
                for k in range(len(coords[i].where[j])):
                    toadd = toadd + chr(32) + str(coords[i].where[j][k])
            wfile.write(toadd + "\n")


def read_whatwhere(fname):
    """Reads definition of primitive
    coordinate, i.e., the orders
    of the atoms involved in the coord.
    """
    with open(fname, "r") as wfile:
        inttags = []
        intwhat = []
        intwhere = []
        for line in wfile.readlines():
            line = string.split(line, chr(58))
            tagline = line[0]
            inttags.append(tagline)
            whatline = line[1].split()
            whereline = line[2].split()
            if whatline[0] == "None":
                whatline = [None]
            else:
                for i in range(len(whatline)):
                    whatline[i] = int(whatline[i])
            intwhat.append(whatline)
            if whereline[0] == "None":
                delement = [[None]]
            else:
                for i in range(len(whereline)):
                    whereline[i] = int(whereline[i])
                k = len(whereline) / 3
                delement = []
                for j in range(k):
                    element = np.array(
                        [whereline[j * 3], whereline[j * 3 + 1], whereline[j * 3 + 2]]
                    )
                    delement.append(element)
            intwhere.append(delement)
    return inttags, intwhat, intwhere


def read_cwhatwhere(fname, crt, lattmat):
    """Reads definition of primitive
    coordinate, i.e., the orders
    of the atoms involved in the coord.
    """
    with open(fname, "r") as wfile:
        inttags = []
        intwhat = []
        intwhere = []
        intcoefs = []
        intstat = []
        complextype = []
        ctag = 0
        for line in wfile.readlines():
            if ctag == 0:
                line = string.split(line, chr(58))
                if len(line) < 1:
                    break
                if line[0] == "Coefs":
                    ctag = 1
                    continue
                tagline = line[0]
                inttags.append(tagline)
                if tagline == "LV":
                    whatline = [None]
                    whereline = [[None]]
                    intwhat.append(whatline)
                    intwhere.append(whereline)
                    continue
                if tagline in [
                    "LA",
                    "LR",
                    "LB",
                    "RatioLR",
                    "hX",
                    "hY",
                    "hZ",
                    "X",
                    "Y",
                    "Z",
                    "fX",
                    "fY",
                    "fZ",
                ]:
                    whatline = line[1].split()
                    for i in range(len(whatline)):
                        whatline[i] = int(whatline[i])
                        whereline = [[None]]
                        intwhat.append(whatline)
                        intwhere.append(whereline)
                    continue

                whatline = line[1].split()
                for i in range(len(whatline)):
                    whatline[i] = int(whatline[i])
                intwhat.append(whatline)

                whereline = line[2].split()

                if whereline[0] == "S":  #! this is a shortest distance tag
                    if tagline in ["R", "IR1", "IR6"]:
                        delement = np.array(
                            various.shortest_dist(
                                crt, lattmat, whatline[0], whatline[1]
                            )
                        )
                    if tagline == "A":
                        delement1 = various.shortest_dist(
                            crt, lattmat, whatline[1], whatline[0]
                        )
                        delement2 = various.shortest_dist(
                            crt, lattmat, whatline[1], whatline[2]
                        )
                        delement = [delement1[1], delement1[0], delement2[1]]
                    if tagline == "M":
                        delement1 = various.shortest_dist(
                            crt, lattmat, whatline[0], whatline[1]
                        )
                        delement2 = various.shortest_dist(
                            crt, lattmat, whatline[1], whatline[2]
                        )
                        where3 = [0, 0, 0]
                        where3[0] = delement1[1][0] + delement2[1][0]
                        where3[1] = delement1[1][1] + delement2[1][1]
                        where3[2] = delement1[1][2] + delement2[1][2]
                        delement = [delement1[0], delement1[1], where3]
                        # delement=[delement1[0],delement1[1],delement2[1]]
                    if tagline == "T":
                        delement1 = various.shortest_dist(
                            crt, lattmat, whatline[1], whatline[0]
                        )
                        delement2 = various.shortest_dist(
                            crt, lattmat, whatline[1], whatline[2]
                        )
                        delement3 = various.shortest_dist(
                            crt, lattmat, whatline[2], whatline[3]
                        )
                        delement = [
                            delement1[1],
                            delement1[0],
                            delement2[1],
                            np.array(delement3[1]) + np.array(delement2[1]),
                        ]
                    if tagline == "tV":
                        delement1 = various.shortest_dist(
                            crt, lattmat, whatline[0], whatline[1]
                        )
                        delement2 = various.shortest_dist(
                            crt, lattmat, whatline[0], whatline[2]
                        )
                        delement3 = various.shortest_dist(
                            crt, lattmat, whatline[0], whatline[3]
                        )
                        delement = [
                            delement1[0],
                            delement1[1],
                            delement2[1],
                            delement3[1],
                        ]
                    if tagline == "RatioR":
                        delement1 = np.array(
                            various.shortest_dist(
                                crt, lattmat, whatline[0], whatline[1]
                            )
                        )
                        delement2 = np.array(
                            various.shortest_dist(
                                crt, lattmat, whatline[2], whatline[3]
                            )
                        )
                        delement = [
                            delement1[0],
                            delement1[1],
                            delement2[0],
                            delement2[1],
                        ]

                else:
                    for i in range(len(whereline)):
                        whereline[i] = int(whereline[i])
                    k = len(whereline) / 3
                    delement = []
                    for j in range(k):
                        element = np.array(
                            [
                                whereline[j * 3],
                                whereline[j * 3 + 1],
                                whereline[j * 3 + 2],
                            ]
                        )
                        delement.append(element)
                intwhere.append(delement)
            if ctag == 1:
                line = string.split(line, chr(58))
                if len(line) == 3:
                    status = string.split(line[1])
                    comptype = string.split(line[2])
                    intstat.append(status[0])
                    complextype.append(comptype[0])
                    coefs = string.split(line[0])
                    for i in range(len(coefs)):
                        coefs[i] = float(coefs[i])
                    intcoefs.append(coefs)
        if ctag == 0:
            intstat = len(inttags) * ["Keep"]
    return inttags, intwhat, intwhere, intcoefs, intstat, complextype


def write_poscar(filename, cartcoords1, lattmat_, inpt):
    cartcoords3 = np.array(cartcoords1).reshape(len(cartcoords1) / 3, 3)
    fractcoords3 = cart_dir(lattmat_, cartcoords3)
    # fractcoords3=dot(cartcoords3,inverse(lattmat_))
    lattmat = lattmat_ * pC["AU2A"]
    with open(filename, "w") as kkk:
        kkk.write(inpt.comment + "\n")
        kkk.write(str(1.000) + "\n")
        for i in range(len(lattmat)):
            row = (
                "   "
                + "%.12f" % (round(lattmat[i][0], 12))
                + "   "
                + "%.12f" % (round(lattmat[i][1], 12))
                + "   "
                + "%.12f" % (round(lattmat[i][2], 12))
                + "\n"
            )
            kkk.write(row)
        row = "".join(str(inpt.types[i]) + " " for i in range(inpt.ntypes))
        row += "\n"
        kkk.write(row)
        # kkk.write('cartesian \n')
        kkk.write("direct \n")
        for i in range(inpt.numofatoms):
            row = ""
            for j in range(3):
                # row=row+'   '+'%.12f'%(round(cartcoords3[i][j],12))
                row = row + "   " + "%.8f" % (round(fractcoords3[i][j], 8))
            kkk.write(row + "\n")


def read_store(what, NFREE):
    name = "store." + what
    with open(name, "r") as store:
        field = []
        for line in store.readlines():
            prefield = line.split()
            gi = [float(prefield[i]) for i in range(len(prefield))]
            field.append(gi)
    if len(field) > NFREE:
        field = field[len(field) - NFREE : len(field)]
    return field


def write_store(internalvalues, name):
    """Writes  to the end of
    store.x in which all optim. history
    is stored.
    """
    with open(name, "a") as store:
        toadd = ""
        for i in range(len(internalvalues)):
            toadd = toadd + chr(32) + str(internalvalues[i])
        store.write(toadd + "\n")


def write_pstore(coords, name):
    """Writes prim. internals to the end of
    store.xi in which all optim. history
    is stored.
    """
    with open(name, "a") as store:
        toadd = ""
        for i in range(len(coords)):
            toadd = toadd + chr(32) + str(coords[i].value)
        store.write(toadd + "\n")


def write_constrained(fname, xconstrained, numofatoms):
    with open(fname, "w") as cfile:
        for i in xconstrained:
            row = np.zeros(3 * numofatoms, dtype=float)
            row[i] = 1
            wrow = ""
            for _ in range(len(row)):
                wrow = wrow + "row[j]" + chr(32)
            cfile.write(wrow + "\n")


def read_constrained(cfile=""):
    cmodes = []
    if cfile != "":
        try:
            with open(cfile, "r") as cfile:
                for line in cfile.readlines():
                    prefield = line.split()
                    xxx = [float(prefield[i]) for i in range(len(prefield))]
                    cmodes.append(xxx)
        except IOError:
            return cmodes
        else:
            print("FILE WITH CONSTRAINED fractional coordinates read in")
    return cmodes


def read_atoms(cfile="ATOMS"):
    atoms = open(cfile, "r")
    line = atoms.readline()
    return line.split()


def read_matrix(cfile):
    with open(cfile, "r") as ufile:
        matrix = pickle.load(ufile)
    return matrix


def write_matrix(matrix, cfile):
    with open(cfile, "w") as ufile:
        pickle.dump(matrix, ufile)


def write_maxstep(MAXSTEP, cfile):
    with open(cfile, "w") as tfile:
        tfile.write("{0:15.10f}".format(MAXSTEP))


def read_maxstep(cfile):
    with open(cfile, "r") as tfile:
        MAXSTEP = tfile.readline()
    # MAXSTEP=MAXSTEP[:len(MAXSTEP)-1]
    MAXSTEP = string.strip(MAXSTEP)
    MAXSTEP = float(MAXSTEP)
    return MAXSTEP


def read_hmat():
    with open("hmat", "r") as hfile:
        prefield = hfile.readline()
        prefield = prefield.split()
        Hdim = int((len(prefield)) ** 0.5)
        Hmat = np.zeros((Hdim, Hdim), dtype=float)
        for i in range(Hdim):
            for j in range(Hdim):
                num = i * Hdim + j
                Hmat[i][j] = float(prefield[num])
    return np.array(Hmat)


def write_hmat(hnext):
    with open("hmat", "w") as hfile:
        for _ in range(len(hnext)):
            for _ in range(len(hnext)):
                hfile.write("hnext[i][j]" + chr(32))


def read_umat(cfile="UMAT"):
    utrans = []
    with open(cfile, "r") as ufile:
        for line in ufile.readlines():
            line = line.split()
            for i in range(len(line)):
                line[i] = float(line[i])
            utrans.append(line)
    utrans = np.array(utrans)
    return utrans


def write_umat(utrans, cfile="UMAT"):
    with open(cfile, "w") as ufile:
        for i in range(len(utrans)):
            row = ""
            for _ in range(len(utrans[0])):
                row = row + chr(32) + "utrans[i][j]"
            if i == len(utrans) - 1:
                ufile.write(row)
            else:
                ufile.write(row + "\n")


def read_translations(cfile):
    with open(cfile, "r") as ufile:
        translations = []
        for line in ufile.readlines():
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            translations.append(line)
    return translations
