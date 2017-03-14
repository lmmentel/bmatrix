
from __future__ import print_function

import os
import sys
import time
import pickle
from collections import Counter, OrderedDict

import numpy as np


from string import *
from fpformat import *

import bmatrix
import intcoord
import takeinpPOSCAR
import dealxyz
import datastruct
import inputparser

from physconstants import ANGS2BOHR

# c the smallest acceptable non-zero matrix element
MYTINY = 1e-6

#num_cores = multiprocessing.cpu_count()
#print("numCores = " + str(num_cores))


class OrderedCounter(Counter, OrderedDict):
    pass


def read_internals(fname):
    'Read internal coordinates from a file'

    with open(fname, 'r') as fpkl:
        internals = pickle.load(fpkl)
    return internals


def write_internals(internals, fname):
    'Write internal coordinates to a file'

    with open(fname, 'w') as fpkl:
        pickle.dump(internals, fpkl)


def get_internals(atoms, return_bmatrix=False, ascale=1.0, bscale=2.0,
                  anglecrit=6, torsioncrit=4, fragcoord=1, torsions=True,
                  radii='default'):
    '''
    Calculate the internal coordinates and optionally the B matrix
    for the ``atoms`` object

    Args:
        atoms : ase.Atoms
            Atoms must be sorted by species
        return_bmatrix : bool
            If ``True`` Bmatrix will also be calculated and returned as
            numpy.array
    '''

    cartesian = atoms.get_positions() * ANGS2BOHR

    # set frequently used variables
    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    counts = OrderedCounter(atoms.get_chemical_symbols())
    atomtypes = [s for s in counts.keys()]
    atomcounts = counts.values()

    # convert and assign the cell and cartesian coordiantes
    cell = atoms.get_cell() * ANGS2BOHR
    cell_inv = np.linalg.inv(cell)

    cov_radii = datastruct.get_covalent_radii(atomtypes, source=radii)

    # arguments that are normally set through arguments, here set by hand
    relax = False
    subst = 100
    coordinates = 'cartesian'

    if os.path.exists('internals.pkl'):
        primcoords = read_internals('internals.pkl')

        deal = dealxyz.Dealxyz(cartesian, primcoords, cell)
        for i in range(len(primcoords)):
            primcoords[i].value = deal.internals[i]
        print('Internal coordinates were read from the file: internals.pkl')
    else:
        intrn = intcoord.Internals(atoms, cov_radii, ascale, bscale,
                                   anglecrit, torsioncrit, fragcoord,
                                   relax, torsions, subst)

        primcoords = intrn.internalcoords
        write_internals(primcoords, 'internals.pkl')
        print('Internal coordinates were newly generated, save to: internals.pkl')

    internals = np.array([(p.tag, p.value) for p in primcoords],
                         dtype=[('type', 'S4'), ('value', np.float32)])

    if return_bmatrix:

        # now compute the Bmatrix (wrt. fractional coordinates!)
        b = bmatrix.Bmatrix(cartesian, primcoords, natoms, cell, relax)
        Bmat = b.Bmatrix

        if relax:
            transmat = np.zeros((ndof + 9, ndof + 9), dtype=float)
            for i in range(ndof + 9):
                transmat[i, i] = 1.0
            transmat[0: ndof, 0: ndof] = np.kron(np.eye(natoms), cell_inv.T)
        else:
            transmat = np.kron(np.eye(natoms), cell_inv.T)

        Bmat_c2 = np.dot(Bmat, transmat)

        if coordinates == 'cartesian':
            Bmat = Bmat_c2

        return internals, Bmat
    else:
        return internals


def main():
    'Main program'

    args = inputparser.get_input_args()

    inpt = takeinpPOSCAR.TakeInput()
    inpt.read(args.filename)
    inpt.convert_to_au()
    lattmat = inpt.lattmat
    lattinv = inpt.lattinv
    volume = inpt.volume

    atomquality = inpt.atomicFlags

    args.atradii = datastruct.get_covalent_radii(atomquality)

    cartesian = np.hstack((inpt.coords_c.ravel(), lattmat.ravel()))

    # either read-in existing definition of internal coordinates
    # or generate them afresh
    t0 = time.time()

    try:
        primcoords = read_internals('internals.pkl')
        deal = dealxyz.Dealxyz(cartesian[:-9], primcoords, lattmat)
        for i in range(len(primcoords)):
            primcoords[i].value = deal.internals[i]
        print('Internal coordinates were read-in from the file COORDINATES.gadget')
    except IOError:
        print('Internal coordinates were newly generated')
        intrn = intcoord.Internals(atomquality, args.atradii, args.ascale, args.bscale, args.anglecrit,
                                args.torsioncrit, args.fragcoord, args.relax, args.torsions,
                                args.subst, inpt.numofatoms, inpt.lattmat, inpt.coords_d,
                                inpt.coords_c, inpt.types)
        primcoords = intrn.internalcoords
        write_internals(primcoords, 'internals.pkl')

    print(time.time() - t0, "seconds wall time coord check")

    #c now compute the Bmatrix (wrt. fractional coordinates!)
    b = bmatrix.Bmatrix(cartesian[:-9], primcoords, inpt.numofatoms, lattmat, args.relax)
    Bmat = b.Bmatrix
    #b=bmatrix_cart.bmatrix(cartesian[:-9],primcoords,inpt.numofatoms,lattmat,args.RELAX)
    #Bmat_c=b.Bmatrix

    #c compute the Bmatrix wrt. Cartesian coordinates
    t0 = time.time()

    if args.relax:
        transmat = np.zeros((3*inpt.numofatoms+9, 3*inpt.numofatoms+9), dtype=float)
        #  Parallel(n_jobs=num_cores)(delayed(processInput)(i)for i in range(3*inpt.numofatoms+9))
        #  transmat[i][i]=1.
        for i in range(3*inpt.numofatoms + 9):
            transmat[i][i] = 1.0
        transmat[0:3*inpt.numofatoms, 0:3*inpt.numofatoms] = np.kron(np.eye(inpt.numofatoms), lattinv.T)
    else:
        transmat = np.kron(np.eye(inpt.numofatoms), lattinv.T)

    print(time.time() - t0, "seconds wall time for Bmat computation")


    t0 = time.time()
    #Bmat_c2=matrixmultiply(Bmat,transmat)
    Bmat_c2 = np.dot(Bmat, transmat)
    print(time.time() - t0, "seconds wall time for Bmat multiplication")

    if args.coordinates == 'cartesian':
        Bmat = Bmat_c2

    t0 = time.time()

    f = open('bmat.dat', 'w')
    row = 'Dimensions'
    f.write(row + '\n')
    row = str(len(Bmat)) + '\t' + str(len(Bmat[0]))
    f.write(row + '\n')
    f.write('\n')

    row = "Coordinates (au):"
    f.write(row + '\n')
    for i in range(len(primcoords)):
        row=str(primcoords[i].tag)+'  '+str(primcoords[i].value)
        if primcoords[i].tag=='R' or primcoords[i].tag=="IR1" or primcoords[i].tag=="IR6":
            what=str(primcoords[i].what[0]+1)+' '+str(primcoords[i].what[1]+1)
            where=str(primcoords[i].where[0][0])+' '+str(primcoords[i].where[0][1])+' '+str(primcoords[i].where[0][2])
            where+=' '+str(primcoords[i].where[1][0])+' '+str(primcoords[i].where[1][1])+' '+str(primcoords[i].where[1][2])
        elif primcoords[i].tag=='A':
            what=str(primcoords[i].what[0]+1)+' '+str(primcoords[i].what[1]+1)+' '+str(primcoords[i].what[2]+1)
            where=     str(primcoords[i].where[0][0])+' '+str(primcoords[i].where[0][1])+' '+str(primcoords[i].where[0][2])
            where+=' '+str(primcoords[i].where[1][0])+' '+str(primcoords[i].where[1][1])+' '+str(primcoords[i].where[1][2])
            where+=' '+str(primcoords[i].where[2][0])+' '+str(primcoords[i].where[2][1])+' '+str(primcoords[i].where[2][2])
        elif primcoords[i].tag=='T':
            what=str(primcoords[i].what[0]+1)+' '+str(primcoords[i].what[1]+1)+' '+str(primcoords[i].what[2]+1)+' '+str(primcoords[i].what[3]+1)
            where=     str(primcoords[i].where[0][0])+' '+str(primcoords[i].where[0][1])+' '+str(primcoords[i].where[0][2])
            where+=' '+str(primcoords[i].where[1][0])+' '+str(primcoords[i].where[1][1])+' '+str(primcoords[i].where[1][2])
            where+=' '+str(primcoords[i].where[2][0])+' '+str(primcoords[i].where[2][1])+' '+str(primcoords[i].where[2][2])
            where+=' '+str(primcoords[i].where[3][0])+' '+str(primcoords[i].where[3][1])+' '+str(primcoords[i].where[3][2])
        else:
            print('Error: unsupported coordinate type',primcoords[i].tag)
            sys.exit()
        row+='  '+what+': '+where
        f.write(row+'\n')


    f.write('\n')
    row="Bmatrix(ij):"
    f.write(row+'\n')
    for i in range(len(Bmat)):
        for j in range(len(Bmat[0])):
            if abs(Bmat[i][j]) >= MYTINY:
                row=str(i+1)+'\t'+str(j+1)+'\t'+str(Bmat[i][j])
                f.write(row+'\n')

    f.close()

    print(time.time() - t0, "seconds wall time for writing on bmat.dat")


if __name__ == '__main__':
    main()
