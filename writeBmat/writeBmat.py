#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import time

import numpy as np

from math import *
from string import *

import bmatrix
import intcoord
import takeinpPOSCAR, various, mymath, dealxyz
import datastruct
import inputparser

from pprint import pprint

#from joblib import Parallel, delayed
#import multiprocessing

#num_cores = multiprocessing.cpu_count()
#print("numCores = " + str(num_cores))

def read_matrix(fname):
    'Read an array from file'
    matrix = np.load(fname)
    return matrix

def write_matrix(matrix, fname):
    'Write a matrix to file'
    np.save(fname, matrix)

#c the smallest acceptable non-zero matrix element
MYTINY=1e-6

def main():
    'Main program'

    args = inputparser.get_input_args()

    inpt = takeinpPOSCAR.TakeInput()
    inpt.read(args.filename)
    #inpt=takeinp.TakeInput()
    #inpt.read("OUTCAR")
    inpt.convert_to_au()
    lattmat = inpt.lattmat
    lattinv = inpt.lattinv
    volume = inpt.volume
    atomquality = inpt.atomicFlags

    if len(args.atradii) != len(atomquality):
        args.atradii = []
        COVALENTRADII = datastruct.Elementprop().covalentradii
    for i in range(len(atomquality)):
        index=atomquality[i]
        args.atradii.append(COVALENTRADII[index])
    atomictags=[]
    for i in range(inpt.ntypes):
        for j in range(inpt.types[i]):
            atomictags.append(atomquality[i])

    crt = various.change_format(inpt.coords_c)
    cartesian = np.zeros(len(crt) + 9, dtype=float)
    cartesian[:-9] = crt
    cartesian[-9:-6] = lattmat[0]
    cartesian[-6:-3] = lattmat[1]
    cartesian[-3:] = lattmat[2]

    #c either read-in existing definition of internal coordinates
    #c or generate them afresh
    t0 = time.time()

    try:
        primcoords=read_matrix('COORDINATES.gadget')
        deal=dealxyz.Dealxyz(cartesian[:-9], primcoords, lattmat)
        for i in range(len(primcoords)):
            primcoords[i].value=deal.internals[i]
        print('Internal coordinates were read-in from the file COORDINATES.gadget')
    except IOError:
        print('Internal coordinates were newly generated')
        intrn = intcoord.Intern(atomquality, args.atradii, args.ascale, args.bscale, args.anglecrit,
                                args.torsioncrit, args.fragcoord, args.relax, args.torsions,
                                args.subst, inpt.numofatoms, inpt.lattmat, inpt.coords_d,
                                inpt.coords_c, inpt.types)
        primcoords = intrn.internalcoords
        write_matrix(primcoords, 'COORDINATES.gadget')

    print(time.time() - t0, "seconds wall time coord check")

    #c now compute the Bmatrix (wrt. fractional coordinates!)
    b = bmatrix.bmatrix(cartesian[:-9], primcoords, inpt.numofatoms, lattmat, args.relax)
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
        transmat[0:3*inpt.numofatoms, 0:3*inpt.numofatoms] = mymath.cd_transmatrix(lattinv, 3*inpt.numofatoms)
    else:
        transmat = mymath.cd_transmatrix(lattinv, 3*inpt.numofatoms)

    print(time.time() - t0, "seconds wall time for Bmat computation")


    t0 = time.time()
    #Bmat_c2=matrixmultiply(Bmat,transmat)
    Bmat_c2 = np.dot(Bmat, transmat)
    print(time.time() - t0, "seconds wall time for Bmat multiplication")

    if args.coordinates == 'cartesian':
        Bmat = Bmat_c2

    t0 = time.time()

    f = open('bmat.dat','w')
    row = 'Dimensions'
    f.write(row+'\n')
    row=str(len(Bmat))+'\t'+str(len(Bmat[0]))
    f.write(row+'\n')
    f.write('\n')


    row = "Coordinates (au):"
    f.write(row+'\n')
    for i in range(len(primcoords)):
    #  row=str(primcoords[i].tag)+': '+str(primcoords[i].value)
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
            print('Error: unsupported coordinate type', primcoords[i].tag)
            sys.exit()
    #  row+=': '+what+': '+where
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


