#!/usr/bin/python
from numpy.oldnumeric import *
from numpy.oldnumeric.linear_algebra import *
from math import *
from string import *
from fpformat import *
import sys
import bmatrix,intcoord
import takeinpPOSCAR,various,mymath,dealxyz
import datastruct
import inputer
import pickle
import time

#from joblib import Parallel, delayed
#import multiprocessing

#num_cores = multiprocessing.cpu_count()
#print("numCores = " + str(num_cores))

def read_matrix(cfile):
  ufile=open(cfile,'r')
  matrix=pickle.load(ufile)
  ufile.close()
  return matrix

def write_matrix(matrix,cfile):
  ufile=open(cfile,'w')
  pickle.dump(matrix,ufile)
  ufile.close()

#c the smallest acceptable non-zero matrix element
mytiny=1e-6

#c B matrix wrt.fractional(0) or cartesian(1) coordinates
csystem=1 


#c read-in input file (POSCAR format)
fname=sys.argv[1]

#c optional
if len(sys.argv)>2:
  csystem=int(sys.argv[2])

inps_=inputer.inputer()


inpt=takeinpPOSCAR.TakeInput()
inpt.read(fname)
#inpt=takeinp.TakeInput()
#inpt.read("OUTCAR")
inpt.convert_to_au()
lattmat=inpt.lattmat
lattinv=inpt.lattinv
volume=inpt.volume
atomquality=inpt.atomicFlags
if len(inps_.ATRADII)!=len(atomquality): 
  inps_.ATRADII=[]
  COVALENTRADII=datastruct.Elementprop().covalentradii
  for i in range(len(atomquality)):
    index=atomquality[i]
    inps_.ATRADII.append(COVALENTRADII[index])
atomictags=[]
for i in range(inpt.ntypes):
  for j in range(inpt.types[i]):
    atomictags.append(atomquality[i])

FRAG=inps_.FRAGCOORD

crt=various.change_format(inpt.coords_c)
cartesian=zeros(len(crt)+9,Float)
cartesian[:-9]=crt
cartesian[-9:-6]=lattmat[0]
cartesian[-6:-3]=lattmat[1]
cartesian[-3:]=lattmat[2]

#c either read-in existing definition of internal coordinates
#c or generate them afresh
t0 = time.time()

try:
  primcoords=read_matrix('COORDINATES.gadget')
  deal=dealxyz.Dealxyz(cartesian[:-9],primcoords,lattmat)
  for i in range(len(primcoords)):
    primcoords[i].value=deal.internals[i]
  print 'Internal coordinates were read-in from the file COORDINATES.gadget'
except IOError:
  print 'Internal coordinates were newly generated'
  intrn=intcoord.Intern(atomquality,inps_.ATRADII,inps_.ASCALE,inps_.BSCALE,inps_.ANGLECRIT,inps_.TORSIONCRIT,FRAG,inps_.RELAX,inps_.TORSIONS,inps_.SUBST,inpt.numofatoms,inpt.lattmat,inpt.coords_d,inpt.coords_c,inpt.types)
  primcoords=intrn.internalcoords
  write_matrix(primcoords,'COORDINATES.gadget')

print time.time() - t0, "seconds wall time coord check"

#c now compute the Bmatrix (wrt. fractional coordinates!)
b=bmatrix.bmatrix(cartesian[:-9],primcoords,inpt.numofatoms,lattmat,inps_.RELAX)
Bmat=b.Bmatrix
#b=bmatrix_cart.bmatrix(cartesian[:-9],primcoords,inpt.numofatoms,lattmat,inps_.RELAX)
#Bmat_c=b.Bmatrix

#c compute the Bmatrix wrt. Cartesian coordinates
t0 = time.time()

if inps_.RELAX==0:
  transmat=mymath.cd_transmatrix(lattinv,3*inpt.numofatoms)
else:
  transmat=zeros((3*inpt.numofatoms+9,3*inpt.numofatoms+9),Float)
#  Parallel(n_jobs=num_cores)(delayed(processInput)(i)for i in range(3*inpt.numofatoms+9))
#  transmat[i][i]=1.
  for i in range(3*inpt.numofatoms+9): transmat[i][i]=1.
  transmat[0:3*inpt.numofatoms,0:3*inpt.numofatoms]=mymath.cd_transmatrix(lattinv,3*inpt.numofatoms)
print time.time() - t0, "seconds wall time for Bmat computation"


t0 = time.time()
#Bmat_c2=matrixmultiply(Bmat,transmat)
Bmat_c2 = dot(Bmat,transmat)
print time.time() - t0, "seconds wall time for Bmat multiplication"

if csystem==1:
  Bmat=Bmat_c2

t0 = time.time()

f=open('bmat.dat','w')
row='Dimensions'
f.write(row+'\n')
row=str(len(Bmat))+'\t'+str(len(Bmat[0]))
f.write(row+'\n')
f.write('\n')


row="Coordinates (au):"
f.write(row+'\n')
for i in range(len(primcoords)):
#  row=str(primcoords[i].tag)+': '+str(primcoords[i].value)
  row=str(primcoords[i].tag)+'  '+str(primcoords[i].value) 
  if primcoords[i].tag=='R' or primcoords[i].tag=="IR1" or primcoords[i].tag=="IR6":
    what=str(primcoords[i].what[0]+1)+' '+str(primcoords[i].what[1]+1)
    where=     str(primcoords[i].where[0][0])+' '+str(primcoords[i].where[0][1])+' '+str(primcoords[i].where[0][2])
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
    print 'Error: unsupported coordinate type',primcoords[i].tag
    sys.exit()
#  row+=': '+what+': '+where
  row+='  '+what+': '+where 
  f.write(row+'\n')


f.write('\n')
row="Bmatrix(ij):"
f.write(row+'\n')
for i in range(len(Bmat)):
  for j in range(len(Bmat[0])):
    if abs(Bmat[i][j]) >=mytiny:
      row=str(i+1)+'\t'+str(j+1)+'\t'+str(Bmat[i][j])
      f.write(row+'\n')

f.close()

print time.time() - t0, "seconds wall time for writing on bmat.dat"


