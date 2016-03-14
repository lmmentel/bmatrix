#!/usr/bin/python
from numpy.oldnumeric import *
from numpy.oldnumeric.linear_algebra import *
from math import *
from string import *
from fpformat import *
import sys
import bmatrix,intcoord
#import bmatrix_cart
import takeinpPOSCAR,various,mymath
#import takeinp
import datastruct
import inputer

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
intrn=intcoord.Intern(atomquality,inps_.ATRADII,inps_.ASCALE,inps_.BSCALE,inps_.ANGLECRIT,inps_.TORSIONCRIT,FRAG,inps_.RELAX,inps_.TORSIONS,inps_.SUBST,inpt.numofatoms,inpt.lattmat,inpt.coords_d,inpt.coords_c,inpt.types)
crt=various.change_format(inpt.coords_c)
cartesian=zeros(len(crt)+9,Float)
cartesian[:-9]=crt
cartesian[-9:-6]=lattmat[0]
cartesian[-6:-3]=lattmat[1]
cartesian[-3:]=lattmat[2]
primcoords=intrn.internalcoords

#c now compute the Bmatrix (wrt. fractional coordinates!)
b=bmatrix.bmatrix(cartesian[:-9],primcoords,inpt.numofatoms,lattmat,inps_.RELAX)
Bmat=b.Bmatrix
#b=bmatrix_cart.bmatrix(cartesian[:-9],primcoords,inpt.numofatoms,lattmat,inps_.RELAX)
#Bmat_c=b.Bmatrix

#c compute the Bmatrix wrt. Cartesian coordinates
if inps_.RELAX==0:
  transmat=mymath.cd_transmatrix(lattinv,3*inpt.numofatoms)
else:
  transmat=zeros((3*inpt.numofatoms+9,3*inpt.numofatoms+9),Float)
  for i in range(3*inpt.numofatoms+9): transmat[i][i]=1.
  transmat[0:3*inpt.numofatoms,0:3*inpt.numofatoms]=mymath.cd_transmatrix(lattinv,3*inpt.numofatoms)
Bmat_c2=matrixmultiply(Bmat,transmat)

if csystem==1:
  Bmat=Bmat_c2

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
