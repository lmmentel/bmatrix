
from __future__ import print_function

from math import *
from mymath import *
import string

import numpy as np

from physconstants import physical_constants as pC

class ParseException:
  "This exception is thrown when parsing-error occures."
  def __init__(self,value):
    self.value = value
  def __str__(self):
    return self.value


class TakeInput:
 def __init__(self):
   self.comment="" # comment written at the beginning of POSCAR
   self.numofatoms=0 # total number of atoms in the system
   self.ntypes=0 # number of types
   self.types=[] # number of atoms of each type 
   self.atomicFlags=[] # flags corresponding to different types
   self.atomicMass=[] # mass for each atomic type (amu)
   self.scaling=1.
   self.lattmat = np.zeros((3,3), dtype=float) # lattice vectors (A or au)
   self.lattinv = np.zeros((3,3), dtype=float) # reciprocal lattice vectors
   self.volume=0. # cell volume
   self.coords_d=[] # fractional coordinates
   self.coords_c=[] # cartesial coordinates
   self.xconstrained=[]
   self.energy=0. # total energy
   self.stress = np.zeros(6, dtype=float) # components of the stress tensor
   self.gradients=[] # atomic forces



 def convert_to_au(self):
   if len(self.lattmat)==3: self.lattmat/=pC['AU2A']
   if len(self.lattinv)==3: self.lattinv = np.linalg.inv(self.lattmat)
   self.volume=abs(np.linalg.det(self.lattmat))
   if len(self.coords_c)>0: self.coords_c/=pC['AU2A']
   self.energy/=pC['Hartree2eV']
   if len(self.stress)>0:self.stress/=pC['Hartree2eV']
   if len(self.gradients)>0:self.gradients*=pC['AU2A']/pC['Hartree2eV']
  
 def read(self,ifile): 
   dum=1
   local_dire=None
   coordtype='direct'
   calctype='nonselective'
   calctarget=7
   atomtag=0
   old=open(ifile)
   i=0                                 # row counter
   lattmat_tmp = np.zeros((3,3), dtype=float)
   # reads scaling factor and lattice parameters
   for line in old.readlines():
     i=i+1
     if i==1:
       self.comment=line
     else:
       line=line.split()
     if i==2:
     # factor for scaling of the lattice vectors
       self.scaling=float(line[0])
     if i>2 and i<6:
     # lattice-vectors matrix
       lattmat_tmp[i-3]=[float(line[0]),float(line[1]),float(line[2])]
     if i==6:
       if self.scaling<0:          # negative scaling - required volume
         self.volume=cross_product(lattmat_tmp[0],lattmat_tmp[1])
         self.volume=abs(sum(self.volume*lattmat_tmp[2]))
         self.scaling=(abs(self.scaling)/self.volume)**(1.0/3.0)
       lattmat_tmp=self.scaling*lattmat_tmp
       self.lattmat=lattmat_tmp
       self.lattinv = np.linalg.inv(self.lattmat)
       self.volume=cross_product(lattmat_tmp[0],lattmat_tmp[1])
       self.volume=abs(sum(self.volume*lattmat_tmp[2]))
       self.atomicFlags=line  
     if i==7:  
       katoms=[]
       for j in range(len(line)):
         katoms.append(int(line[j]))
       self.types=katoms
       self.numofatoms=sum(katoms)
       lattmat_tmp = np.zeros((3,3), dtype=float)
       coords = np.zeros((self.numofatoms,3), dtype=float)
     if i==8:
       if (line[0][0])=="C" or (line[0][0])=="c":
         coordtype='cart'

     if i>8:

       indx=(i)%(self.numofatoms+8)
       if (indx==0): indx=self.numofatoms+8
       #if indx<=2 or indx==6 or indx==7 or indx==8: 
       #  break
       if (indx==1):
         break
       elif indx>8:
         coords[indx-9][0]=float(line[0])
         coords[indx-9][1]=float(line[1])
         coords[indx-9][2]=float(line[2])
         if indx==self.numofatoms+8:
           if coordtype=='direct':
             self.coords_d=coords
             self.coords_c=np.dot(self.coords_d,self.lattmat)
           else:
             self.coords_c=coords
             self.coords_d = np.dot(self.coords_c,self.lattinv)
   old.close()

#check internal consistency: TODO

#out=TakeInput()
#out.read('XDATCAR')
#print "out.numofatoms",out.numofatoms
#print "out.numconfig",out.numconfig
#for i in range(len(out.lattmat)):
#  print i,out.lattmat[i][-1]
#for i in range(len(out.lattmat)):
#  print i,out.coords_d[i][-1]
#print "out.ntypes",out.ntypes
#print "out.types",out.types
#print "out.atomicFlags",out.atomicFlags
#print "out.atomicMass",out.atomicMass
#print "out.lattmat",out.lattmat
#print "out.volume",out.volume
#print "out.coords_d",out.coords_d
#print "out.coords_c",out.coords_c
#print "out.energy",out.energy
#print "out.stress",out.stress
#print "out.gradients",out.gradients
#out.convert_to_au()
#print "converted to a.u."
#print "out.numofatoms",out.numofatoms
#print "out.ntypes",out.ntypes
#print "out.types",out.types
#print "out.atomicFlags",out.atomicFlags
#print "out.lattmat",out.lattmat
#print "out.volume",out.volume
#print "out.coords_d",out.coords_d
#print "out.coords_c",out.coords_c
#print "out.energy",out.energy
#print "out.stress",out.stress
#print "out.gradients",out.gradients




