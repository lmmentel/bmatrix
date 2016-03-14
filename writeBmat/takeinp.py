#!/usr/local/bin/python

from numpy.oldnumeric import *
from numpy.oldnumeric.linear_algebra import *
from math import *
import re
import physconstants

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
   self.lattmat=zeros((3,3),Float) # lattice vectors (A or au)
   self.lattinv=zeros((3,3),Float) # reciprocal lattice vectors
   self.volume=0. # cell volume
   self.coords_d=[] # fractional coordinates
   self.coords_c=[] # cartesial coordinates
   self.xconstrained=[]
   self.energy=0. # total energy
   self.stress=zeros(6,Float) # components of the stress tensor
   self.gradients=[] # atomic forces
   #TODO: read Selective Dynamics!!!
   #TODO: check if NSW==1!!!
   #TODO: check if IBRION has a reasonable value!!!
 
 def read(self,f):
   "Read data from the OUTCAR file."
   f = open(f,"r") 
   task=0 #TODO now compute the cell volume, coords_c,.
   for line in f.readlines():
     if task==0:
       dummy=re.search("VRHFIN =",line)
       if dummy:
	 k=dummy.end()
	 line=line[k:]
	 dummy=re.search(":",line)
	 if dummy:
	   k=dummy.start()
	   line=line[:k].strip()
	   self.atomicFlags.append(line[:k])
	 else:
	   print "problem reading atomic flags!!!" 
	 continue
       
       dummy=re.search("POMASS =",line)
       if dummy:
	 k=dummy.end()
	 line=line[k:]
	 dummy=re.search(";",line)
	 if dummy:
	   k=dummy.start()
	   line=line[:k].strip()
	   self.atomicMass.append(float(line[:k]))
	 else:
	   print "problem reading atomic mass!!!" 
	 continue
       
       dummy=re.search("POSCAR:",line)
       if dummy:
	 k=dummy.end()
	 self.comment=line[k+1:-1]	 
	 continue
       
       dummy=re.search("ions per type =",line)
       if dummy:
         k=dummy.end()
         types_=line[k:].split()
         for i in range(len(types_)):
	   self.types.append(int(types_[i]))
	 self.numofatoms=sum(self.types)
	 self.ntypes=len(self.types)
	 self.coords_d=zeros((self.numofatoms,3),Float)
	 self.gradients=zeros((self.numofatoms,3),Float)
	 task=1   
	 continue 
     if task==1:
       dummy=re.search("direct lattice vectors",line)
       if dummy:
         task=2
       continue
     if task>1 and task<5:
       #TODO: check if the format of each line is consistent with the following code!!!
       line=line.split()
       if (len(line)==6):
         self.lattmat[task-2][0]=float(line[0])
         self.lattmat[task-2][1]=float(line[1])
         self.lattmat[task-2][2]=float(line[2])
       else:
	 raise ParseException('problem when reading lattice vectors')
       task+=1
       continue
     if task==5:
       dummy=re.search("position of ions in fractional coordinates",line)
       if dummy:
	 task=6
       continue
     if task>5 and task<=(self.numofatoms+5):
       #TODO: check if the format of each line is consistent with the following code!!!
       line=line.split()
       if len(line)==3:
	 self.coords_d[task-6][0]=float(line[0])
         self.coords_d[task-6][1]=float(line[1])
         self.coords_d[task-6][2]=float(line[2]) 
       else:
	 raise ParseException('problem reading atomic positions')
       task+=1
       continue
     if task==self.numofatoms+6:
       dummy=re.search("FORCE on cell =-STRESS",line)
       #c maybe stress is not computed - read in forces in that case
       dummy2=re.search("TOTAL-FORCE",line)
       if dummy:
         task+=1
       if dummy2:
         task=self.numofatoms+9
       continue
     if task==self.numofatoms+7:
       dummy=re.search("Total",line)
       if dummy:
         k=dummy.end()
         line=line[k:].split()
         if len(line)==6:
	   for i in range(6):
	     self.stress[i]=float(line[i])
	 else:
	   raise ParseException('problem reading stress tensor')
	 task+=1   
	 continue
     if task==self.numofatoms+8: 
       dummy=re.search("TOTAL-FORCE",line)
       if dummy:
	 task+=1
       continue
     if task>self.numofatoms+8 and task<self.numofatoms+10:
       task+=1
       continue
     if task>=self.numofatoms+10 and task<2*self.numofatoms+10:
       line=line.split()
       if len(line)==6:
	 self.gradients[task-(self.numofatoms+10)][0]=float(line[3])
	 self.gradients[task-(self.numofatoms+10)][1]=float(line[4])
	 self.gradients[task-(self.numofatoms+10)][2]=float(line[5])
       else:
         raise ParseException('problem reading atomic forces')
       task+=1
       continue
     if task==2*self.numofatoms+10:
       dummy=re.search("free  energy   TOTEN  =",line)
       if dummy:
         k=dummy.end()
         ene=line[k:].split()
         self.energy=float(ene[0])
	 task+=1   
	 continue 
   f.close()
   self.lattmat=array(self.lattmat)
   self.lattinv=inverse(self.lattmat)
   self.coords_d=array(self.coords_d)
   self.coords_c=dot(self.coords_d,self.lattmat)
   self.volume=abs(determinant(self.lattmat))
   
 def convert_to_au(self):
   pC=physconstants.physConstants()
   self.lattmat/=pC.AU2A
   self.lattinv=inverse(self.lattmat)
   self.volume=abs(determinant(self.lattmat))
   self.coords_c/=pC.AU2A
   self.energy/=pC.Hartree2eV
   self.stress/=pC.Hartree2eV
   self.gradients*=pC.AU2A/pC.Hartree2eV
 
 
#out=TakeInput()
#out.read("OUTCAR")
#print "out.numofatoms",out.numofatoms
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
