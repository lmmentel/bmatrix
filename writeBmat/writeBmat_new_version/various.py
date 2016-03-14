from numpy.oldnumeric import *

def change_format(xyz):
# transform the three-column format into
# the one column one
  newxyz=[]
  for i in range(len(xyz)):
    for j in range(3):
      newxyz.append(xyz[i][j])
  return array(newxyz)

def format_change(xyz):
# transform the one column format into
# three column one
  newxyz=[]
  for i in range(len(xyz)/3):
    newxyz.append([xyz[3*i],xyz[3*i+1],xyz[3*i+2]])
  return array(newxyz)

def step_lim(prims1,prims2,coords,STEPLIM):
  step=abs(prims2-prims1)
  criter=1
  for i in range(len(coords)):
   # print step[i],STEPLIM
    if coords[i].tag=='R':
      if step[i]>STEPLIM:
       # print 'R:',step[i]
        criter=0
    elif coords[i].tag=='A' or coords[i].tag=='T':
      if step[i]>STEPLIM:
       # print 'A/T:',step[i]
        criter=0
    elif coords[i].tag=='IR1':
      if step[i]>STEPLIM/5*prims1[i]**2:
      #if step[i]>2*STEPLIM/5*prims1[i]**2:
        #print 'IR:',step[i],STEPLIM/5*prims1[i]**2
        criter=0
    elif coords[i].tag=='IR6':
      if step[i]>STEPLIM:    #2*STEPLIM*6/(2000/prims1[i])**5:
        #print 'IR6:',step[i]
        criter=0
    else: 
      if step[i]>STEPLIM:
        criter=0
  return criter

def give_dist(A,B):
  dist=(sum((A-B)**2))**0.5
  return dist

def shortest_dist(cartesians,lattmat,atom1,atom2):
  """finds the shortest distance between two atoms
  """
  cartesians=format_change(cartesians)
  cart1=cartesians[atom1]
  cart2=cartesians[atom2]
  dists=[]
  what=[]

  for i in [-1,0,1]:
    for j in [-1,0,1]:
      for k in [-1,0,1]:
        trans=i*lattmat[0]+j*lattmat[1]+k*lattmat[2]
        point2=cart2+trans
        dist=give_dist(cart1,point2)
        dists.append(dist)
        what.append([i,j,k])

  dists=array(dists)
  dummy=argmin(dists)
  return [[0,0,0],[what[dummy][0],what[dummy][1],what[dummy][2]]]
