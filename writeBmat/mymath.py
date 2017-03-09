from math import *

import numpy as np


def vector_size(vector):
    """calculates the size of the vector,
    takes array as the argument!!!
    """

    d = (sum((vector) * (vector)))**0.5
    return d


def cross_product(a, b):
    """returns the [x,y,z] components of
    the cross product
    """

    x = a[1] * b[2] - b[1] * a[2]
    y = a[2] * b[0] - b[2] * a[0]
    z = a[0] * b[1] - b[0] * a[1]
    return np.array([x, y, z])


def mygeneralized_inverse(matrix):

    try:
        val, vect = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        invmatrix = np.linalg.inv(matrix)
    else:
        dim = len(matrix)
        invmatrix = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            if abs(val[i]) > 1e-7:
                invmatrix[i][i] = 1 / val[i]
        invmatrix = np.dot(np.transpose(vect), invmatrix)
        invmatrix = np.dot(invmatrix, vect)
    return invmatrix


def normalize_matrow(matrix):
    'normalizes rows in matrix'

    for i in range(len(matrix)):
        norm = vector_size(matrix[i])
        if norm > 1e-05:
            matrix[i] = matrix[i] / norm
        else:
            matrix[i] *= 0.0
    return matrix


def build_umat(rank):
    'builds up a unit matrix of a given rank'

    matrix = np.zeros((rank, rank), dtype=float)
    for i in range(rank):
        matrix[i][i] = 1.0
    return matrix


def orthonormalize_mat(matrix):
    'orthogonalizes the rows of the matrix'

    matrix = normalize_matrow(matrix)
    sizes = []
    for i in range(len(matrix)):
        s = vector_size(matrix[i])
        if s > 1e-05:
            sizes.append(i)
    if len(sizes) > 1:
        for j in range(1, len(sizes)):
            i = sizes[j]
            for k in range(0, j):
                l = sizes[k]
                matrix[i] = matrix[i] - np.inner(matrix[i], matrix[l]) * matrix[l]
                norm = vector_size(matrix[i])
                if norm > 1e-05:
                    matrix[i] = matrix[i] / norm
                else:
                    matrix[i] = matrix[i] * 0.0
    return matrix


def remove_zrows(matrix):
    'removes rows with zeros'

    newmatrix = []
    for i in range(len(matrix)):
        if vector_size(matrix[i]) > 1e-06:
            newmatrix.append(matrix[i])
    newmatrix = np.array(newmatrix)
    return newmatrix

 
def de_cycle(prims1,prims2,coords):
  for m in range(len(coords)):
    if coords[m].dtyp=='simple':
      if coords[m].tag=='A' or coords[m].tag=='T':
        while (prims2[m]-prims1[m])>pi:
          prims2[m]=prims2[m]-2*pi
        while (prims2[m]-prims1[m])<-pi:
          prims2[m]=prims2[m]+2*pi
      if coords[m].tag=='A':prims2[m]=abs(prims2[m])
  return prims2


def line_def(point1,point2):
  """from two point calculates parameters
  for line in 3D. (y=k1+c1, z=k2+c2)
  """
  vect=point1-point2
  k1=vect[1]/vect[0]
  k2=vect[2]/vect[0]
  c1=point1[1]-k1*point1[0]
  c2=point1[2]-k2*point1[0]
  return k1,k2,c1,c2


def point_paraline(point,k1,k2,c1,c2):
  """given the line A defined by k1,k2,c1 and c2
  (y=k1+c1, z=k2+c2), and line B through point ortoghonal
  to A. cross point is calculated
  """
  x=(k1+k2-point[0]-k1*point[1]-k2*point[2])/(1+k1**2+k2**2)
  y=k1*x+c1
  z=k2*x+c2
  crosspoint=[x,y,z]
  return crosspoint


def dir_cart(lvect,dirs):
  """transforms fractional coordinates
  to cartesians
  """
  carts=np.dot(dirs,lvect)
  return carts


def cart_dir(lvect,carts):
  """transforms cartesian coordinates
  to fractional
  """
  m = np.linalg.inv(lvect)
  direct=np.dot(carts,m)
  for i in range(len(direct)):
    for j in range(3):
      while direct[i][j]>1:
        direct[i][j]=direct[i][j]-1
      while direct[i][j]<0:
        direct[i][j]=direct[i][j]+1
  idirect=[]
  for i in range(len(direct)):
    idirect.append(direct[i])
  return idirect

def cd_transmatrix(lvect,dim):
  """B-matrix for cartesian
  coordinates
  """
  transmat=np.zeros((dim,dim),dtype=float)
  for i in range(dim/3):
    transmat[3*i:3*i+3,3*i:3*i+3]=np.transpose(lvect)
  return transmat

def sym_threemat(mat):
  """symmetrization of
  the 3x3 matrix
  """
  err=1.0
  while err>1e-5:
    alpha=atan((mat[1,0]-mat[0,1])/(mat[0,0]+mat[1,1]))
    mat_a=np.zeros((3,3),dtype=float)
    mat_a[0,0]=cos(alpha)
    mat_a[1,1]=cos(alpha)
    mat_a[2,2]=1.0
    mat_a[0,1]=sin(alpha)
    mat_a[1,0]=-sin(alpha)
    mat=np.dot(mat,mat_a)

    betha=atan((mat[2,0]-mat[0,2])/(mat[0,0]+mat[2,2]))
    mat_b=np.zeros((3,3),dtype=float)
    mat_b[0,0]=cos(betha)
    mat_b[2,2]=cos(betha)
    mat_b[1,1]=1.0
    mat_b[0,2]=sin(betha)
    mat_b[2,0]=-sin(betha)
    mat=np.dot(mat,mat_b)

    gamma=atan((mat[2,1]-mat[1,2])/(mat[1,1]+mat[2,2]))
    mat_c=np.zeros((3,3),dtype=float)
    mat_c[1,1]=cos(gamma)
    mat_c[2,2]=cos(gamma)
    mat_c[0,0]=1.0
    mat_c[1,2]=sin(gamma)
    mat_c[2,1]=-sin(gamma)
    mat=np.dot(mat,mat_c)

    err1=abs(mat[0,1]-mat[1,0])
    err2=abs(mat[2,0]-mat[0,2])
    err3=abs(mat[2,1]-mat[1,2])
    err=max(err1,err2,err3)

  return mat


def regr_two(x, y):

    Xmat = np.zeros((len(x), 3), dtype=float)
    for i in range(len(Xmat)):
        Xmat[i] = [1, x[i], x[i]**2]
    Xtrans = np.transpose(Xmat)
    Bmat = np.dot(Xtrans, Xmat)
    Binv = np.linalg.inv(Bmat)
    cvect = np.dot(Binv, Xtrans)
    cvect = np.dot(cvect, np.transpose(y))
    return cvect
