import string

class Complextype:
 def __init__(self,dtyp,coefs,tag,what,whattags,where,value,status):
   self.dtyp=dtyp           # sum/norm
   self.coefs=coefs
   self.tag=tag
   self.what=what
   self.whattags=whattags
   self.where=where
   self.value=value
   self.status=status
   self.labels=None

class Linesearch:
 def __init__(self):
   self.a=[None]         # braketing left
   self.b=[None]         # minimum
   self.c=[None]         # braketing right
   self.direction=None
   self.energies=[None]
   self.steps=[None]
   self.oldgrad=None                 # gradients
   self.newgrad=None
   self.oldpos=None                  # positions
   self.nexpos=None
   self.status='u'
   self.cg=None

# TODO:  what are the units of the radii and what is mendelejev?

class Elementprop:
  def __init__(self):
    self.covalentradii={'ru': 1.5, 're': 1.55, 'ra': 2.100, 'rb': 1.670, 'rn': 0.200, 'rh': 1.650,\
     'be': 0.550, 'ba': 1.540, 'bi': 1.740, 'bk': 0.200, 'br': 1.410, 'h': 0.430, 'p': 1.250, \
     'os': 1.570, 'hg': 1.900, 'ge': 1.370, 'gd': 1.990, 'ga': 1.420, 'pr': 2.020, 'pt': 1.700, \
     'pu': 0.200, 'c': 0.900, 'pb': 1.740, 'pa': 1.810, 'pd': 1.700, 'cd': 1.890, 'po': 1.880, \
     'pm': 2.000, 'ho': 1.940, 'hf': 1.770, 'k': 1.530, 'he': 0.741, 'mg': 1.300, 'mo': 1.670,\
     'mn': 1.550, 'o': 0.880, 's': 1.220, 'w': 1.570, 'zn': 1.650, 'eu': 2.190, 'zr': 1.760, \
     'er': 1.930, 'ni': 1.700, 'na': 1.170, 'nb': 1.680, 'nd': 2.010, 'ne': 0.815, 'np': 1.750, \
     'fr': 0.200, 'fe': 1.540, 'b': 1.030, 'f': 0.840, 'sr': 1.320, 'n': 0.880, 'kr': 1.069, \
     'si': 1.000, 'sn': 1.660, 'sm': 2.000, 'v': 1.530, 'sc': 1.640, 'sb': 1.660, 'se': 1.420,\
     'co': 1.530, 'cm': 0.200, 'cl': 1.190, 'ca': 1.190, 'cf': 1.730, 'ce': 2.030, 'xe': 1.750, \
     'lu': 1.920, 'cs': 1.870, 'cr': 1.550, 'cu': 1.000, 'la': 2.070, 'li': 0.880, 'tl': 1.750, \
     'tm': 1.920, 'th': 1.990, 'ti': 1.670, 'te': 1.670, 'tb': 1.960, 'tc': 1.550, 'ta': 1.630, \
     'yb': 2.140, 'dy': 1.950, 'i': 1.600, 'u': 1.780, 'y': 1.980, 'ac': 2.080, 'ag': 1.790, \
     'ir': 1.520, 'am': 1.710, 'al': 1.550, 'as': 1.410, 'ar': 0.995, 'au': 1.500, 'at': 0.200, \
     'in': 1.830}
    self.mendelejev={'ru': 4, 're': 5, 'ra': 6, 'rb': 4, 'rn': 5, 'rh': 4, 'be': 1, 'ba': 5, \
    'bi': 5, 'bk': 6, 'br': 3, 'h': 0, 'p': 2, 'os': 5, 'hg': 5, 'ge': 3, 'gd': 5, 'ga': 3, \
    'pr': 5, 'pt': 5, 'pu': 6, 'c': 1, 'pb': 5, 'pa': 6, 'pd': 4, 'cd': 4, 'po': 5, 'pm': 5, \
    'ho': 5, 'hf': 5, 'k': 3, 'he': 0, 'mg': 2, 'mo': 4, 'mn': 3, 'o': 1, 's': 2, 'w': 5, \
    'zn': 3, 'eu': 5, 'zr': 4, 'er': 5, 'ni': 3, 'na': 2, 'nb': 4, 'nd': 5, 'ne': 1, 'np': 6, \
    'fr': 6, 'fe': 3, 'b': 1, 'f': 1, 'sr': 4, 'n': 1, 'kr': 3, 'si': 2, 'sn': 4, 'sm': 5, \
    'v': 3, 'sc': 3, 'sb': 4, 'se': 3, 'co': 3, 'cm': 6, 'cl': 2, 'ca': 3, 'cf': 6, 'ce': 5, \
    'xe': 4, 'lu': 5, 'cs': 5, 'cr': 3, 'cu': 3, 'la': 5, 'li': 1, 'tl': 5, 'tm': 5, 'th': 6, \
    'ti': 3, 'te': 4, 'tb': 5, 'tc': 4, 'ta': 5, 'yb': 5, 'dy': 5, 'i': 4, 'u': 6, 'y': 4, \
    'ac': 6, 'ag': 4, 'ir': 5, 'am': 6, 'al': 2, 'as': 3, 'ar': 2, 'au': 5, 'at': 5, 'in': 4}
