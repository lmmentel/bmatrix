
import warnings


class Complextype:

    def __init__(self, dtyp, coefs, tag, what, whattags, where, value, status):

        self.dtyp = dtyp           # sum/norm
        self.coefs = coefs
        self.tag = tag
        self.what = what
        self.whattags = whattags
        self.where = where
        self.value = value
        self.status = status
        self.labels = None


class Linesearch:

    def __init__(self):

        self.a = [None]         # braketing left
        self.b = [None]         # minimum
        self.c = [None]         # braketing right
        self.direction = None
        self.energies = [None]
        self.steps = [None]
        self.oldgrad = None                 # gradients
        self.newgrad = None
        self.oldpos = None                  # positions
        self.nexpos = None
        self.status = 'u'
        self.cg = None


covalent_radii_default = {
    'Ru': 1.5,   'Re': 1.55,  'Ra': 2.100, 'Rb': 1.670, 'Rn': 0.200, 'Rh': 1.650,\
    'Be': 0.550, 'Ba': 1.540, 'Bi': 1.740, 'Bk': 0.200, 'Br': 1.410, 'H':  0.430, 'P':  1.250, \
    'Os': 1.570, 'Hg': 1.900, 'Ge': 1.370, 'Gd': 1.990, 'Ga': 1.420, 'Pr': 2.020, 'Pt': 1.700, \
    'Pu': 0.200, 'C':  0.900, 'Pb': 1.740, 'Pa': 1.810, 'Pd': 1.700, 'Cd': 1.890, 'Po': 1.880, \
    'Pm': 2.000, 'Ho': 1.940, 'Hf': 1.770, 'K':  1.530, 'He': 0.741, 'Mg': 1.300, 'Mo': 1.670,\
    'Mn': 1.550, 'O':  0.880, 'S':  1.220, 'W':  1.570, 'Zn': 1.650, 'Eu': 2.190, 'Zr': 1.760, \
    'Er': 1.930, 'Ni': 1.700, 'Na': 1.170, 'Nb': 1.680, 'Nd': 2.010, 'Ne': 0.815, 'Np': 1.750, \
    'Fr': 0.200, 'Fe': 1.540, 'B':  1.030, 'F':  0.840, 'Sr': 1.320, 'N':  0.880, 'Kr': 1.069, \
    'Si': 1.000, 'Sn': 1.660, 'Sm': 2.000, 'V':  1.530, 'Sc': 1.640, 'Sb': 1.660, 'Se': 1.420,\
    'Co': 1.530, 'Cm': 0.200, 'Cl': 1.190, 'Ca': 1.190, 'Cf': 1.730, 'Ce': 2.030, 'Xe': 1.750, \
    'Lu': 1.920, 'Cs': 1.870, 'Cr': 1.550, 'Cu': 1.000, 'La': 2.070, 'Li': 0.880, 'Tl': 1.750, \
    'Tm': 1.920, 'Th': 1.990, 'Ti': 1.670, 'Te': 1.670, 'Tb': 1.960, 'Tc': 1.550, 'Ta': 1.630, \
    'Yb': 2.140, 'Dy': 1.950, 'I':  1.600, 'U':  1.780, 'Y':  1.980, 'Ac': 2.080, 'Ag': 1.790, \
    'Ir': 1.520, 'Am': 1.710, 'Al': 1.550, 'As': 1.410, 'Ar': 0.995, 'Au': 1.500, 'At': 0.200, \
    'In': 1.830}

# Covalent radii from Cordero in Angstrom
covalent_radii_cordero = {
    'Ac': 2.15, 'Ag': 1.45, 'Al': 1.21, 'Am': 1.80, 'Ar': 1.06,
    'As': 1.19, 'At': 1.50, 'Au': 1.36, 'B':  0.84, 'Ba': 2.15,
    'Be': 0.96, 'Bi': 1.48, 'Br': 1.20, 'C':  0.73, 'Ca': 1.76,
    'Cd': 1.44, 'Ce': 2.04, 'Cl': 1.02, 'Cm': 1.69, 'Co': 1.38,
    'Cr': 1.39, 'Cs': 2.44, 'Cu': 1.32, 'Dy': 1.92, 'Er': 1.89,
    'Eu': 1.98, 'F':  0.57, 'Fe': 1.42, 'Fr': 2.60, 'Ga': 1.22,
    'Gd': 1.96, 'Ge': 1.20, 'H':  0.31, 'He': 0.28, 'Ff': 1.75,
    'Hg': 1.32, 'Ho': 1.92, 'I':  1.39, 'In': 1.42, 'Ir': 1.41,
    'K':  2.03, 'Kr': 1.16, 'La': 2.07, 'Li': 1.28, 'Lu': 1.87,
    'Mg': 1.41, 'Mn': 1.50, 'Mo': 1.54, 'N':  0.71, 'Na': 1.66,
    'Nb': 1.64, 'Nd': 2.01, 'Ne': 0.58, 'Ni': 1.24, 'Np': 1.90,
    'O':  0.66, 'Os': 1.44, 'P':  1.07, 'Pa': 2.00, 'Pb': 1.46,
    'Pd': 1.39, 'Pm': 1.99, 'Po': 1.40, 'Pr': 2.03, 'Pt': 1.36,
    'Pu': 1.87, 'Ra': 2.21, 'Rb': 2.20, 'Re': 1.51, 'Rh': 1.42,
    'Rn': 1.50, 'Ru': 1.46, 'S':  1.05, 'Sb': 1.39, 'Sc': 1.70,
    'Se': 1.20, 'Si': 1.11, 'Sm': 1.98, 'Sn': 1.39, 'Sr': 1.95,
    'Ta': 1.70, 'Tb': 1.94, 'Tc': 1.47, 'Te': 1.38, 'Th': 2.06,
    'Ti': 1.60, 'Tl': 1.45, 'Tm': 1.90, 'U':  1.96, 'V':  1.53,
    'W':  1.62, 'Xe': 1.40, 'Y':  1.90, 'Yb': 1.87, 'Zn': 1.22,
    'Zr': 1.75
}

covalentradii = {
    'cordero': covalent_radii_cordero,
    'default': covalent_radii_default,
}


def get_covalent_radii(symbols, source='default'):
    '''
    Return a list of covalent radii for chemical symbols defined in the
    ``symbols`` list.

    Args:
        symbols : list
            List of chemical symbols
        source : str
            Name of the source of the covalent radii data
    '''

    if source not in covalentradii.keys():
        source = 'default'
        warnings.warn("<{}> not found using source='default'".format(source))

    return [covalentradii[source][s] for s in symbols]


mendelejev = {
    'ru': 4, 're': 5, 'ra': 6, 'rb': 4, 'rn': 5, 'rh': 4, 'be': 1, 'ba': 5, \
    'bi': 5, 'bk': 6, 'br': 3, 'h': 0, 'p': 2, 'os': 5, 'hg': 5, 'ge': 3, 'gd': 5, 'ga': 3, \
    'pr': 5, 'pt': 5, 'pu': 6, 'c': 1, 'pb': 5, 'pa': 6, 'pd': 4, 'cd': 4, 'po': 5, 'pm': 5, \
    'ho': 5, 'hf': 5, 'k': 3, 'he': 0, 'mg': 2, 'mo': 4, 'mn': 3, 'o': 1, 's': 2, 'w': 5, \
    'zn': 3, 'eu': 5, 'zr': 4, 'er': 5, 'ni': 3, 'na': 2, 'nb': 4, 'nd': 5, 'ne': 1, 'np': 6, \
    'fr': 6, 'fe': 3, 'b': 1, 'f': 1, 'sr': 4, 'n': 1, 'kr': 3, 'si': 2, 'sn': 4, 'sm': 5, \
    'v': 3, 'sc': 3, 'sb': 4, 'se': 3, 'co': 3, 'cm': 6, 'cl': 2, 'ca': 3, 'cf': 6, 'ce': 5, \
    'xe': 4, 'lu': 5, 'cs': 5, 'cr': 3, 'cu': 3, 'la': 5, 'li': 1, 'tl': 5, 'tm': 5, 'th': 6, \
    'ti': 3, 'te': 4, 'tb': 5, 'tc': 4, 'ta': 5, 'yb': 5, 'dy': 5, 'i': 4, 'u': 6, 'y': 4, \
    'ac': 6, 'ag': 4, 'ir': 5, 'am': 6, 'al': 2, 'as': 3, 'ar': 2, 'au': 5, 'at': 5, 'in': 4}
