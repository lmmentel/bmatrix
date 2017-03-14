
import warnings


class Complextype:
    '''
    A container for the internal coordinates

    Args:

        dtyp (str) :
            Type

        coeffs (list of int) :
            Coefficients

        tag (str) :
            Coordinate tag, usually `R` - distance, `A` - angle,
            `T` - torsion

        what (list of int) :
            Indices of atoms that belong to the coordinate

        whattags (list of str) :
            Symbols of the atoms that belong to the coordinate

        where (list of arrays) :
            List of unit cell indices singifying the origin of each
            atom that belongs to the coordinate

        value (float) :
            value of the coordinate

    '''

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

    def __repr__(self):

        return '<Complextype(' + ', '.join([
            'dtyp="{}"'.format(self.dtyp),
            'tag="{}"'.format(self.tag),
            'value={}'.format(self.value),
            'what={}'.format(', '.join([str(i) for i in self.what])),
            'whattags={}'.format(', '.join([str(i) for i in self.whattags])),
            'where={}'.format(', '.join([str(i) for i in self.where]))]) + ')>'


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
    'Ac': 3.9306303409522827,
    'Ag': 3.3826097645695126,
    'Al': 2.9290754944596338,
    'Am': 3.231431674532886,
    'Ar': 1.880277494830539,
    'As': 2.6645138368955377,
    'At': 0.37794522509156564,
    'Au': 2.834589188186742,
    'B': 1.946417909221563,
    'Ba': 2.9101782332050554,
    'Be': 1.0393493690018056,
    'Bi': 3.288123458296621,
    'Bk': 0.37794522509156564,
    'Br': 2.6645138368955377,
    'C': 1.7007535129120455,
    'Ca': 2.2487740892948156,
    'Cd': 3.571582377115295,
    'Ce': 3.836144034679391,
    'Cf': 3.269226197042043,
    'Cl': 2.2487740892948156,
    'Cm': 0.37794522509156564,
    'Co': 2.891280971950477,
    'Cr': 2.9290754944596338,
    'Cs': 3.533787854606139,
    'Cu': 1.8897261254578281,
    'Dy': 3.684965944642765,
    'Er': 3.6471714221336082,
    'Eu': 4.138500214752644,
    'F': 1.5873699453845755,
    'Fe': 2.9101782332050554,
    'Fr': 0.37794522509156564,
    'Ga': 2.683411098150116,
    'Gd': 3.760554989661078,
    'Ge': 2.5889247918772247,
    'H': 0.8125822339468661,
    'He': 1.4002870589642507,
    'Hf': 3.344815242060356,
    'Hg': 3.590479638369873,
    'Ho': 3.6660686833881866,
    'I': 3.023561800732525,
    'In': 3.4581988095878256,
    'Ir': 2.8723837106958987,
    'K': 2.891280971950477,
    'Kr': 2.0201172281144184,
    'La': 3.911733079697704,
    'Li': 1.6629589904028887,
    'Lu': 3.62827416087903,
    'Mg': 2.4566439630951766,
    'Mn': 2.9290754944596338,
    'Mo': 3.1558426295145727,
    'N': 1.6629589904028887,
    'Na': 2.210979566785659,
    'Nb': 3.174739890769151,
    'Nd': 3.798349512170234,
    'Ne': 1.5401267922481299,
    'Ni': 3.212534413278308,
    'Np': 3.307020719551199,
    'O': 1.6629589904028887,
    'Os': 2.9668700169687905,
    'P': 2.3621576568222853,
    'Pa': 3.420404287078669,
    'Pb': 3.288123458296621,
    'Pd': 3.212534413278308,
    'Pm': 3.7794522509156563,
    'Po': 3.552685115860717,
    'Pr': 3.817246773424813,
    'Pt': 3.212534413278308,
    'Pu': 0.37794522509156564,
    'Ra': 3.9684248634614394,
    'Rb': 3.1558426295145727,
    'Re': 2.9290754944596338,
    'Rh': 3.1180481070054165,
    'Rn': 0.37794522509156564,
    'Ru': 2.834589188186742,
    'S': 2.30546587305855,
    'Sb': 3.1369453682599944,
    'Sc': 3.099150845750838,
    'Se': 2.683411098150116,
    'Si': 1.8897261254578281,
    'Sm': 3.7794522509156563,
    'Sn': 3.1369453682599944,
    'Sr': 2.4944384856043333,
    'Ta': 3.0802535844962597,
    'Tb': 3.7038632058973433,
    'Tc': 2.9290754944596338,
    'Te': 3.1558426295145727,
    'Th': 3.760554989661078,
    'Ti': 3.1558426295145727,
    'Tl': 3.307020719551199,
    'Tm': 3.62827416087903,
    'U': 3.363712503314934,
    'V': 2.891280971950477,
    'W': 2.9668700169687905,
    'Xe': 3.307020719551199,
    'Y': 3.7416577284064996,
    'Yb': 4.044013908479752,
    'Zn': 3.1180481070054165,
    'Zr': 3.3259179808057775,
}

# Covalent radii from Cordero in Angstrom
covalent_radii_cordero = {
    'Ac': 4.06291116973433,
    'Ag': 2.7401028819138507,
    'Al': 2.286568611803972,
    'Am': 3.401507025824091,
    'Ar': 2.003109692985298,
    'As': 2.2487740892948156,
    'At': 2.834589188186742,
    'Au': 2.5700275306226463,
    'B': 1.5873699453845755,
    'Ba': 4.06291116973433,
    'Be': 1.814137080439515,
    'Bi': 2.7967946656775857,
    'Br': 2.2676713505493935,
    'C': 1.3795000715842145,
    'Ca': 3.3259179808057775,
    'Cd': 2.7212056206592723,
    'Ce': 3.8550412959339693,
    'Cl': 1.9275206479669846,
    'Cm': 3.1936371520237294,
    'Co': 2.6078220531318026,
    'Cr': 2.626719314386381,
    'Cs': 4.6109317461171,
    'Cu': 2.4944384856043333,
    'Dy': 3.62827416087903,
    'Er': 3.571582377115295,
    'Eu': 3.7416577284064996,
    'F': 1.077143891510962,
    'Fe': 2.683411098150116,
    'Ff': 3.307020719551199,
    'Fr': 4.913287926190353,
    'Ga': 2.30546587305855,
    'Gd': 3.7038632058973433,
    'Ge': 2.2676713505493935,
    'H': 0.5858150988919267,
    'He': 0.5291233151281919,
    'Hg': 2.4944384856043333,
    'Ho': 3.62827416087903,
    'I': 2.626719314386381,
    'In': 2.683411098150116,
    'Ir': 2.6645138368955377,
    'K': 3.836144034679391,
    'Kr': 2.1920823055310805,
    'La': 3.911733079697704,
    'Li': 2.41884944058602,
    'Lu': 3.533787854606139,
    'Mg': 2.6645138368955377,
    'Mn': 2.834589188186742,
    'Mo': 2.9101782332050554,
    'N': 1.341705549075058,
    'Na': 3.1369453682599944,
    'Nb': 3.099150845750838,
    'Nd': 3.798349512170234,
    'Ne': 1.0960411527655403,
    'Ni': 2.343260395567707,
    'Np': 3.590479638369873,
    'O': 1.2472192428021667,
    'Os': 2.7212056206592723,
    'P': 2.022006954239876,
    'Pa': 3.7794522509156563,
    'Pb': 2.759000143168429,
    'Pd': 2.626719314386381,
    'Pm': 3.760554989661078,
    'Po': 2.6456165756409593,
    'Pr': 3.836144034679391,
    'Pt': 2.5700275306226463,
    'Pu': 3.533787854606139,
    'Ra': 4.1762947372618004,
    'Rb': 4.1573974760072225,
    'Re': 2.8534864494413203,
    'Rh': 2.683411098150116,
    'Rn': 2.834589188186742,
    'Ru': 2.759000143168429,
    'S': 1.9842124317307197,
    'Sb': 2.626719314386381,
    'Sc': 3.212534413278308,
    'Se': 2.2676713505493935,
    'Si': 2.0975959992581896,
    'Sm': 3.7416577284064996,
    'Sn': 2.626719314386381,
    'Sr': 3.684965944642765,
    'Ta': 3.212534413278308,
    'Tb': 3.6660686833881866,
    'Tc': 2.7778974044230074,
    'Te': 2.6078220531318026,
    'Th': 3.892835818443126,
    'Ti': 3.023561800732525,
    'Tl': 2.7401028819138507,
    'Tm': 3.590479638369873,
    'U': 3.7038632058973433,
    'V': 2.891280971950477,
    'W': 3.061356323241682,
    'Xe': 2.6456165756409593,
    'Y': 3.590479638369873,
    'Yb': 3.533787854606139,
    'Zn': 2.30546587305855,
    'Zr': 3.307020719551199,
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
