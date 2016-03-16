
from scipy.constants import value, angstrom

class physConstants:
    def __init__(self):
        self.Hartree2eV = value('Hartree energy in eV')
        self.AU2A = value('atomic unit of length')/angstrom
        self.ev2j = value('electron volt-joule relationship')
        self.bolkEV = value('Boltzmann constant in eV/K')
        self.bolk = value('Boltzmann constant')
        self.planck = value('Planck constant')
        self.amutokg = value('atomic mass constant')
        self.cl = value('speed of light in vacuum')
