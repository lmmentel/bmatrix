from scipy.constants import value, angstrom

ANGS2BOHR = angstrom / value("atomic unit of length")

physical_constants = {
    "Hartree2eV": value("Hartree energy in eV"),
    "ev2j": value("electron volt-joule relationship"),
    "bolkEV": value("Boltzmann constant in eV/K"),
    "bolk": value("Boltzmann constant"),
    "planck": value("Planck constant"),
    "amutokg": value("atomic mass constant"),
    "cl": value("speed of light in vacuum"),
}
