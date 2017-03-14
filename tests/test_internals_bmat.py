
import os
import numpy as np
import ase.io
from writeBmat import (get_internals, recalculate_internals,
                       internals_to_array, get_bmatrix)

data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')


def test_methanol(tmpdir):

    tmpdir.chdir()
    xyzpath = os.path.join(data, 'meoh.xyz')
    atoms = ase.io.read(xyzpath)

    internals = get_internals(atoms)
    intc = internals_to_array(internals)
    refintc = np.load(os.path.join(data, 'meoh_internals.npy'))
    assert np.allclose(intc['value'], refintc['value'])

    bmat = get_bmatrix(atoms, internals)
    refbmat = np.load(os.path.join(data, 'meoh_bmatrix.npy'))
    assert np.allclose(bmat, refbmat)


def test_hton(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, 'hton.xyz')
    atoms = ase.io.read(xyzpath)

    internals = get_internals(atoms)
    intc = internals_to_array(internals)
    refintc = np.load(os.path.join(data, 'hton_internals.npy'))
    assert np.allclose(intc['value'], refintc['value'])

    bmat = get_bmatrix(atoms, internals)
    refbmat = np.load(os.path.join(data, 'hton_bmatrix.npy'))
    assert np.allclose(bmat, refbmat)


def test_hafi_isobu(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, 'hafi-isobu.xyz')
    atoms = ase.io.read(xyzpath)

    internals = get_internals(atoms)
    intc = internals_to_array(internals)
    refintc = np.load(os.path.join(data, 'hafi-isobu_internals.npy'))
    assert np.allclose(intc['value'], refintc['value'])

    bmat = get_bmatrix(atoms, internals)
    refbmat = np.load(os.path.join(data, 'hafi-isobu_bmatrix.npy'))
    assert np.allclose(bmat, refbmat)


def test_regenerate_internals(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, 'meoh.xyz')
    atoms = ase.io.read(xyzpath)
    internals = get_internals(atoms)

    recalculate_internals(atoms, internals)
    intc = internals_to_array(internals)
    refintc = np.load(os.path.join(data, 'meoh_internals.npy'))
    assert np.allclose(intc['value'], refintc['value'])
