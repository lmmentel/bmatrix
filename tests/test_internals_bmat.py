
import os
import numpy as np
import ase.io
from writeBmat import get_internals

data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')


def test_methanol(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, 'meoh.xyz')

    atoms = ase.io.read(xyzpath)

    internals, bmat = get_internals(atoms, return_bmatrix=True)

    refintc = np.load(os.path.join(data, 'meoh_internals.npy'))
    refbmat = np.load(os.path.join(data, 'meoh_bmatrix.npy'))

    assert np.allclose(internals['value'], refintc['value'])
    assert np.allclose(bmat, refbmat)


def test_hton(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, 'hton.xyz')

    atoms = ase.io.read(xyzpath)

    internals, bmat = get_internals(atoms, return_bmatrix=True)

    refintc = np.load(os.path.join(data, 'hton_internals.npy'))
    refbmat = np.load(os.path.join(data, 'hton_bmatrix.npy'))

    assert np.allclose(internals['value'], refintc['value'])
    assert np.allclose(bmat, refbmat)


def test_hafi_isobu(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, 'hafi-isobu.xyz')

    atoms = ase.io.read(xyzpath)

    internals, bmat = get_internals(atoms, return_bmatrix=True)

    refintc = np.load(os.path.join(data, 'hafi-isobu_internals.npy'))
    refbmat = np.load(os.path.join(data, 'hafi-isobu_bmatrix.npy'))

    assert np.allclose(internals['value'], refintc['value'])
    assert np.allclose(bmat, refbmat)
