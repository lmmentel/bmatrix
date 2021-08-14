import os
import numpy as np
import ase.io
from bmatrix import (
    get_internals,
    recalculate_internals,
    get_bmatrix,
    complextype_to_dataframe,
)

data = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


def test_methanol(tmpdir):

    tmpdir.chdir()
    xyzpath = os.path.join(data, "meoh.xyz")
    atoms = ase.io.read(xyzpath)

    internals = get_internals(atoms)
    intdf = complextype_to_dataframe(internals)
    refintc = np.load(os.path.join(data, "meoh_internals.npy"))
    assert np.allclose(intdf["value"].values, refintc["value"])

    bmat = get_bmatrix(atoms, internals)
    refbmat = np.load(os.path.join(data, "meoh_bmatrix.npy"))
    assert np.allclose(bmat, refbmat)


def test_hton(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, "hton.xyz")
    atoms = ase.io.read(xyzpath)

    internals = get_internals(atoms)
    intdf = complextype_to_dataframe(internals)
    refintc = np.load(os.path.join(data, "hton_internals.npy"))
    assert np.allclose(intdf["value"].values, refintc["value"])

    bmat = get_bmatrix(atoms, internals)
    refbmat = np.load(os.path.join(data, "hton_bmatrix.npy"))
    assert np.allclose(bmat, refbmat)


def test_hafi_isobu(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, "hafi-isobu.xyz")
    atoms = ase.io.read(xyzpath)

    internals = get_internals(atoms)
    intdf = complextype_to_dataframe(internals)
    refintc = np.load(os.path.join(data, "hafi-isobu_internals.npy"))
    assert np.allclose(intdf["value"].values, refintc["value"])

    bmat = get_bmatrix(atoms, internals)
    refbmat = np.load(os.path.join(data, "hafi-isobu_bmatrix.npy"))
    assert np.allclose(bmat, refbmat)


def test_regenerate_internals(tmpdir):

    tmpdir.chdir()

    xyzpath = os.path.join(data, "meoh.xyz")
    atoms = ase.io.read(xyzpath)
    internals = get_internals(atoms)

    recalculate_internals(atoms, internals)
    intdf = complextype_to_dataframe(internals)
    refintc = np.load(os.path.join(data, "meoh_internals.npy"))
    assert np.allclose(intdf["value"].values, refintc["value"])
