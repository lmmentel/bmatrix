
from __future__ import print_function

from math import *

import datastruct

import numpy as np

from physconstants import physical_constants as pC


class Intern:
    """
    Identification of primitive internal coordinates.

    Args:
        atomtypes (list of str) :
            List of atom's symbols

        radii (list) :
            List of atomic radii per symbol (same order as atomtypes)

        ascale (float) :
            Scaling factor for the radii

        bscale (float) :
            Scaling factor for the inter fragment distance

        anglecrit (int) :
            Critical angle

        torsioncrit (int) :
            Critical torsion angle

        fragcoord (int) :
            Fragment coordinate

        relax (bool) :
            Relax

        tors :

        subs (int) :

        natoms (int) :
            Number of atoms

        cell (array_like) :
            3 x 3 matrix of lattice vectors in atomic units

        fractional (array_like):
            Array of fractional coordinates (natoms x 3)

        cartesian (array_like) :
            Array of cartesian coordinates (natoms x 3)

        atomcounts (list of ints) :
            List of counts of different atoms

    """

    def __init__(self, atomtypes, radii, ascale, bscale, anglecrit,
                 torsioncrit, fragcoord, relax, TORS, SUBST, natoms,
                 cell, fractional, cartesian, atomcounts):

        self.atomtypes = atomtypes
        self.radii = radii
        self.ascale = ascale
        self.bscale = bscale
        self.anglecrit = anglecrit
        self.torsioncrit = torsioncrit
        self.fragcoord = fragcoord
        self.atoms = natoms
        self.relax = relax
        self.natoms = natoms
        self.cell = cell
        self.fractional = fractional
        self.cartesian = cartesian

        verbose = False
        if verbose:
            print(' input args in <Intern> '.center(80, '='))
            print('atomtypes   : ', atomtypes)
            print('radii       : ', radii)
            print('ascale      : ', ascale)
            print('bscale      : ', bscale)
            print('anglecrit   : ', anglecrit)
            print('torsioncrit : ', torsioncrit)
            print('fragcoord   : ', fragcoord)
            print('relax       : ', relax)
            print('TORS        : ', TORS)
            print('SUBST       : ', SUBST)
            print('natoms      : ', natoms)
            print('cell     : ')
            print(cell)
            print('fractional     : ')
            print(fractional)
            print('cartesian   : ')
            print(cartesian)
            print('atomcounts       : ', atomcounts)

        ascale = ascale / pC['AU2A']

        self.trust = 0.15  # criteria for acceptance of angle

        katoms = np.array(atomcounts) / 1
        if len(katoms) != len(radii):
            raise IOError
        for i in range(1, len(katoms)):
            katoms[i] = katoms[i] + katoms[i - 1]

        self.atomictags = []
        for i in range(len(atomcounts)):
            for j in range(atomcounts[i]):
                self.atomictags.append(self.atomtypes[i])

        # print radii
        atradii = ascale * np.array(radii)
        atrad = max(np.array(atradii))  # maximal allowed length of bond in the system

        self.criteria = self.set_criteria(atrad)
        self.pexcluded = [None, None, None]
        self.mexcluded = [None, None, None]
        intrawhat = []
        intrawhere = []

        # intracellparameters
        for i in range(len(self.fractional)):
            intrawhat.append(i)
            intrawhere.append(np.array([0, 0, 0]))

        # intercell parameters
        interfractional, interwhat, interwhere = self.inter_search(self.criteria)  # TODO: self.criteria shouldn't be passed as argument

        if len(interfractional) > 0:
            allfractional = np.zeros((len(self.fractional) + len(interfractional), 3),
                                  dtype=float)
            allfractional[:len(self.fractional)] = self.fractional
            allfractional[len(self.fractional):] = interfractional
        else:
            allfractional = self.fractional
        #allwhat=np.zeros((len(intrawhat)+len(interwhat)),Int)
        #allwhat[:len(intrawhat)]=intrawhat
        #allwhat[len(intrawhat):]=interwhat
        allwhat = intrawhat + interwhat
        #allwhere=np.zeros((len(intrawhere)+len(interwhere)),Int)
        #allwhere[:len(intrawhere)]=intrawhere
        #allwhere[len(intrawhere):]=interwhere
        allwhere = intrawhere + interwhere
        allcartesian = self.dirto_cart(allfractional)  # modified cell converted to cart coords.

        shortradii = 0.2 * np.array(atradii)    # minimal lengths
        longradii = np.array(atradii)           # upper limit for bond length

        bonds = self.bond_lengths(intrawhat, intrawhere,
                                  allcartesian, allwhat, allwhere, shortradii,
                                  longradii, katoms, 'R')

        fragments, substrate = self.frac_struct(bonds, SUBST)

        if len(bonds) == 0:
            print('no bonds detected, are you sure about the at. rad.?')

        angles, iangwhat, iangwhere = self.set_angles(bonds, 'A')

        self.internalcoords = []
        self.longinternalcoords = []

        if TORS == 1:
            torsions = self.set_dihedrals(iangwhat, iangwhere, 'T')
            self.internalcoords = bonds + angles + torsions
        else:
            self.internalcoords = bonds + angles
        self.bonds = bonds
        self.angles = angles

        #######################################################################

        if len(fragments) > 1:
            largestfrag = 0
            maxfr = 0
            for i in range(len(fragments)):
                if len(fragments[i]) > maxfr:
                    largestfrag = i

        if (len(fragments) > 1 or len(substrate) > 0) and self.fragcoord == 0:
            singles = []
            swhere = [0, 0, 0]
            for i in range(len(self.cartesian)):
                singles.append(datastruct.Complextype('simple', [1], 'X', [i],
                                                      [self.atomictags[i]],
                                                      [swhere],
                                                      self.cartesian[i][0],
                                                      'free'))
                singles.append(datastruct.Complextype('simple', [1], 'Y', [i],
                                                      [self.atomictags[i]],
                                                      [swhere],
                                                      self.cartesian[i][1],
                                                      'free'))
                singles.append(datastruct.Complextype('simple', [1], 'Z', [i],
                                                      [self.atomictags[i]],
                                                      [swhere],
                                                      self.cartesian[i][2],
                                                      'free'))
            if self.relax:
                for i in range(3):
                    singles.append(datastruct.Complextype('simple', [1], 'hX',
                                                          [i], [None],
                                                          [[0, 0, 0]],
                                                          self.cell[i][0],
                                                          'free'))
                    singles.append(datastruct.Complextype('simple', [1], 'hY',
                                                          [i], [None],
                                                          [[0, 0, 0]],
                                                          self.cell[i][1],
                                                          'free'))
                    singles.append(datastruct.Complextype('simple', [1], 'hZ',
                                                          [i], [None],
                                                          [[0, 0, 0]],
                                                          self.cell[i][2],
                                                          'free'))
            self.internalcoords += singles

        if (len(fragments) > 1 or len(substrate) > 0) and self.fragcoord != 0:
            longradii = bscale * atradii
            for i in range(len(longradii)):
                if self.atomtypes[i] == 'H':
                    longradii[i] *= 2.0

            longbonds = self.bond_fragments(fragments, substrate, longradii,
                                            katoms, 'R')

            if self.fragcoord == 2:
                for j in range(len(longbonds)):
                    longbonds[j].value = 5 / longbonds[j].value
                    longbonds[j].tag = 'IR1'
                if self.fragcoord == 3:
                    for j in range(len(longbonds)):
                        longbonds[j].value = 2000 / longbonds[j].value**6
                        longbonds[j].tag = 'IR6'

            for i in range(len(longbonds)):
                self.internalcoords += [longbonds[i]]
                self.longinternalcoords += [longbonds[i]]

    def set_criteria(self, atrad):
        """
        Projection of the bond-radius to the lattice vectors.
        """

        criteria = [None, None, None]
        #creates cutoff criterion
        norm1 = np.cross(self.cell[1], self.cell[2])  # normal vector to the 23 plane
        norm2 = np.cross(self.cell[2], self.cell[0])  # normal vector to the 31 plane
        norm3 = np.cross(self.cell[0], self.cell[1])  # normal vector to the 12 plane

        norm1 = norm1 / np.linalg.norm(norm1)  # normalized norm1
        norm2 = norm2 / np.linalg.norm(norm2)  # normalized norm2
        norm3 = norm3 / np.linalg.norm(norm3)  # normalized norm3

        cang10 = sum(norm1 * self.cell[0])  # cos of angle between norm1 and cell[0]
        cang21 = sum(norm2 * self.cell[1])  # cos of angle between norm2 and cell[1]
        cang32 = sum(norm3 * self.cell[2])  # cos of angle between norm3 and cell[2]

        criteria[0] = abs(2 * atrad / cang10)
        criteria[1] = abs(2 * atrad / cang21)
        criteria[2] = abs(2 * atrad / cang32)

        return np.array(criteria)

    def make_exclusions(self, which, criteria):
        """
        Excludes those data from row_d, which do not satisfy the
        criteria which = 0,1,2 i.e. a b c; .
        """

        pexclusions = []
        nexclusions = []

        # excluded for positive translation
        for i in range(len(self.fractional)):
            if self.fractional[i][which] > criteria[which]:
                pexclusions = pexclusions + [i]

        # excluded for negative translation
        for i in range(len(self.fractional)):
            if self.fractional[i][which] < 1 - criteria[which]:
                nexclusions = nexclusions + [i]

        return pexclusions, nexclusions

    def multy_cell(self, first, second, third, pexcluded, mexcluded):
        """
        Consideres atoms outside cell which can form intercell bonds.

        first, second, third - translations +/- 1
        """

        exclusions = []
        intfractional = []
        intwhat = []
        intwhere = []
        j = 0

        for i in (first, second, third):
            if i == 1:
                exclusions = exclusions + pexcluded[j]
            elif i == -1:
                exclusions = exclusions + mexcluded[j]
            j = j + 1
        transform = len(self.fractional) * [1]

        for i in exclusions:
            transform[i] = 0

        for i in range(len(self.fractional)):
            if transform[i] == 1:
                intfractional.append(self.fractional[i] + np.array([first,
                                                                    second,
                                                                    third]))
                intwhat.append(i)
                intwhere.append(np.array([first, second, third]))
        return intfractional, intwhat, intwhere

    def dirto_cart(self, fractional):
        """
        Conversion from fractional to cartesian coords.
        """

        carts = np.dot(fractional, self.cell)
        return carts

    def inter_search(self, criteria):
        '''
        group of those atoms which can not form intercell bonds
        '''

        pexcluded = [None, None, None]
        mexcluded = [None, None, None]
        interfractional = []
        interwhat = []
        interwhere = []
        for i in range(3):
            pexcluded[i], mexcluded[i] = self.make_exclusions(i, criteria)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if (i**2 + j**2 + k**2) != 0:
                        ifractional, iwhat, iwhere = self.multy_cell(i, j, k,
                                                                     pexcluded,
                                                                     mexcluded)
                        interfractional = interfractional + ifractional
                        interwhat = interwhat + iwhat
                        interwhere = interwhere + iwhere
        return interfractional, interwhat, interwhere

    def bond_fragments(self, fragments, substrate, radii_l, katoms, tag):
        bonds = []
        ibonds = []
        ibondwhat = []
        ibondwhattags = []
        ibondwhere = []

        if (len(fragments) < 2 and len(substrate) == 0):
            return bonds

        for jj in range(len(substrate)):
            target = 0
            while substrate[jj] >= katoms[target]:
                target = target + 1
            radii1_l = radii_l[target]
            a = self.fractional[substrate[jj]]
            for i in range(len(fragments)):
                for ii in range(len(fragments[i])):
                    target = 0
                    while fragments[i][ii] >= katoms[target]:
                        target = target + 1
                    radii2_l = radii_l[target]
                    criteria_l = (radii1_l + radii2_l)
                    b_ = self.fractional[fragments[i][ii]]
                    for t1 in (-1, 0, 1):
                        b = np.zeros(3, dtype=float)
                        b[0] = b_[0] + t1
                        for t2 in (-1, 0, 1):
                            b[1] = b_[1] + t2
                            for t3 in (-1, 0, 1):
                                b[2] = b_[2] + t3
                                r = b - a
                                r = np.dot(r, self.cell)
                                r = sum(r * r)**0.5
                                if r <= criteria_l:
                                    ibonds.append(r)
                                    ibondwhat.append([substrate[jj],
                                                      fragments[i][ii]])
                                    ibondwhattags.append([self.atomictags[substrate[jj]],
                                                          self.atomictags[fragments[i][ii]]])
                                    ibondwhere.append([[0, 0, 0],
                                                      [t1, t2, t3]])

        for i in range(len(fragments)):
            for j in range(len(fragments)):
                if j > i:
                    for ii in range(len(fragments[i])):
                        target = 0
                        while fragments[i][ii] >= katoms[target]:
                            target = target + 1
                        radii1_l = radii_l[target]
                        a = self.fractional[fragments[i][ii]]
                        for jj in range(len(fragments[j])):
                            if fragments[j][jj] != fragments[i][ii]:
                                target = 0
                                while fragments[j][jj] >= katoms[target]:
                                    target = target + 1
                                radii2_l = radii_l[target]
                                criteria_l = (radii1_l + radii2_l)
                                b_ = self.fractional[fragments[j][jj]]
                                for t1 in (-1, 0, 1):
                                    b = np.zeros(3, dtype=float)
                                    b[0] = b_[0] + t1
                                    for t2 in (-1, 0, 1):
                                        b[1] = b_[1] + t2
                                        for t3 in (-1, 0, 1):
                                            b[2] = b_[2] + t3
                                            r = b - a
                                            r = np.dot(r, self.cell)
                                            r = sum(r * r)**0.5
                                            if r <= criteria_l:
                                                ibonds.append(r)
                                                ibondwhat.append([fragments[i][ii],
                                                                  fragments[j][jj]])
                                                ibondwhattags.append([self.atomictags[fragments[i][ii]],
                                                                      self.atomictags[fragments[j][jj]]])
                                                ibondwhere.append([[0, 0, 0],
                                                                   [t1, t2, t3]])

        ksort = []
        for i in range(len(ibonds)):
            kk = [ibondwhat[i][0], i]
            ksort.append(kk)

        ksort.sort()

        for i in range(len(ksort)):
            bindex = ksort[i][1]
            bonds.append(datastruct.Complextype('simple', [1], tag,
                                                ibondwhat[bindex],
                                                ibondwhattags[bindex],
                                                ibondwhere[bindex],
                                                ibonds[bindex], 'free'))

        return bonds

    def bond_lengths(self, what, where, allcart, allwhat, allwhere,
                     radii_s, radii_l, katoms, tag):
        """
        Finds and calculates bond lengths.
        """

        ibonds = []
        ibondwhat = []
        ibondwhattags = []
        ibondwhere = []
        for i in range(len(what)):
            for j in range(len(allcart)):
                diffvec = self.cartesian[what[i]] - allcart[j]
                length = np.linalg.norm(diffvec)
                target = 0
                while what[i] >= katoms[target]:
                    target = target + 1
                radii1_s = radii_s[target]
                radii1_l = radii_l[target]
                target = 0
                while allwhat[j] >= katoms[target]:
                    target = target + 1
                radii2_s = radii_s[target]
                radii2_l = radii_l[target]
                criteria_s = (radii1_s + radii2_s)
                criteria_l = (radii1_l + radii2_l)
                if length > criteria_s and length <= criteria_l and what[i] <= allwhat[j]:
                    ibonds.append(length)
                    ibondwhat.append([what[i], allwhat[j]])
                    ibondwhattags.append([self.atomictags[what[i]],
                                          self.atomictags[allwhat[j]]])
                    ibondwhere.append([where[i], allwhere[j]])
        ksort = []
        for i in range(len(ibonds)):
            kk = [ibondwhat[i][0], i]
            ksort.append(kk)
        ksort.sort()
        bonds = []
        for i in range(len(ksort)):
            bindex = ksort[i][1]
            bonds.append(datastruct.Complextype('simple', [1], tag,
                                                ibondwhat[bindex],
                                                ibondwhattags[bindex],
                                                ibondwhere[bindex],
                                                ibonds[bindex], 'free'))
        return bonds

    def set_angles(self, bonds, tag):
        """
        Finds and calculates angles.
        """

        iangwhat = self.natoms * [None]
        iangwhere = self.natoms * [None]
        emerbonds = []

        for i in range(self.natoms):
            iangwhat[i] = []
            iangwhere[i] = []
        angles = []

        for i in range(len(bonds)):
            ii = bonds[i].what[0]
            jj = bonds[i].what[1]
            kk = bonds[i].where[0]
            ll = bonds[i].where[1]
            iangwhat[ii].append(jj)
            iangwhat[jj].append(ii)
            iangwhere[ii].append(ll)
            iangwhere[jj].append(kk - ll)

        for i in range(self.natoms):
            vectors = []
            # if len(self.topmap[i])>6:continue #######
            if len(self.topmap[i]) > self.anglecrit:
                continue
            for j in range(len(iangwhat[i])):
                windex = iangwhat[i][j]
                vec = self.fractional[i] - (self.fractional[windex] + iangwhere[i][j])
                vec = np.dot(vec, self.cell)
                value = np.linalg.norm(vec)
                both = [vec, value]
                vectors.append(both)
            for k in range(len(vectors)):
                for l in range(len(vectors)):
                    #if l<k:
                    if l < k and iangwhat[i][k] != i and iangwhat[i][l] != i:
                    #if l<k and iangwhat[i][k]!=iangwhat[i][l]:   # angles containing one atom twice -this has to be solved
                        angle = (sum(vectors[k][0] * vectors[l][0])) /\
                                (vectors[k][1] * vectors[l][1])
                        if angle > 1:
                            angle = 1
                        if angle < -1:
                            angle = -1
                        angle = abs(acos(angle))
                        # if not(sum(self.topology_matrix[iangwhat[i][k],:])>6 and sum(self.topology_matrix[iangwhat[i][l],:])>6 and sum(self.topology_matrix[i,:])>6):
                        if sin(angle) > self.trust:
                            if iangwhat[i][k] < iangwhat[i][l]:
                                awhat = [iangwhat[i][k], i, iangwhat[i][l]]
                                awhattags = [self.atomictags[iangwhat[i][k]],
                                             self.atomictags[i],
                                             self.atomictags[iangwhat[i][l]]]
                                awhere = [iangwhere[i][k], [0, 0, 0],
                                          iangwhere[i][l]]
                            else:
                                awhat = [iangwhat[i][l], i, iangwhat[i][k]]
                                awhattags = [self.atomictags[iangwhat[i][l]],
                                             self.atomictags[i],
                                             self.atomictags[iangwhat[i][k]]]
                                awhere = [iangwhere[i][l], [0, 0, 0],
                                          iangwhere[i][k]]
                                angles.append(datastruct.Complextype('simple',
                                    [1], tag, awhat, awhattags, awhere, angle,
                                    'free'))
                        else:
                            continue
                            # awhat=[iangwhat[i][k],iangwhat[i][l]]
                            # awhattags=[self.atomictags[iangwhat[i][k]],self.atomictags[iangwhat[i][l]]]
                            # awhere=[iangwhere[i][k],iangwhere[i][l]]
                            # emerbonds.append(datastruct.Complextype('simple',[1],'R',awhat,awhattags,\
                            # awhere,None,'free'))
                            # print 'stretched',i,awhat,awhere
        for i in range(len(emerbonds)):
            indx1 = emerbonds[i].what[0]
            indx2 = emerbonds[i].what[1]
            exitus = 0
            for j in range(i):
                if emerbonds[i].what[0] == emerbonds[j].what[0] and \
                        emerbonds[i].what[1] == emerbonds[j].what[1]:
                    if emerbonds[i].where[0][0] == emerbonds[j].where[0][0] and \
                            emerbonds[i].where[0][1] == emerbonds[j].where[0][1] and \
                            emerbonds[i].where[0][2] == emerbonds[j].where[0][2] and \
                            emerbonds[i].where[1][0] == emerbonds[j].where[1][0] and \
                            emerbonds[i].where[1][1] == emerbonds[j].where[1][1] and \
                            emerbonds[i].where[1][2] == emerbonds[j].where[1][2]:
                        exitus = 1
            if exitus == 1:
                continue
            for j in range(len(iangwhat[indx1])):
                if iangwhat[indx1][j] < indx2:
                    awhat = [iangwhat[indx1][j], indx1, indx2]
                    awhattags = [self.atomictags[iangwhat[indx1][j]],
                                 self.atomictags[indx1],
                                 self.atomictags[indx2]]
                    awhere = [iangwhere[indx1][j], [0, 0, 0],
                              emerbonds[i].where[1] - emerbonds[i].where[0]]
                else:
                    awhat = [indx2, indx1, iangwhat[indx1][j]]
                    awhattags = [self.atomictags[indx2],
                                 self.atomictags[indx1],
                                 self.atomictags[iangwhat[indx1][j]]]
                    awhere = [emerbonds[i].where[1] - emerbonds[i].where[0],
                              [0, 0, 0], iangwhere[indx1][j]]
                angle = self.calc_angle(awhat, awhere)
                if sin(angle) > self.trust:
                    angles.append(datastruct.Complextype('simple', [1], tag,
                                                         awhat, awhattags,
                                                         awhere, angle, 'free'))
            for j in range(len(iangwhat[indx2])):
                if iangwhat[indx2][j] < indx1:
                    awhat = [iangwhat[indx2][j], indx2, indx1]
                    awhattags = [self.atomictags[iangwhat[indx2][j]],
                                 self.atomictags[indx2],
                                 self.atomictags[indx1]]
                    awhere = [iangwhere[indx2][j], [0, 0, 0],
                              emerbonds[i].where[0] - emerbonds[i].where[1]]
                else:
                    awhat = [indx1, indx2, iangwhat[indx2][j]]
                    awhattags = [self.atomictags[indx1],
                                 self.atomictags[indx2],
                                 self.atomictags[iangwhat[indx2][j]]]
                    awhere = [emerbonds[i].where[0] - emerbonds[i].where[1],
                              [0, 0, 0], iangwhere[indx2][j]]
                angle = self.calc_angle(awhat, awhere)
                if sin(angle) > self.trust:
                    angles.append(datastruct.Complextype('simple', [1], tag,
                                                         awhat, awhattags,
                                                         awhere, angle, 'free'))
        return angles, iangwhat, iangwhere

    def calc_angle(self, awhat, awhere):

        v1 = self.fractional[awhat[1]] - (self.fractional[awhat[0]] + awhere[0])
        v1 = np.dot(v1, self.cell)
        v2 = self.fractional[awhat[1]] - (self.fractional[awhat[2]] + awhere[2])
        v2 = np.dot(v2, self.cell)
        angle = sum(v1 * v2) / (sum(v1**2) * sum(v2**2))**0.5
        if angle > 1:
            angle = 1.0
        if angle < -1:
            angle = -1.0
        return acos(angle)

    def set_dihedrals(self, iangwhat, iangwhere, tag):
        """
        Finds and calculates dihedral angles.
        """

        torsions = []
        for i in range(self.natoms):
            if len(iangwhat[i]) > 1:
                secondwhat = i
                # if len(self.topmap[secondwhat])>4:continue
                secondwhere = np.array([0, 0, 0])
                for j in range(len(iangwhat[i])):
                    firstwhat = iangwhat[i][j]
                    # if (len(self.topmap[secondwhat])>4 and len(self.topmap[firstwhat])>4):continue
                    if (len(self.topmap[secondwhat]) > self.torsioncrit and
                            len(self.topmap[firstwhat]) > self.torsioncrit):
                        continue
                    if firstwhat == secondwhat:
                        continue
                    firstwhere = iangwhere[i][j]
                    for k in range(len(iangwhat[i])):
                        if k != j and iangwhat[i][k] > secondwhat:
                            thirdwhat = iangwhat[i][k]
                            if thirdwhat == firstwhat or thirdwhat == secondwhat:
                                continue
                            thirdwhere = iangwhere[i][k]
                            # if len(self.topmap[thirdwhat])>4:continue
                            if (len(self.topmap[thirdwhat]) > 4 and
                                    len(self.topmap[firstwhat]) > 4):
                                continue
                            for l in range(len(iangwhat[thirdwhat])):
                                fourthwhat = iangwhat[thirdwhat][l]
                                if fourthwhat == firstwhat or fourthwhat == secondwhat or fourthwhat == thirdwhat:
                                    continue
                                fourthwhere = iangwhere[thirdwhat][l] + thirdwhere
                                itorsion = self.calculate_da(firstwhat,
                                                             secondwhat,
                                                             thirdwhat,
                                                             fourthwhat,
                                                             firstwhere,
                                                             thirdwhere,
                                                             fourthwhere)
                                if itorsion is not None:
                                    torwhat = [firstwhat, secondwhat,
                                               thirdwhat, fourthwhat]
                                    torwhattags = [self.atomictags[firstwhat],
                                                   self.atomictags[secondwhat],
                                                   self.atomictags[thirdwhat],
                                                   self.atomictags[fourthwhat]]
                                    torwhere = [firstwhere, secondwhere,
                                                thirdwhere, fourthwhere]
                                    torsions.append(datastruct.Complextype(
                                        'simple', [1], tag, torwhat,
                                        torwhattags, torwhere, itorsion,
                                        'free'))
        return torsions

    def calculate_da(self, firstwhat, secondwhat, thirdwhat, fourthwhat,
                     firstwhere, thirdwhere, fourthwhere):
        """
        Calculates dihedral angles.
        """

        if firstwhat == secondwhat or firstwhat == thirdwhat or \
                firstwhat == fourthwhat or secondwhat == thirdwhat or \
                secondwhat == fourthwhat or thirdwhat == fourthwhat:
            return None      # this is provisorium, will be fixed soon!!!

        a = self.fractional[firstwhat] + firstwhere
        b = self.fractional[secondwhat]
        c = self.fractional[thirdwhat] + thirdwhere
        d = self.fractional[fourthwhat] + fourthwhere
        checkpoint = np.linalg.norm(a - d)

        if checkpoint != 0:
            vector1 = a - b
            vector2 = b - c
            vector3 = c - d

            vector1 = np.dot(vector1, self.cell)
            vector2 = np.dot(vector2, self.cell)
            vector3 = np.dot(vector3, self.cell)

            vector1size = np.linalg.norm(vector1)
            vector2size = np.linalg.norm(vector2)
            vector3size = np.linalg.norm(vector3)

            cross1 = np.cross(vector1, vector2)
            cross2 = np.cross(vector2, vector3)

            cross1_size = np.linalg.norm(cross1)
            cross2_size = np.linalg.norm(cross2)

            treshold1 = cross1_size / (vector1size * vector2size)
            treshold2 = cross2_size / (vector2size * vector3size)

            alph1 = sum(vector1 * vector2) / (vector1size * vector2size)
            alph2 = sum(vector2 * vector3) / (vector2size * vector3size)
            if alph1 > 1:
                alph1 = 1
            if alph1 < -1:
                alph1 = -1
            if alph2 > 1:
                alph2 = 1
            if alph2 < -1:
                alph2 = -1

            alph1 = acos(alph1)
            alph2 = acos(alph2)

            if cross1_size != 0 and cross2_size != 0:
                fuck = sum(cross1 * cross2) / (cross1_size * cross2_size)
                if fuck > 1:
                    fuck = 1.0
                if fuck < -1:
                    fuck = -1.0
                dangle = acos(fuck)
                if sum(cross1 * vector3) >= 0:
                    dangle = -dangle
                if abs(sin(alph1)) > self.trust and abs(sin(alph2)) > self.trust:  # and abs(sin(dangle))>self.trust:
                    return dangle
            return None

    def frac_struct(self, bonds, SUBST):
        """
        This function finds how many structural
        fragments are there. Returns array 'molecules'
        in which atoms of each fraction except of
        the largest one are listed.
        """

        # self.topology=np.zeros((natoms,natoms),dtype=float)

        self.topology = np.zeros((self.natoms, self.natoms))
        self.topology_matrix = np.zeros((self.natoms, self.natoms))

        fract = self.natoms * [None]
        self.topmap = self.natoms * [None]
        for i in range(len(fract)):
            fract[i] = [i]
            self.topmap[i] = []
        for i in range(len(bonds)):
            index1 = bonds[i].what[0]
            index2 = bonds[i].what[1]
            self.topology[index1, index2] += 1
            self.topology[index2, index1] += 1
            self.topology_matrix[index1, index2] += 1
            self.topology_matrix[index2, index1] += 1

            if fract[index1] is None:
                fract[index1] = []
            if index2 >= index1:
                fract[index1].append(index2)
            self.topmap[index1].append(index2)
            self.topmap[index2].append(index1)
        self.topology_matrix_adds = 1 * self.topology_matrix

        for i in range(self.natoms):
            self.topology[i, i] = 2
            dummy = 0
            for j in range(self.natoms):
                if self.topology[i, j] == 1:
                    dummy = 1
                    for k in range(self.natoms):
                        if self.topology[i, k] == 1 and self.topology[j, k] == 0:
                            self.topology[j, k] = 1
                        if self.topology[i, k] == 2:
                            self.topology[j, k] = 2
            if dummy == 1:
                self.topology[i] = 0
        fractions = self.find_fragments(fract)
        substrate = []
        if len(fractions) == 1:
            print('ONE FRAGMENT (MOLECULE OR SOLID) WAS DETECTED.')
            # identify substrate
            fract = []
            for i in range(len(self.topology_matrix)):
                if sum(self.topology_matrix[i]) >= SUBST:
                    self.topology_matrix_adds[i, :] = 0
                    self.topology_matrix_adds[:, i] = 0
                    fract.append([None])
                    substrate.append(i)
                else:
                    fract.append([i])
            if len(substrate) > 0:
                for ii in range(len(fract)):
                    if fract[ii][0] is not None:
                        i = fract[ii][0]
                        for j in range(self.natoms):
                            if j > i:
                                if self.topology_matrix_adds[i, j] == 1:
                                    fract[ii].append(j)
                fractions = self.find_fragments(fract)
                print('SUBSTRATE ATOMS DETECTED: ', substrate)
                print('INTERMOL. DISTANCES BETWEEN ADD-MOLECULES WILL BE ADDED')
        else:
            print(len(fractions), 'FRAGMENTS DETECTED')
        return fractions, substrate

    def find_fragments(self, fract):
        """
        """

        fractions = []
        for i in range(len(fract)):
            if fract[i][0] is not None:
                fract[i].sort()
                indx = 0
                nextind = fract[i][indx]
                while nextind <= i:
                    indx = indx + 1
                    if indx >= len(fract[i]):
                        break
                    nextind = fract[i][indx]
                if nextind > i:
                    for j in range(len(fract[i])):
                        fract[nextind].append(fract[i][j])
                    fract[i] = [None]
        for i in range(len(fract)):
            if fract[i][0] is not None:
                newfract = [fract[i][0]]
                for j in range(1, len(fract[i])):
                    if fract[i][j] != fract[i][j - 1]:
                        newfract.append(fract[i][j])
                fractions.append(newfract)

        return fractions

    def make_singles(self, molecules):
        """
        Createsian xyz coordinates for given atoms
        """

        singles = 3 * len(molecules) * [None]
        for i in range(len(molecules)):
            cartusa = self.cartesian[molecules[i]]
            for j in range(3):
                indx = 3 * i + j
                singles[indx] = cartusa[j]
        return singles
