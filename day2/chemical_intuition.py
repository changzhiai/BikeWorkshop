import os
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.symbols import string2symbols
from mendeleev import element
from ase.db import connect
from operator import itemgetter
from itertools import product

all_elems = ['Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Co',
             'Cr', 'Cs', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 'Hg', 'In', 'Ir', 'K',
             'La', 'Li', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Ni', 'Os', 'Pb', 'Pd',
             'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'Sb', 'Sc', 'Si', 'Sn', 'Sr', 'Ta',
             'Te', 'Ti', 'Tl', 'V', 'W', 'Y', 'Zn', 'Zr', 'N', 'O', 'F', 'S']


def is_possible_oxidation_state_0(elements, all_oxistates):
    all_states = [all_oxistates[e] for e in elements]
    for c in product(*all_states):
        if sum(c) == 0:
            return True
    return False


def get_total_electrons(elements):
    return sum([atomic_numbers[e] for e in elements])


def get_goldschmidt_tolerance_factor(A, B, X):
    return (get_radius(A) + get_radius(X)) / (np.sqrt(2) * (get_radius(B) +
                                                            get_radius(X)))


def get_radius(e):
    """Using covalent_radii instead of the questionable procedure of taking
    the unweighted average of all available ionic radii."""
    r = 0
    s = string2symbols(e)
    for elem in s:
        r += covalent_radii[atomic_numbers[elem]]
    return r / len(s)


def generate_chemical_intuition_list(fname):
    ref_db = connect('cubic_perovskites.db')
    all_oxistates = dict((e, element(e).oxistates) for e in all_elems)
    all_list = []
    for row in ref_db.select('anion'):
        if get_total_electrons(row.symbols) % 2 == 0:
            # Even number of electrons
            if is_possible_oxidation_state_0(row.symbols, all_oxistates):
                # Total oxidation state of 0 is possible
                s = '-'.join([row.A_ion, row.B_ion, row.anion])
                all_list.append((s, abs(get_goldschmidt_tolerance_factor(
                    row.A_ion, row.B_ion, row.anion) - 1)))

    fd = open(fname, 'w')
    all_list.sort(key=itemgetter(1))
    for st, rs in all_list:
        print(st, rs, file=fd)
    fd.close()


def get_chemical_intuition_found_list(top_list):
    fname = 'chemical_intuition_list.txt'
    if not os.path.isfile(fname):
        generate_chemical_intuition_list(fname)

    f = open(fname, 'r')
    intuition_found = [0]
    i = 0
    for l in f:
        i += 1
        cand = l.split(' ')[0]
        if cand in top_list:
            intuition_found.append(i)
    f.close()

    return intuition_found
