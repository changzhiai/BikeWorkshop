import os
import numpy as np
from ase.db import connect
from ase.ga import set_raw_score
from ase import Atoms
from operator import itemgetter

# def get_hof(row):
#     from ase.phasediagram import PhaseDiagram
#     try:
#         pd = PhaseDiagram(refs, filter=row.formula, verbose=False)
#         e0 = pd.decompose(row.formula)[0]
#         hof = (row.energy - e0) / row.natoms
#     except (AssertionError, QhullError):
#         hof = row.heat_of_formation_all
#     return hof


def get_hof(row):
    return row.heat_of_formation_all


def get_atoms_string(atoms):
    if atoms is None:
        return None
    if isinstance(atoms, tuple):
        return atoms[0].info['key_value_pairs']['atoms_string']
    return atoms.info['key_value_pairs']['atoms_string']


def evaluate(atoms, refs):
    cand_row = refs[get_atoms_string(atoms)]
    rs = get_raw_score(cand_row, method='product')
    # If you want to see every tested candidate printed to the screen:
    # Comment the following lines in
    # print(atoms.info['key_value_pairs']['atoms_string'], rs,
    #       atoms.info['key_value_pairs']['origin'])
    set_raw_score(atoms, raw_score=rs)


def evaluate_two_photons(atoms, refs, minimum_band_gap):
    cand_row = refs[get_atoms_string(atoms)]
    rs = get_raw_score_two_photons(
        cand_row, minimum_band_gap, method='product')
    set_raw_score(atoms, raw_score=rs)


def get_evaluated_set(ga_db_file):
    con = connect(ga_db_file)
    calced = set()
    for row in con.select('relaxed=1'):
        calced.add(row.atoms_string)
    return calced


def set_syms(atoms, row):
    atoms += Atoms(row.symbols)
    # atoms.set_chemical_symbols(row.symbols)
    atoms.set_positions(row.positions)
    atoms.pbc = row.pbc
    atoms.cell = row.cell


def get_raw_score(row, verbose=False, method='sum'):
    # Heat of formation should be below some threshold
    hof = get_hof(row)
    if hof < .1:
        hof_rs = 10.
    else:
        hof_rs = 10. * np.exp(-(hof - .1))
    scores = {'hof': hof_rs}

    # Band gap should ideally be 1.23 eV, more is ok but will
    # take advantage of fewer photons
    # Use the indirect gap as an including (as opposed to excluding) measure
    # If indirect gap is too small use direct gap

    # Determine gap
    gap = row.gllbsc_ind_gap
    if gap < 1.23:
        gap = row.gllbsc_dir_gap

    # Get raw score contribution

    # # Use approximated solar spectrum
    # from bbs import get_normalized_efficiency
    # if gap >= 1.23:
    #     gap_rs = 10. * get_normalized_efficiency(gap)
    # else:
    #     gap_rs = 0.

    # Use flat value between 1.23 and 3.0 eV
    if gap < 1.23:
        gap_rs = 10. * np.exp(gap - 1.23)
    elif gap > 3.:
        gap_rs = 10. * np.exp(-(gap - 3.))
    else:
        gap_rs = 10.

    scores['gap'] = gap_rs

    # Valence and conduction band should be positioned at the
    # O2/H2O and H+/H2 potential respectively
    # Setting 0 at the H+/H2 level by subtracting 4.5 eV
    CB_edge = row.CB_ind - 4.5  # should be < 0 eV
    if CB_edge > 0.:
        CB_edge = row.CB_dir - 4.5
    if CB_edge < 0.:
        CB_rs = 5.
    else:
        CB_rs = 5. * np.exp(-CB_edge)
    scores['CB_rs'] = CB_rs

    VB_edge = row.VB_ind - 4.5  # should be > 1.23 eV
    if VB_edge < 1.23:
        VB_edge = row.VB_dir - 4.5
    if VB_edge > 1.23:
        VB_rs = 5.
    else:
        VB_rs = 5. * np.exp(-abs(VB_edge - 1.23))
    scores['VB_rs'] = VB_rs

    if verbose:
        print(scores)

    if method == 'sum':
        rs = sum(scores.values())
    elif method == 'product':
        rs = scores['hof'] * (scores['gap'] +
                              scores['CB_rs'] +
                              scores['VB_rs'])

    return rs


def get_raw_score_two_photons(row, minimum_band_gap, verbose=False, method='sum'):
    # Heat of formation should be below some threshold
    hof = get_hof(row)
    if hof < .1:
        hof_rs = 10.
    else:
        hof_rs = 10. * np.exp(-(hof - .1))
    scores = {'hof': hof_rs}

    # Band gap should ideally be minimum_band_gap eV, more is ok but will
    # take advantage of fewer photons, less will require lower internal losses
    # Use the indirect gap as an including (as opposed to excluding) measure
    # If indirect gap is too small use direct gap

    # Determine gap
    gap = row.gllbsc_ind_gap
    if gap < minimum_band_gap:
        gap = row.gllbsc_dir_gap

    # Get raw score contribution

    # Use flat value between minimum_band_gap and 2.4 eV
    if gap < minimum_band_gap:
        gap_rs = 10. * np.exp(gap - minimum_band_gap)
    elif gap > 2.4:
        gap_rs = 10. * np.exp(-(gap - 2.4))
    else:
        gap_rs = 10.

    scores['gap'] = gap_rs

    # Valence band should be positioned at the
    # O2/H2O potential
    # Setting 0 at the H+/H2 level by subtracting 4.5 eV
    VB_edge = row.VB_ind - 4.5  # should be > 1.23 eV
    if VB_edge < 1.23:
        VB_edge = row.VB_dir - 4.5
    if VB_edge > 1.23:
        VB_rs = 5.
    else:
        VB_rs = 5. * np.exp(-abs(VB_edge - 1.23))
    scores['VB_rs'] = VB_rs

    if verbose:
        print(scores)

    if method == 'sum':
        rs = sum(scores.values())
    elif method == 'product':
        rs = scores['hof'] * (scores['gap'] +
                              scores['VB_rs'])

    return rs


def generate_top_list(fname, query, raw_score_function, *args, **kwargs):
    ref_db = connect('cubic_perovskites.db')
    all_list = []
    for row in ref_db.select(query):
        s = '-'.join([row.A_ion, row.B_ion, row.anion])
        all_list.append((s, raw_score_function(row, *args, **kwargs)))

    fd = open(fname, 'w')
    # sort the list by the raw score
    all_list.sort(key=itemgetter(1), reverse=True)
    for st, rs in all_list:
        print(st, rs, file=fd)
    fd.close()

def get_db_file_name(project):
    legal_projects = ['perovskite', 'Ni118Pt29_emt']
    msg = 'Argument should be {}'.format(' or '.join(legal_projects))
    assert project in legal_projects, msg
    folder = os.path.join('/work3/projects/week11/users/', os.environ['USER'])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, project + '.db')

        

if __name__ == '__main__':
    # Getting the phase spaces A_ion, B_ion and anion
    generate_top_list('oxides-two-photon.txt',
                      'anion=O3', get_raw_score_two_photons,
                      1.7, method='product')
    # A_ions, B_ions, anions = set(), set(), set()
    # db = connect('cubic_perovskites.db')
    # for dct in db.select('anion=O3'):
    #     A_ions.add(dct.A_ion)
    #     B_ions.add(dct.B_ion)
    #     anions.add(dct.anion)
    # assert A_ions == B_ions
    # AB_ions = sorted(list(A_ions))
    # print(len(AB_ions))
    # anions = sorted(list(anions))

    # # print 'A ions: {0}'.format(sorted(list(A_ions)))
    # # print 'B ions: {0}'.format(sorted(list(B_ions)))
    # # print 'Anions: {0}'.format(sorted(list(anions)))

    # # Create random candidates
    # import random
    # cand = random.sample(AB_ions, 2) + random.sample(anions, 1)
    # cand_row = db.get(A_ion=cand[0], B_ion=cand[1], anion=cand[2])
    # print(get_hof(cand_row), cand_row.CB_ind - 4.5,
    #       cand_row.VB_ind - 4.5, cand_row.gllbsc_ind_gap)
    # print(get_raw_score(cand_row))

    # print('14300', get_raw_score(db.get(14300)))
    # print('14688', get_raw_score(db.get(14688)))
