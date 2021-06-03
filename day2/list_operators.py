"""Operators, that work on the atoms_string located in
atoms.info['key_value_pairs']['atoms_string'].
 This is both mutations and crossovers."""
import random
import numpy as np

from ase.data import atomic_numbers

from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_row_column


class ListOperator(OffspringCreator):
    """The base class for any list operation"""

    def __init__(self, identifier='atoms_string', separator='-',
                 verbose=False, num_muts=1, element_pool=None):
        OffspringCreator.__init__(self, verbose, num_muts=num_muts)

        self.separator = separator
        self.identifier = identifier

        if element_pool is not None:
            if not isinstance(element_pool[0], (list, np.ndarray)):
                self.element_pools = [element_pool]
            else:
                self.element_pools = element_pool

    def get_new_individual(self, parents):
        raise NotImplementedError

    def get_list(self, atoms, identifier=None, separator=None):
        if separator is None:
            separator = self.separator
        if identifier is None:
            identifier = self.identifier

        return atoms.info['key_value_pairs'][identifier].split(separator)


class RandomListCreation(ListOperator):
    """Create a random candidate from the element_pool
    of the specified length.

    Parameters
    ----------
    length : list
        A list of integers the same length as element_pool. Each integer
        says how many elements of the corresponding pool the candidate
        should hold. (See example)

    Example
    -------
    If element_pool = [['A', 'B', 'C'], ['a', 'b', 'c', 'd']]
    and length = [2, 4]
    a candidate could be: 'A-B-a-b-b-d'
    """

    def __init__(self, element_pool, length, identifier='atoms_string',
                 separator='-', verbose=False, num_muts=1):
        ListOperator.__init__(self, identifier, separator,
                              verbose, num_muts=num_muts,
                              element_pool=element_pool)

        self.length = length
        self.descriptor = 'RandomListCreation'

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        cand = []
        for pool, l in zip(self.element_pools, self.length):
            cand += random.sample(pool, l)

        indi.info['key_value_pairs'][self.identifier] = self.separator.join(
            cand)

        return (self.finalize_individual(indi),
                self.descriptor + ': Parent {0}'.format(f.info['confid']))


class RandomListMutation(ListOperator):
    """Mutate an element with another in the same element pool.
    If the individual consists of different groups of elements the element
    pool can be supplied as a list of lists.

    Parameters
    ----------
    element_pool : list
        Elements in the phase space. The elements can be grouped if the
        individual consist of different types of elements.
        The list should then be a list of lists e.g. [[list1], [list2]]

    identifier : str
        Where should the list be saved in the key_value_pairs.
        I.e. in atoms.info['key_value_pairs'][identifier]
        Note the list will be saved as a string using the separator between
        the list elements.
        Default: 'atoms_string'

    separator : str
        How should the string for saving be construted from the list or the
        list be constructed from the saved string.
        Default: '-'

        Example :
        separator = '-'
        string = separator.join(list)
        list = string.split(separator)

    verbose : bool
        Be verbose and explain more steps.
        Default: False
    """

    def __init__(self, element_pool, identifier='atoms_string', separator='-',
                 verbose=False, num_muts=1):
        ListOperator.__init__(self, identifier, separator,
                              verbose, num_muts=num_muts,
                              element_pool=element_pool)

        self.descriptor = 'RandomListMutation'

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        l = self.get_list(f)
        itbm_ok = False
        while not itbm_ok:
            itbm = random.choice(range(len(l)))  # index to be mutated
            for e in self.element_pools:
                if l[itbm] in e:
                    elems = e[:]
                    elems.remove(l[itbm])
                    itbm_ok = True
                    break

        l[itbm] = random.choice(elems)

        indi.info['key_value_pairs'][self.identifier] = self.separator.join(l)
        if self.verbose:
            fl = f.info['key_value_pairs'][self.identifier]
            print(self.descriptor,
                  'changed {1} to {0}'.format(self.separator.join(l),
                                              fl))

        return (self.finalize_individual(indi),
                self.descriptor + ': Parent {0}'.format(f.info['confid']))


class NeighborhoodListMutation(ListOperator):
    """Mutate an element with the nearest one in the periodic table
    from the same element pool.
    If the individual consists of different groups of elements the element
    pool can be supplied as a list of lists.

    Parameters
    ----------
    element_pool : list
        Elements in the phase space. The elements can be grouped if the
        individual consist of different types of elements.
        The list should then be a list of lists e.g. [[list1], [list2]]

    identifier : str
        Where should the list be saved in the key_value_pairs.
        I.e. in atoms.info['key_value_pairs'][identifier]
        Note the list will be saved as a string using the separator between
        the list elements.
        Default: 'atoms_string'

    separator : str
        How should the string for saving be construted from the list or the
        list be constructed from the saved string.
        Default: '-'

        Example :
        separator = '-'
        string = separator.join(list)
        list = string.split(separator)

    verbose : bool
        Be verbose and explain more steps.
        Default: False
    """

    def __init__(self, element_pool, identifier='atoms_string', separator='-',
                 verbose=False, num_muts=1):
        ListOperator.__init__(self, identifier, separator,
                              verbose, num_muts=num_muts,
                              element_pool=element_pool)

        self.descriptor = 'NeighborhoodListMutation'

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        l = self.get_list(f)
        itbm_ok = False
        while not itbm_ok:
            itbm = random.choice(range(len(l)))  # index to be mutated
            for e in self.element_pools:
                if l[itbm] in e:
                    elems = e[:]
                    elems.remove(l[itbm])
                    itbm_ok = True
                    break

        random.shuffle(elems)
        min_dist = np.inf
        el_list = string2symbols(l[itbm])
        rc = np.average([get_row_column(el) for el in el_list], axis=0)
        for el in elems:
            el_list = string2symbols(el)
            rc_el = np.average([get_row_column(e) for e in el_list], axis=0)
            dist = sum(np.abs(rc - rc_el))
            if dist < min_dist:
                min_dist = dist
                tbmt = el  # to be mutated to
        l[itbm] = tbmt

        indi.info['key_value_pairs'][self.identifier] = self.separator.join(l)

        return (self.finalize_individual(indi),
                self.descriptor + ': Parent {0}'.format(f.info['confid']))


class OnePointListCrossover(ListOperator):
    """Cross two candidates with one crossing point
    and return one candidate.

    Parameters
    ----------
    identifier : str
        Where should the list be saved in the key_value_pairs.
        I.e. in atoms.info['key_value_pairs'][identifier]
        Note the list will be saved as a string using the separator between
        the list elements.
        Default: 'atoms_string'

    separator : str
        How should the string for saving be construted from the list or the
        list be constructed from the saved string.
        Default: '-'

        Example :
        separator = '-'
        string = separator.join(list)
        list = string.split(separator)

    verbose : bool
        Be verbose and explain more steps.
        Default: False
    """

    def __init__(self,  identifier='atoms_string', separator='-',
                 verbose=False, num_muts=1):
        ListOperator.__init__(self, identifier, separator,
                              verbose, num_muts=num_muts)

        self.descriptor = 'OnePointListCrossover'

    def get_new_individual(self, parents):
        f, m = parents

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        fl = self.get_list(f)
        ml = self.get_list(m)
        cut_choices = [i for i in range(1, len(fl))]
        random.shuffle(cut_choices)
        for cut in cut_choices:
            nl = fl[:cut] + ml[cut:]
            ok = True
            if nl == fl or nl == ml:
                ok = False

            if ok:
                break

        if not ok:
            # No legal crossover could be made
            return None

        indi.info['key_value_pairs'][self.identifier] = self.separator.join(nl)
        parent_message = ': Parents {0} {1}'.format(f.info['confid'],
                                                    m.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)


class ListPermutation(ListOperator):
    """Permute two elements in the list.
    Both will be in the same pool of elements.

    If the individual consists of different groups of elements the element
    pool can be supplied as a list of lists.

    Parameters
    ----------
    element_pool : list
        Elements in the phase space. The elements can be grouped if the
        individual consist of different types of elements.
        The list should then be a list of lists e.g. [[list1], [list2]]

    identifier : str
        Where should the list be saved in the key_value_pairs.
        I.e. in atoms.info['key_value_pairs'][identifier]
        Note the list will be saved as a string using the separator between
        the list elements.
        Default: 'atoms_string'

    separator : str
        How should the string for saving be construted from the list or the
        list be constructed from the saved string.
        Default: '-'

        Example :
        separator = '-'
        string = separator.join(list)
        list = string.split(separator)

    verbose : bool
        Be verbose and explain more steps.
        Default: False
    """

    def __init__(self, element_pool,
                 identifier='atoms_string', separator='-',
                 verbose=False, num_muts=1):
        ListOperator.__init__(self, identifier, separator,
                              verbose, num_muts=num_muts,
                              element_pool=element_pool)

        self.descriptor = 'ListPermutation'

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        l = self.get_list(f)
        ok = False
        while not ok:
            i1, i2 = random.sample(range(len(l)), 2)
            for elems in self.element_pools:
                if l[i1] in elems and l[i2] in elems:
                    ok = True

        l[i1], l[i2] = l[i2], l[i1]

        indi.info['key_value_pairs'][self.identifier] = self.separator.join(l)

        return (self.finalize_individual(indi),
                self.descriptor + ': Parent {0}'.format(f.info['confid']))


def string2symbols(s):
    """Convert string to list of chemical symbols."""
    n = len(s)
    if n == 0:
        return []

    c = s[0]

    if c.isdigit():
        i = 1
        while i < n and s[i].isdigit():
            i += 1
        return int(s[:i]) * string2symbols(s[i:])

    if c == '(':
        p = 0
        for i, c in enumerate(s):
            if c == '(':
                p += 1
            elif c == ')':
                p -= 1
                if p == 0:
                    break
        j = i + 1
        while j < n and s[j].isdigit():
            j += 1
        if j > i + 1:
            m = int(s[i + 1:j])
        else:
            m = 1
        return m * string2symbols(s[1:i]) + string2symbols(s[j:])

    if c.isupper():
        i = 1
        if 1 < n and s[1].islower():
            i += 1
        j = i
        while j < n and s[j].isdigit():
            j += 1
        if j > i:
            m = int(s[i:j])
        else:
            m = 1
        symbol = s[:i]
        if symbol not in atomic_numbers:
            raise ValueError
        return m * [symbol] + string2symbols(s[j:])
    else:
        raise ValueError
