import tensorly as tl
from tensorly.tt_tensor import TTTensor
import itertools
from torch import int as pt_int

from .tt_circuit import qubits_contract, TTCircuit


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


def calculate_cut(spins, spins1, spins2, weights, get_cut=False):
    """Calculates the MaxCut value of a given state (set of spins) for a given graph (weights). The weights
    can be given in an arbitrary order, which makes the indices of the spins for each weight necessary.

    Parameters
    ----------
    spins : List/tensor of real floats, state spin values
    wspins1 : List/tensor of ints, spin indices
    wspins2 : List/tensor of ints, spin indices
    weights : List/tensor of real floats, graph weights
    get_cut : Boolean, if False - return Ising energy definition, if True - return traditional MaxCut metric
    
    Returns
    -------
    The energy/MaxCut metric for the state of spins of the graph described by spins1, spins2, and weights.
    """
    spins1, spins2, weights = spins1.reshape((-1,)), spins2.reshape((-1,)), weights.reshape((-1,1))
    if get_cut:
        energy = tl.sum(weights*(1-spins[spins1]*spins[spins2]))/2.
    else:
        energy = tl.sum(spins[spins1]*spins[spins2]*weights)
    return energy


def brute_force_calculate_maxcut(nqubits, spins1, spins2, weights):
    """Brute force calculation of MaxCut for a given set of spins and weights. Caution: scales exponentially in number of qubits.

    Parameters
    ----------
    nqubits : int, number of qubits
    wspins1 : List/tensor of ints, spin indices
    wspins2 : List/tensor of ints, spin indices
    weights : List/tensor of real floats, graph weights
    
    Returns
    -------
    The MaxCut for the graph described by spin1, spin2, and weights.
    """
    tot, bits = 2**(nqubits-1), [BIT for i in range(nqubits-1)]
    bits = list(itertools.product(*bits))
    bits = [list(tup) for tup in bits]
    min_energy, min_perm = 1000, None
    for perm in bits:
        spins = -tl.tensor([0]+perm).to(pt_int)
        spins[spins == 0] = 1
        energy = calculate_cut(spins.reshape(-1,1), spins1, spins2, weights)
        if energy < min_energy:
            min_energy, min_perm = energy, [0]+perm
    return min_energy, min_perm


BIT = [0, 1]
