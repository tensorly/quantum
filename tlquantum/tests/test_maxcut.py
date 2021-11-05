import tensorly as tl
from tensorly.tt_tensor import TTTensor
from tensorly.tt_matrix import TTMatrix
from tensorly.testing import assert_array_almost_equal
from torch import randint
from torch import int as pt_int
from numpy import argmin
from numpy.random import permutation
from numpy.linalg import eig

from ..maxcut import brute_force_calculate_maxcut, calculate_cut
from ..tt_circuit import TTCircuit
from ..tt_operators import pauli_z, binary_hamiltonian
from ..tt_gates import Unitary, IDENTITY
from ..tt_precontraction import qubits_contract
from ..tt_state import spins_to_tt_state


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


err_tol = 1e-05


def test_calculate_cut():
    op, nqubits, nweights, ncontraq, ncontral = pauli_z(), 8, 40, 1, 1
    nlayers, ncontraq = 0, 1
    weights = tl.randn((nweights,))
    spins1, spins2 = randint(nqubits, (nweights,)), randint(nqubits, (nweights,))
    spins2[spins2==spins1] += 1
    spins2[spins2 >= nqubits] = 0
    H = binary_hamiltonian(op, nqubits, spins1, spins2, weights)
    H = qubits_contract(H, ncontraq)
    perm = tl.sign(tl.randn((nqubits, 1))).type(pt_int)
    spins = -perm
    perm[perm == -1] = 0
    state = spins_to_tt_state(perm)
    state = qubits_contract(state, 1)
    unitaries = [Unitary([IDENTITY() for i in range(nqubits)], nqubits, ncontraq)]
    circuit = TTCircuit(unitaries, ncontraq, ncontral)
    true_energy = circuit.forward_expectation_value(state, H)
    energy = calculate_cut(spins, spins1, spins2, weights)
    assert_array_almost_equal(energy, true_energy, decimal=err_tol)
    resort_inds = permutation(len(weights))
    spins1, spins2, weights = spins1[resort_inds], spins2[resort_inds], weights[resort_inds]
    new_energy = calculate_cut(spins, spins1, spins2, weights)
    assert_array_almost_equal(new_energy, true_energy, decimal=err_tol)


def test_brute_force_calculate_maxcut():
    op, nqubits, nweights = pauli_z(), 8, 40
    weights = tl.randn((nweights,))
    spins1, spins2 = randint(nqubits, (nweights,)), randint(nqubits, (nweights,))
    spins2[spins2==spins1] += 1
    spins2[spins2 >= nqubits] = 0
    H = binary_hamiltonian(op, nqubits, spins1, spins2, weights)
    vals, vecs = eig(TTMatrix(H).to_matrix().numpy())
    ind = argmin(vals)
    true_gs_energy, true_gs = vals[ind], vecs[ind]
    gs_energy, gs = brute_force_calculate_maxcut(nqubits, spins1, spins2, weights)
    assert_array_almost_equal(gs_energy, true_gs_energy, decimal=2)
    assert_array_almost_equal(TTTensor(spins_to_tt_state(gs)).to_tensor().reshape(-1,1), tl.tensor(true_gs).reshape(-1,1))
