import tensorly as tl
tl.set_backend('pytorch')
from tensorly.tt_matrix import TTMatrix
from torch import minimum, maximum

from .tt_sum import tt_matrix_sum


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


def binary_hamiltonian(op, nqubits, qubits1, qubits2, weights):
    """Generates tt-tensor classical Ising model Hamiltonian (two-qubit interaction terms in a single basis).
    Hamiltonian of the form:
    
    H = sum_i omega_i sigma_ind1(i) sigma_ind2(i)

    where omega_i are the Hamiltonian weights, sigma is the operator specified by op, and ind1, ind2 are
    the qubit numbers specified of index i.

    spins and weight values.

    Parameters
    ----------
    op : tt-tensor, single-qubit operator to encode MaxCut graph
    nqubits : int, number of qubits (vertices) to encode in MaxCut problem
    qubits1 : List/tensor of ints, qubit indices
    qubits2 : List/tensor of ints, qubit indices
    weights : List/tensor of real floats, graph weights

    Returns
    -------
    Hamiltonian encoding specified classical Ising model graph.
    """
    H, inds_min, inds_max = [], minimum(qubits1, qubits2), maximum(qubits1, qubits2)
    for i in range(0, len(qubits1)):
        H = tt_matrix_sum(H, _two_qubit_interaction(op, op, inds_min[i], inds_max[i], weights[i], nqubits))
    return [*H]


def unary_hamiltonian(op, nqubits, qubits, weights):
    """Generates tt-tensor unitary of one single-qubit operator per qubit.

    Parameters
    ----------
    op : tt-tensor, single-qubit operator
    nqubits : int, number of qubits (vertices) to encode in MaxCut problem
    qubits : List/tensor of ints, qubit indices
    weights : List/tensor of real floats, graph weights

    Returns
    -------
    Unitary of one single-qubit operator per qubit in tt-tensor format
    """
    #H = TTMatrix([weights[0]*op] + [IDENTITY for i in range(1, nqubits)])
    H, iden = [], identity(device=op.device)
    for q in qubits:
        H = tt_matrix_sum(H, TTMatrix([iden for i in range(q)] + [weights[q]*op] + [iden for i in range(q+1, nqubits)]))
    return H


def _two_qubit_interaction(op1, op2, ind1, ind2, weight, nqubits):
    """Generates tt-tensor Hamiltonian of two single-qubit operators with given weight.

    Parameters
    ----------
    op1 : tt-tensor, single-qubit operator
    op2 : tt-tensor, single-qubit operator
    ind1 : int, index of first qubit
    ind2 : int, index of second qubit
    weight : float, weight of two-qubit interaction
    nqubits : int, number of qubits in the Hamiltonian
    device : string, device on which to run the computation.
    
    Returns
    -------
    Hamiltonian of n qubits with two interacions at ind1 and ind2 in tt-tensor form
    """
    iden = identity(device=op1.device)
    return TTMatrix([iden for k in range(ind1)] + [weight*op1] + [iden for k in range(ind2-ind1-1)] + [op2] + [iden for k in range(ind2+1, nqubits, 1)])


def identity(device=None):
    """Single-qubit identity opertor in the tt-tensor format.

    Parameters
    ----------
    device : string, device on which to run the computation.
    
    Returns
    -------
    Identity operator in tt-form.
    """
    return tl.tensor([[[[1],[0]],[[0],[1]]]], device=device)


def pauli_x(device=None):
    """Single-qubit Pauli-X opertor in the tt-tensor format.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Pauli-X operator in tt-form.
    """
    return tl.tensor([[[[0],[1]],[[1],[0]]]], device=device)


def pauli_z(device=None):
    """Single-qubit Pauli-Z opertor in the tt-tensor format.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Pauli-Z operator in tt-form.
    """
    return tl.tensor([[[[1],[0]],[[0],[-1]]]], device=device)
