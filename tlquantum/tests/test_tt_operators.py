import tensorly as tl
from tensorly.tt_matrix import TTMatrix
from tensorly.testing import assert_array_almost_equal
from torch import randint


from ..tt_operators import identity, pauli_z, binary_hamiltonian, unary_hamiltonian


def test_identity():
    ID = TTMatrix([identity()])
    assert_array_almost_equal(ID.to_matrix(), tl.tensor([[1,0],[0,1]]))


def test_pauli_z():
    pauliZ = TTMatrix([pauli_z()])
    assert_array_almost_equal(pauliZ.to_matrix(), tl.tensor([[1,0],[0,-1]]))


def test_binary_hamiltonian():
    op, nqubits, nterms = pauli_z(), 4, 5
    weights = tl.randn((nterms,))
    spins1, spins2 = randint(nqubits, (nterms,)), randint(nqubits, (nterms,))
    spins2[spins2==spins1] += 1
    spins2[spins2 >= nqubits] = 0
    H = binary_hamiltonian(op, nqubits, spins1, spins2, weights)
    H_matrix, count = tl.zeros((2**nqubits, 2**nqubits)), 0
    op_matrix = TTMatrix([op]).to_matrix()
    for i in range(nterms):
        ind1, ind2 = min(spins1[i], spins2[i]), max(spins1[i], spins2[i])
        identity_dim = max(1, 2**ind1)
        term = tl.kron(tl.eye(identity_dim), op_matrix)
        identity_dim = 2**(ind2-ind1-1)
        term = tl.kron(tl.kron(term, tl.eye(identity_dim)), op_matrix)
        identity_dim = 2**(nqubits-ind2-1)
        term = tl.kron(term, tl.eye(identity_dim))
        H_matrix += term*weights[count]
        count += 1
    assert_array_almost_equal(TTMatrix(H).to_matrix(), H_matrix)


def test_unary_hamiltonian():
    op, nqubits, nterms = pauli_z(), 3, 3
    qubits, weights = tl.tensor([0,1,2], dtype=int), tl.randn((nterms,))
    H = unary_hamiltonian(op, nqubits, qubits, weights)
    H = TTMatrix(H).to_matrix()
    op, iden = TTMatrix([op]).to_matrix(), tl.tensor([[1,0],[0,1]])
    H_manual = tl.kron(weights[0]*op, tl.kron(iden, iden)) + tl.kron(iden, tl.kron(weights[1]*op, iden)) + tl.kron(iden, tl.kron(iden, weights[2]*op))
    assert_array_almost_equal(H, H_manual)
