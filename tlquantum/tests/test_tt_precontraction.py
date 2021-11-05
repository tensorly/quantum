import tensorly as tl
from tensorly.random import random_tt
from tensorly.tt_tensor import TTTensor
from tensorly.tt_matrix import TTMatrix
from tensorly.testing import assert_array_almost_equal
from numpy import ceil
from opt_einsum import contract

from ..tt_precontraction import qubits_contract, layers_contract
from ..tt_gates import UnaryGatesUnitary, BinaryGatesUnitary, cz, exp_pauli_y
from ..tt_operators import identity
from ..tt_contraction import contraction_eq


err_tol = 4


def test_qubits_contract():
    nqubits, ncontraq, nlayers = 8, 5, 3
    nqsubsets = int(ceil(nqubits/ncontraq))
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    dense_state = state.to_tensor().reshape(-1,1)
    contr_state = qubits_contract([*state], ncontraq)

    assert len(contr_state) == nqsubsets
    assert_array_almost_equal(TTTensor(contr_state).to_tensor().reshape(-1,1), dense_state)

    CZ_layer0 = BinaryGatesUnitary(nqubits, 1, cz(), 0).forward()
    CZ_layer1 = BinaryGatesUnitary(nqubits, 1, cz(), 1).forward()
    CZ_layers = [qubits_contract(CZ_layer0, ncontraq), qubits_contract(CZ_layer1, ncontraq)]
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    dense_layer = dense_CZ
    for i in range(nqubits//2 - 2):
        dense_layer = tl.kron(dense_layer, dense_CZ)
    dense_layer1 = tl.kron(dense_layer, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_layer), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_layer), tl.tensor([[0,0],[0,1]]))
    RYU = UnaryGatesUnitary(nqubits, ncontraq)
    RY = RYU.forward()
    thetas = tl.tensor([theta.data for theta in RYU.parameters()])
    dense_RY = TTMatrix(RY).to_matrix()
    eq = contraction_eq(nqsubsets, nlayers)
    out = contract(eq, *contr_state, *RY, *CZ_layers[0], *CZ_layers[1], *contr_state)
    mat = tl.dot(tl.dot(dense_layer2, dense_layer1), dense_RY)
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(mat, dense_state))
    assert (len(CZ_layers[0]) == nqsubsets) and (len(CZ_layers[1]) == nqsubsets) and (len(RY) == nqsubsets)
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)


def test_layers_contract():
    nqubits, ncontral, nlayers = 4, 5, 10
    nlsubsets = int(ceil(nlayers/ncontral))
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    dense_state = state.to_tensor().reshape(-1,1)

    def manual_rotY_unitary(thetas):
        nqubits, layer = len(thetas), []
        iden, epy = identity(thetas.device), exp_pauli_y(thetas.device)
        for i in range(nqubits):
            layer.append(iden*tl.cos(thetas[i]/2)+epy*tl.sin(thetas[i]/2))
        return layer

    thetas1, thetas2, thetas3 = tl.randn((nqubits, 1)), tl.randn((nqubits, 1)), tl.randn((nqubits, 1))
    RY1, RY2, RY3 = manual_rotY_unitary(thetas1), manual_rotY_unitary(thetas2), manual_rotY_unitary(thetas3)
    CZ_layer0 = BinaryGatesUnitary(nqubits, 1, cz(), 0).forward()
    CZ_layer1 = BinaryGatesUnitary(nqubits, 1, cz(), 1).forward()
    RY1_dense, RY2_dense, RY3_dense = TTMatrix(RY1).to_matrix(), TTMatrix(RY2).to_matrix(), TTMatrix(RY3).to_matrix()
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    layers = layers_contract([RY1, CZ_layer0, CZ_layer1, RY2, CZ_layer0, CZ_layer1, RY3, CZ_layer0, CZ_layer1, RY1], ncontral)

    dense_layer = dense_CZ
    for i in range(nqubits//2 - 2):
        dense_layer = tl.kron(dense_layer, dense_CZ)
    dense_layer1 = tl.kron(dense_layer, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_layer), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_layer), tl.tensor([[0,0],[0,1]]))

    eq = contraction_eq(nqubits, nlsubsets)
    out = contract(eq, *state, *layers, *state)

    mat1 = tl.dot(dense_layer2, tl.dot(dense_layer1, RY1_dense))
    mat2 = tl.dot(dense_layer2, tl.dot(dense_layer1, RY2_dense))
    mat3 = tl.dot(tl.dot(RY1_dense, dense_layer2), tl.dot(dense_layer1, RY3_dense))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(mat3, tl.dot(mat2, tl.dot(mat1, dense_state))))

    assert len(layers) == nqubits*nlsubsets
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)
