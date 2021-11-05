import tensorly as tl
from tensorly.tt_matrix import TTMatrix
from tensorly.random import random_tt
from tensorly.testing import assert_array_almost_equal
from torch import cos, sin
from opt_einsum import contract

from ..tt_gates import exp_pauli_y, UnaryGatesUnitary, RotY, cnot, cz, so4, BinaryGatesUnitary
from ..tt_operators import identity
from ..tt_contraction import contraction_eq


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


err_tol = 5 #decimals precision
def manual_rotY_unitary(thetas):
    nqubits, layer = len(thetas), []
    iden, epy = IDENTITY.to(thetas.device), EXP_PAULI_Y.to(thetas.device)
    for i in range(nqubits):
        layer.append(iden*tl.cos(thetas[i]/2)+epy*tl.sin(thetas[i]/2))
    return layer


def test_EXP_PAULI_Y():
    exp_pauli_y_temp = TTMatrix([exp_pauli_y()])
    assert_array_almost_equal(exp_pauli_y_temp.to_matrix(), tl.tensor([[0., -1],[1, 0]]))


def test_RotY():
    rotY = RotY()
    RotY_temp = TTMatrix([rotY.forward()])
    theta = tl.tensor([rotY.theta])
    RotY_dense = tl.tensor([[1,0],[0,1]])*tl.cos(theta/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(theta/2)
    assert_array_almost_equal(RotY_temp.to_matrix(), RotY_dense)


def test_CNOT():
    CNOT_temp = TTMatrix([cnot()[0].forward(), cnot()[1].forward()])
    CNOT_temp = CNOT_temp.to_matrix()
    dense_CNOT = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    assert_array_almost_equal(CNOT_temp, dense_CNOT)


def test_cz_tt():
    CZ_temp = TTMatrix([cz()[0].forward(), cz()[1].forward()])
    CZ_temp = CZ_temp.to_matrix()
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    assert_array_almost_equal(CZ_temp, dense_CZ)


def test_RotYUnitary():
    nqubits, ncontraq = 8, 2
    unitary = UnaryGatesUnitary(nqubits, ncontraq)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[1,0],[0,1]])*tl.cos(thetas[0]/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(thetas[0]/2)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[1,0],[0,1]])*tl.cos(theta/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(theta/2))
    assert_array_almost_equal(layer, dense_layer)

    nqubits, contraq = 9, 2
    unitary = UnaryGatesUnitary(nqubits, ncontraq)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[1,0],[0,1]])*tl.cos(thetas[0]/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(thetas[0]/2)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[1,0],[0,1]])*tl.cos(theta/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(theta/2))
    assert_array_almost_equal(layer, dense_layer)


def test_q2_gate_layers():
    nqubits = 4
    nlayers = 2
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    dense_state = state.to_tensor().reshape(-1,1)

    layers = [BinaryGatesUnitary(nqubits, 1, cz(), 0).forward(), BinaryGatesUnitary(nqubits, 1, cz(), 1).forward()]
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *layers[0], *layers[1], *state)
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    dense_layer1 = tl.kron(dense_CZ, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_CZ), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_CZ), tl.tensor([[0,0],[0,1]]))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(dense_layer2, tl.dot(dense_layer1, dense_state)))
    assert_array_almost_equal(out, true_out[0], decimal=5)

    layers = [BinaryGatesUnitary(nqubits, 1, cnot(), 0).forward(), BinaryGatesUnitary(nqubits, 1, cnot(), 1).forward()]
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *layers[0], *layers[1], *state)
    dense_CNOT = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    dense_layer1 = tl.kron(dense_CNOT, dense_CNOT)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_CNOT), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[0,1],[1,0]]), dense_CNOT), tl.tensor([[0,0],[0,1]]))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(dense_layer2, tl.dot(dense_layer1, dense_state)))
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)

    nqubits = 5
    nlayers = 2
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    dense_state = state.to_tensor().reshape(-1,1)

    layers = [BinaryGatesUnitary(nqubits, 1, cnot(), 0).forward(), BinaryGatesUnitary(nqubits, 1, cnot(), 1).forward()]
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *layers[0], *layers[1], *state)
    dense_CNOT = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    dense_layer1 = tl.kron(tl.kron(dense_CNOT, dense_CNOT), tl.tensor([[1,0],[0,1]]))
    dense_layer2 = tl.kron(tl.tensor([[1,0],[0,1]]), tl.kron(dense_CNOT, dense_CNOT))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(dense_layer2, tl.dot(dense_layer1, dense_state)))
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)


def test_so4():
    so4_01 = so4(1, 0)
    theta01 = so4_01[0].theta
    true_so4_01 = tl.tensor([[1,0,0,0], [0,1,0,0], [0,0,cos(theta01),-sin(theta01)], [0,0,sin(theta01),cos(theta01)]])
    assert_array_almost_equal(TTMatrix([so4_01[0].forward(), so4_01[1].forward()]).to_matrix(), true_so4_01, decimal=err_tol)

    so4_12 = so4(1, 2)
    theta12 = so4_12[0].theta
    true_so4_12 = tl.tensor([[1,0,0,0], [0,cos(theta12),-sin(theta12),0], [0,sin(theta12),cos(theta12),0],[0,0,0,1]])
    assert_array_almost_equal(TTMatrix([so4_12[0].forward(), so4_12[1].forward()]).to_matrix(), true_so4_12, decimal=err_tol)

    so4_23 = so4(2, 3)
    theta23 = so4_23[0].theta
    true_so4_23 = tl.tensor([[cos(theta23),-sin(theta23),0,0], [sin(theta23),cos(theta23),0,0], [0,0,1,0], [0,0,0,1]])
    assert_array_almost_equal(TTMatrix([so4_23[0].forward(), so4_23[1].forward()]).to_matrix(), true_so4_23, decimal=err_tol)

    nqubits, nlayers, ncontraq = 4, 2, 1
    unitary0, unitary1 = BinaryGatesUnitary(nqubits, ncontraq, so4_01, 0).forward(), BinaryGatesUnitary(nqubits, ncontraq, so4_01, 1).forward()
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    dense_state = state.to_tensor().reshape(-1,1)
    true_unitary0 = tl.kron(true_so4_01, true_so4_01)
    true_unitary1 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), true_so4_01), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[cos(theta01),-sin(theta01)], [sin(theta01),cos(theta01)]]), true_so4_01), tl.tensor([[0,0],[0,1]]))
    eq = contraction_eq(nqubits, nlayers)
    inner_prod = contract(eq, *state, *unitary1, *unitary0, *state)
    true_inner_prod = tl.dot(tl.transpose(dense_state), tl.dot(true_unitary0, tl.dot(true_unitary1, dense_state)))
    assert_array_almost_equal(inner_prod, true_inner_prod[0], decimal=err_tol)

    nqubits = 5
    unitary0, unitary1 = BinaryGatesUnitary(nqubits, ncontraq, so4_01, 0).forward(), BinaryGatesUnitary(nqubits, ncontraq, so4_01, 1).forward()
    nlayers = 2
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    dense_state = state.to_tensor().reshape(-1,1)
    true_unitary0 = tl.kron(true_so4_01, tl.kron(true_so4_01, tl.eye(2)))
    true_unitary1 = tl.kron(tl.eye(2), tl.kron(true_so4_01, true_so4_01))
    eq = contraction_eq(nqubits, nlayers)
    inner_prod = contract(eq, *state, *unitary1, *unitary0, *state)
    true_inner_prod = tl.dot(tl.transpose(dense_state), tl.dot(true_unitary0, tl.dot(true_unitary1, dense_state)))
    assert_array_almost_equal(inner_prod, true_inner_prod[0], decimal=err_tol)
