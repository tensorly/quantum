import tensorly as tl
from tensorly.tt_matrix import TTMatrix
from tensorly.random import random_tt
from tensorly.testing import assert_array_almost_equal
from torch import cos, sin, complex64, float32, exp, randn, matrix_exp
from opt_einsum import contract

from ..tt_gates import exp_pauli_y, UnaryGatesUnitary, RotY, GPI2, cnot, cz, so4, o4_phases, ms, BinaryGatesUnitary, InvolutoryGeneratorUnitary
from ..tt_operators import identity, pauli_y, pauli_x
from ..tt_contraction import contraction_eq
from ..tt_sum import tt_matrix_sum


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


err_tol = 4 #decimals precision
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


def test_GPI2():
    gpi2 = GPI2()
    phi = gpi2.phi
    true_gpi2 = tl.tensor([[1, -1j*exp(-1j*phi)], [-1j*exp(1j*phi), 1]], dtype=complex64)
    assert_array_almost_equal(TTMatrix([gpi2.forward()]).to_matrix(), true_gpi2)


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
    nqubits, ncontraq, dtype = 8, 2, float32
    unitary = UnaryGatesUnitary(nqubits, ncontraq, dtype=dtype)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[1,0],[0,1]])*tl.cos(thetas[0]/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(thetas[0]/2)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[1,0],[0,1]])*tl.cos(theta/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(theta/2))
    assert_array_almost_equal(layer, dense_layer)

    nqubits = 9
    unitary = UnaryGatesUnitary(nqubits, ncontraq, dtype=dtype)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[1,0],[0,1]])*tl.cos(thetas[0]/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(thetas[0]/2)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[1,0],[0,1]])*tl.cos(theta/2) + tl.tensor([[0, -1],[1, 0]])*tl.sin(theta/2))
    assert_array_almost_equal(layer, dense_layer)


def test_InvolutoryGeneratorUnitary():
    nqubits, ncontraq, dtype = 4, 2, complex64
    iden, py = identity().reshape(2,2), pauli_y().reshape(2,2)
    unitary = []
    id_list = [0,2]
    new_theta = randn(1)
    for ind in range(nqubits):
        if ind in id_list:
            unitary.append(identity())
        else:
            unitary.append(pauli_y())
    unitary = InvolutoryGeneratorUnitary(nqubits, ncontraq, unitary)
    for param in unitary.parameters():
        param.data = new_theta
    layer = TTMatrix(unitary.forward()).to_matrix()
    generator = tl.kron(tl.kron(iden, py), tl.kron(iden, py))
    dense_layer = tl.eye(2**nqubits)*cos(new_theta) + 1j*generator*sin(new_theta)
    assert_array_almost_equal(layer, dense_layer)

    nqubits, ncontraq, dtype = 5, 2, complex64
    iden, px = identity().reshape(2,2), pauli_x().reshape(2,2)
    unitary = []
    id_list = [0,1,4]
    new_theta = randn(1)
    for ind in range(nqubits):
        if ind in id_list:
            unitary.append(identity())
        else:
            unitary.append(pauli_x())
    unitary = InvolutoryGeneratorUnitary(nqubits, ncontraq, unitary)
    for param in unitary.parameters():
        param.data = new_theta
    layer = TTMatrix(unitary.forward()).to_matrix()
    generator = tl.kron(tl.kron(tl.kron(iden, iden), tl.kron(px, px)), iden)
    dense_layer = tl.eye(2**nqubits)*cos(new_theta) + 1j*generator*sin(new_theta)
    assert_array_almost_equal(layer, dense_layer)

    ### New for Grace. We exponentiate the matrix the old fashioned way
    dense_layer = matrix_exp(1j*new_theta*generator) # ground truth using general matrix exponentiation
    assert_array_almost_equal(layer, dense_layer, 4)


def test_RotXUnitary():
    nqubits, ncontraq, dtype = 8, 2, complex64
    unitary = UnaryGatesUnitary(nqubits, ncontraq, axis='x', dtype=dtype)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[1,0],[0,1]])*tl.cos(thetas[0]/2) + tl.tensor([[0, -1j],[-1j, 0]], dtype=dtype)*tl.sin(thetas[0]/2)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[1,0],[0,1]])*tl.cos(theta/2) + tl.tensor([[0, -1j],[-1j, 0]], dtype=dtype)*tl.sin(theta/2))
    assert_array_almost_equal(layer, dense_layer)

    nqubits, contraq = 9, 2
    unitary = UnaryGatesUnitary(nqubits, ncontraq, axis='x', dtype=dtype)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[1,0],[0,1]])*tl.cos(thetas[0]/2) + tl.tensor([[0, -1j],[-1j, 0]], dtype=dtype)*tl.sin(thetas[0]/2)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[1,0],[0,1]])*tl.cos(theta/2) + tl.tensor([[0, -1j],[-1j, 0]], dtype=dtype)*tl.sin(theta/2))
    assert_array_almost_equal(layer, dense_layer)


def test_RotZUnitary():
    nqubits, ncontraq, dtype = 8, 2, complex64
    unitary = UnaryGatesUnitary(nqubits, ncontraq, axis='z', dtype=dtype)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[exp(-1j*thetas[0]/2),0],[0,exp(1j*thetas[0]/2)]], dtype=dtype)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[exp(-1j*theta/2),0],[0,exp(1j*theta/2)]], dtype=dtype))
    assert_array_almost_equal(layer, dense_layer)

    nqubits, contraq = 9, 2
    unitary = UnaryGatesUnitary(nqubits, ncontraq, axis='z', dtype=dtype)
    thetas = tl.tensor([theta.data for theta in unitary.parameters()])
    layer = TTMatrix(unitary.forward()).to_matrix()
    dense_layer = tl.tensor([[exp(-1j*thetas[0]/2),0],[0,exp(1j*thetas[0]/2)]], dtype=dtype)
    for theta in thetas[1::]:
        dense_layer = tl.kron(dense_layer, tl.tensor([[exp(-1j*theta/2),0],[0,exp(1j*theta/2)]], dtype=dtype))
    assert_array_almost_equal(layer, dense_layer)


def test_q2_gate_layers():
    nqubits, nlayers, dtype = 4, 2, float32
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank, dtype=dtype)
    dense_state = state.to_tensor().reshape(-1,1)

    layers = [BinaryGatesUnitary(nqubits, 1, cz(dtype=dtype), 0).forward(), BinaryGatesUnitary(nqubits, 1, cz(dtype=dtype), 1).forward()]
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *layers[0], *layers[1], *state)
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=dtype)
    dense_layer1 = tl.kron(dense_CZ, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_CZ), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_CZ), tl.tensor([[0,0],[0,1]]))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(dense_layer2, tl.dot(dense_layer1, dense_state)))
    assert_array_almost_equal(out, true_out[0], decimal=5)

    layers = [BinaryGatesUnitary(nqubits, 1, cnot(dtype=dtype), 0).forward(), BinaryGatesUnitary(nqubits, 1, cnot(dtype=dtype), 1).forward()]
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *layers[0], *layers[1], *state)
    dense_CNOT = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=dtype)
    dense_layer1 = tl.kron(dense_CNOT, dense_CNOT)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_CNOT), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[0,1],[1,0]]), dense_CNOT), tl.tensor([[0,0],[0,1]]))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(dense_layer2, tl.dot(dense_layer1, dense_state)))
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)

    nqubits = 5
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank, dtype=dtype)
    dense_state = state.to_tensor().reshape(-1,1)

    layers = [BinaryGatesUnitary(nqubits, 1, cnot(dtype=dtype), 0).forward(), BinaryGatesUnitary(nqubits, 1, cnot(dtype=dtype), 1).forward()]
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *layers[0], *layers[1], *state)
    dense_CNOT = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=dtype)
    dense_layer1 = tl.kron(tl.kron(dense_CNOT, dense_CNOT), tl.tensor([[1,0],[0,1]]))
    dense_layer2 = tl.kron(tl.tensor([[1,0],[0,1]]), tl.kron(dense_CNOT, dense_CNOT))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(dense_layer2, tl.dot(dense_layer1, dense_state)))
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)


def test_so4():
    so4_01, dtype = so4(1, 0), complex64
    theta01 = so4_01[0].theta
    true_so4_01 = tl.tensor([[1,0,0,0], [0,1,0,0], [0,0,cos(theta01),-sin(theta01)], [0,0,sin(theta01),cos(theta01)]], dtype=dtype)
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
    unitary0, unitary1 = BinaryGatesUnitary(nqubits, ncontraq, so4_01, 0, random_initialization=False).forward(), BinaryGatesUnitary(nqubits, ncontraq, so4_01, 1, random_initialization=False).forward()
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank, dtype=dtype)
    dense_state = state.to_tensor().reshape(-1,1)
    true_unitary0 = tl.kron(true_so4_01, true_so4_01)
    true_unitary1 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), true_so4_01), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[cos(theta01),-sin(theta01)], [sin(theta01),cos(theta01)]]), true_so4_01), tl.tensor([[0,0],[0,1]]))
    eq = contraction_eq(nqubits, nlayers)
    inner_prod = contract(eq, *state, *unitary1, *unitary0, *state)
    true_inner_prod = tl.dot(tl.transpose(dense_state), tl.dot(true_unitary0, tl.dot(true_unitary1, dense_state)))
    assert_array_almost_equal(inner_prod, true_inner_prod[0], decimal=err_tol)

    nqubits, nlayers = 5, 2
    unitary0, unitary1 = BinaryGatesUnitary(nqubits, ncontraq, so4_01, 0, random_initialization=False).forward(), BinaryGatesUnitary(nqubits, ncontraq, so4_01, 1, random_initialization=False).forward()
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank, dtype=dtype)
    dense_state = state.to_tensor().reshape(-1,1)
    true_unitary0 = tl.kron(true_so4_01, tl.kron(true_so4_01, tl.eye(2)))
    true_unitary1 = tl.kron(tl.eye(2), tl.kron(true_so4_01, true_so4_01))
    eq = contraction_eq(nqubits, nlayers)
    inner_prod = contract(eq, *state, *unitary1, *unitary0, *state)
    true_inner_prod = tl.dot(tl.transpose(dense_state), tl.dot(true_unitary0, tl.dot(true_unitary1, dense_state)))
    assert_array_almost_equal(inner_prod, true_inner_prod[0], decimal=err_tol)


def test_o4_phases():
    dtype = complex64
    o4_phases_instance = o4_phases()
    phases = o4_phases_instance[0].phases
    true_o4_phases = tl.tensor([[exp(1j*phases[0]),0,0,0], [0,exp(1j*phases[1]),0,0], [0,0,exp(1j*phases[2]),0], [0,0,0,exp(1j*phases[3])]], dtype=dtype)
    assert_array_almost_equal(TTMatrix([o4_phases_instance[0].forward(), o4_phases_instance[1].forward()]).to_matrix(), true_o4_phases, decimal=err_tol)

    nqubits, nlayers, ncontraq = 4, 2, 1
    unitary0, unitary1 = BinaryGatesUnitary(nqubits, ncontraq, o4_phases_instance, 0, random_initialization=False).forward(), BinaryGatesUnitary(nqubits, ncontraq, o4_phases_instance, 1, random_initialization=False).forward()
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank, dtype=dtype)
    dense_state = state.to_tensor().reshape(-1,1)
    true_unitary0 = tl.kron(true_o4_phases, true_o4_phases)
    true_unitary1 = tl.kron(tl.kron(tl.tensor([[1,0],[0,0]], dtype=complex64), true_o4_phases), tl.tensor([[exp(1j*phases[0]),0],[0,0]], dtype=dtype))
    true_unitary1 += tl.kron(tl.kron(tl.tensor([[0,0],[0,1]], dtype=complex64), true_o4_phases), tl.tensor([[exp(1j*phases[1]),0],[0,0]], dtype=dtype))
    true_unitary1 += tl.kron(tl.kron(tl.tensor([[1,0],[0,0]], dtype=complex64), true_o4_phases), tl.tensor([[0,0],[0,exp(1j*phases[2])]], dtype=dtype))
    true_unitary1 += tl.kron(tl.kron(tl.tensor([[0,0],[0,1]], dtype=complex64), true_o4_phases), tl.tensor([[0,0],[0,exp(1j*phases[3])]], dtype=dtype))
    eq = contraction_eq(nqubits, nlayers)
    inner_prod = contract(eq, *state, *unitary1, *unitary0, *state)
    true_inner_prod = tl.dot(tl.transpose(dense_state), tl.dot(true_unitary0, tl.dot(true_unitary1, dense_state)))
    assert_array_almost_equal(inner_prod, true_inner_prod[0], decimal=err_tol)

    nqubits = 5
    unitary0, unitary1 = BinaryGatesUnitary(nqubits, ncontraq, o4_phases_instance, 0, random_initialization=False).forward(), BinaryGatesUnitary(nqubits, ncontraq, o4_phases_instance, 1, random_initialization=False).forward()
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank, dtype=dtype)
    dense_state = state.to_tensor().reshape(-1,1)
    true_unitary0 = tl.kron(true_o4_phases, tl.kron(true_o4_phases, tl.eye(2, dtype=dtype)))
    true_unitary1 = tl.kron(tl.eye(2), tl.kron(true_o4_phases, true_o4_phases))
    eq = contraction_eq(nqubits, nlayers)
    inner_prod = contract(eq, *state, *unitary1, *unitary0, *state)
    true_inner_prod = tl.dot(tl.transpose(dense_state), tl.dot(true_unitary0, tl.dot(true_unitary1, dense_state)))
    assert_array_almost_equal(inner_prod, true_inner_prod[0], decimal=err_tol)


def test_ms_gate():
    ms_inst, dtype = ms(), complex64
    theta, phi0, phi1 = ms_inst[1].theta, ms_inst[1].phi0, ms_inst[1].phi1
    true_ms = tl.tensor([[cos(theta), 0, 0, sin(theta) * -1j * exp(-1j*(phi0+phi1))],
                        [0, cos(theta), sin(theta) * -1j * exp(-1j*(phi0-phi1)), 0],
                        [0, sin(theta) * -1j * exp(1j * (phi0-phi1)), cos(theta), 0],
                        [sin(theta) * -1j * exp(1j * (phi0+phi1)), 0, 0, cos(theta)]], dtype=dtype)
    assert_array_almost_equal(TTMatrix([ms_inst[0].forward(), ms_inst[1].forward()]).to_matrix(), true_ms, decimal=err_tol)
