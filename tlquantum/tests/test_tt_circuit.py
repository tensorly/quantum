import tensorly as tl
from tensorly.random import random_tt
from tensorly.tt_matrix import TTMatrix
from tensorly.testing import assert_array_almost_equal
from torch import randint
from opt_einsum import contract

from ..tt_circuit import TTCircuit, tt_transpose
from ..density_tensor import DensityTensor
from ..tt_precontraction import qubits_contract
from ..tt_state import tt_norm
from ..tt_gates import cz, BinaryGatesUnitary, UnaryGatesUnitary, exp_pauli_y, CNOTL, CNOTR, CZL, CZR
from ..tt_operators import pauli_z, pauli_x, identity, binary_hamiltonian
from ..tt_contraction import contraction_eq


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>


err_tol = 2 #decimals precision
def manual_rotY_unitary(thetas):
    nqubits, layer = len(thetas), []
    iden, epy = identity(device=thetas.device), exp_pauli_y(thetas.device)
    for i in range(nqubits):
        layer.append(iden*tl.cos(thetas[i]/2)+epy*tl.sin(thetas[i]/2))
    return layer


def test_TTCircuit():
    op, nqubits, nterms = pauli_z(), 4, 4
    nlayers, ncontraq, ncontral = 8, 2, 2
    weights = tl.randn((nterms,))
    spins1, spins2 = randint(nqubits, (nterms,)), randint(nqubits, (nterms,))
    spins2[spins2==spins1] += 1
    spins2[spins2 >= nqubits] = 0
    measurement = binary_hamiltonian(op, nqubits, spins1, spins2, weights)
    dense_measurement = TTMatrix(measurement).to_matrix()
    state = random_tt((2,2,2,2), rank=[1,2,2,2,1])
    dense_state = state.to_tensor().reshape(-1,1)
    state = qubits_contract(state, ncontraq)
    CZ0 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 0)
    CZ1 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 1)
    unitaries = [UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1, UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1]
    circuit = TTCircuit(unitaries, ncontraq, ncontral)
    test_sum = 0
    out = circuit.forward_expectation_value(state, measurement)
    thetas1, thetas2 = tl.tensor([entry.data for entry in circuit.unitaries[0].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[2].parameters()])
    thetas3, thetas4 = tl.tensor([entry.data for entry in circuit.unitaries[4].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[6].parameters()])
    RY1, RY2 = TTMatrix(manual_rotY_unitary(thetas1)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas2)).to_matrix()
    RY3, RY4 = TTMatrix(manual_rotY_unitary(thetas3)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas4)).to_matrix()
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    dense_layer = dense_CZ
    for i in range(nqubits//2 - 2):
        dense_layer = tl.kron(dense_layer, dense_CZ)
    dense_layer1 = tl.kron(dense_layer, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_layer), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_layer), tl.tensor([[0,0],[0,1]]))
    mat1 = tl.dot(tl.dot(dense_layer2, RY2), tl.dot(dense_layer1, RY1))
    mat2 = tl.dot(tl.dot(dense_layer2, RY4), tl.dot(dense_layer1, RY3))
    true_out = tl.dot(tl.dot(mat2, mat1), dense_state)
    true_out = tl.dot(tl.dot(tl.transpose(true_out), dense_measurement), true_out)
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)


def test_forward_for_ptrace():
    op, nqubits, nterms = pauli_z(), 4, 4
    nlayers, ncontraq, ncontral = 8, 2, 2
    weights = tl.randn((nterms,))
    spins1, spins2 = randint(nqubits, (nterms,)), randint(nqubits, (nterms,))
    spins2[spins2==spins1] += 1
    spins2[spins2 >= nqubits] = 0
    measurement = binary_hamiltonian(op, nqubits, spins1, spins2, weights)
    dense_measurement = TTMatrix(measurement).to_matrix()
    state = random_tt((2,2,2,2), rank=[1,2,2,2,1])
    dense_state = state.to_tensor().reshape(-1,1)
    state = qubits_contract(state, ncontraq)
    measurement = qubits_contract(measurement, ncontraq)
    CZ0 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 0)
    CZ1 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 1)
    unitaries = [UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1, UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1]
    circuit = TTCircuit(unitaries, ncontraq, ncontral)
    test_sum = 0
    out = circuit.forward_partial_trace(state, [0]).reshape(4,4)
    thetas1, thetas2 = tl.tensor([entry.data for entry in circuit.unitaries[0].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[2].parameters()])
    thetas3, thetas4 = tl.tensor([entry.data for entry in circuit.unitaries[4].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[6].parameters()])
    RY1, RY2 = TTMatrix(manual_rotY_unitary(thetas1)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas2)).to_matrix()
    RY3, RY4 = TTMatrix(manual_rotY_unitary(thetas3)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas4)).to_matrix()
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    dense_layer = dense_CZ
    for i in range(nqubits//2 - 2):
        dense_layer = tl.kron(dense_layer, dense_CZ)
    dense_layer1 = tl.kron(dense_layer, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_layer), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_layer), tl.tensor([[0,0],[0,1]]))
    mat1 = tl.dot(tl.dot(dense_layer2, RY2), tl.dot(dense_layer1, RY1))
    mat2 = tl.dot(tl.dot(dense_layer2, RY4), tl.dot(dense_layer1, RY3))
    true_out = tl.dot(tl.dot(mat2, mat1), dense_state)
    true_out = tl.dot(true_out, tl.transpose(true_out)).reshape(2,2,2,2,2,2,2,2)
    true_out = DensityTensor(true_out, [[2,2,2,2], [2,2,2,2]]).partial_trace([0,1])[0].reshape(4,4)
    assert_array_almost_equal(out, true_out, decimal=err_tol)


def test_forward_single_qubit():
    op, op2, nqubits, nterms = pauli_z(), pauli_x(), 4, 4
    dense_op, dense_op2, dense_id = TTMatrix([op]).to_matrix(), TTMatrix([op]).to_matrix(), TTMatrix([identity()]).to_matrix()
    nlayers, ncontraq, ncontral = 8, 2, 2
    weights = tl.randn((nterms,))
    spins1, spins2 = randint(nqubits, (nterms,)), randint(nqubits, (nterms,))
    spins2[spins2==spins1] += 1
    spins2[spins2 >= nqubits] = 0
    measurement = binary_hamiltonian(op, nqubits, spins1, spins2, weights)
    dense_measurement = TTMatrix(measurement).to_matrix()
    state = random_tt((2,2,2,2), rank=[1,8,16,8,1])
    state[0] = state[0]/tl.sqrt(tt_norm(state))
    dense_state = state.to_tensor().reshape(-1,1)
    state = qubits_contract(state, ncontraq)
    measurement = qubits_contract(measurement, ncontraq)
    CZ0 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 0)
    CZ1 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 1)
    unitaries = [UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1, UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1]
    circuit = TTCircuit(unitaries, ncontraq, ncontral, max_partial_trace_size=2)
    test_sum = 0
    out, out2 = circuit.forward_single_qubit(state, dense_op, dense_op2)
    thetas1, thetas2 = tl.tensor([entry.data for entry in circuit.unitaries[0].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[2].parameters()])
    thetas3, thetas4 = tl.tensor([entry.data for entry in circuit.unitaries[4].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[6].parameters()])
    RY1, RY2 = TTMatrix(manual_rotY_unitary(thetas1)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas2)).to_matrix()
    RY3, RY4 = TTMatrix(manual_rotY_unitary(thetas3)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas4)).to_matrix()
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    dense_layer = dense_CZ
    for i in range(nqubits//2 - 2):
        dense_layer = tl.kron(dense_layer, dense_CZ)
    dense_layer1 = tl.kron(dense_layer, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_layer), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_layer), tl.tensor([[0,0],[0,1]]))
    mat1 = tl.dot(tl.dot(dense_layer2, RY2), tl.dot(dense_layer1, RY1))
    mat2 = tl.dot(tl.dot(dense_layer2, RY4), tl.dot(dense_layer1, RY3))
    dense_state = tl.dot(tl.dot(mat2, mat1), dense_state)
    true_out = tl.zeros((4,))
    H = tl.kron(dense_op, tl.kron(dense_id, tl.kron(dense_id, dense_id)))
    true_out[0] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    H = tl.kron(dense_id, tl.kron(dense_op, tl.kron(dense_id, dense_id)))
    true_out[1] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    H = tl.kron(dense_id, tl.kron(dense_id, tl.kron(dense_op, dense_id)))
    true_out[2] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    H = tl.kron(dense_id, tl.kron(dense_id, tl.kron(dense_id, dense_op)))
    true_out[3] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    assert_array_almost_equal(out, true_out, decimal=err_tol)
    true_out2 = tl.zeros((4,))
    H = tl.kron(dense_op2, tl.kron(dense_id, tl.kron(dense_id, dense_id)))
    true_out2[0] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    H = tl.kron(dense_id, tl.kron(dense_op2, tl.kron(dense_id, dense_id)))
    true_out2[1] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    H = tl.kron(dense_id, tl.kron(dense_id, tl.kron(dense_op2, dense_id)))
    true_out2[2] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    H = tl.kron(dense_id, tl.kron(dense_id, tl.kron(dense_id, dense_op2)))
    true_out2[3] = tl.dot(tl.transpose(dense_state), tl.dot(H, dense_state))
    assert_array_almost_equal(out2, true_out2, decimal=err_tol)


def test_state_inner_product():
    nqubits, nlayers, ncontraq, ncontral = 4, 8, 2, 2
    state = random_tt((2,2,2,2), rank=[1,4,8,4,1])
    compare_state = random_tt((2,2,2,2), rank=[1,4,8,4,1])
    dense_state = state.to_tensor().reshape(-1,1)
    dense_compare_state = compare_state.to_tensor().reshape(-1,1)
    state = qubits_contract(state, ncontraq)
    compare_state = qubits_contract(compare_state, ncontraq)
    CZ0 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 0)
    CZ1 = BinaryGatesUnitary(nqubits, ncontraq, cz(), 1)
    unitaries = [UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1, UnaryGatesUnitary(nqubits, ncontraq), CZ0, UnaryGatesUnitary(nqubits, ncontraq), CZ1]
    circuit = TTCircuit(unitaries, ncontraq, ncontral)
    test_sum = 0
    out = circuit.state_inner_product(state, compare_state)
    thetas1, thetas2 = tl.tensor([entry.data for entry in circuit.unitaries[0].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[2].parameters()])
    thetas3, thetas4 = tl.tensor([entry.data for entry in circuit.unitaries[4].parameters()]), tl.tensor([entry.data for entry in circuit.unitaries[6].parameters()])
    RY1, RY2 = TTMatrix(manual_rotY_unitary(thetas1)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas2)).to_matrix()
    RY3, RY4 = TTMatrix(manual_rotY_unitary(thetas3)).to_matrix(), TTMatrix(manual_rotY_unitary(thetas4)).to_matrix()
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    dense_layer = dense_CZ
    for i in range(nqubits//2 - 2):
        dense_layer = tl.kron(dense_layer, dense_CZ)
    dense_layer1 = tl.kron(dense_layer, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_layer), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_layer), tl.tensor([[0,0],[0,1]]))
    mat1 = tl.dot(tl.dot(dense_layer2, RY2), tl.dot(dense_layer1, RY1))
    mat2 = tl.dot(tl.dot(dense_layer2, RY4), tl.dot(dense_layer1, RY3))
    dense_state = tl.dot(tl.dot(mat2, mat1), dense_state)
    true_out = tl.dot(tl.transpose(dense_compare_state), dense_state)
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)


def test_tt_transpose():
    nqubits, nlayers, ncontraq = 8, 3, 1
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    dense_state = state.to_tensor().reshape(-1,1)

    thetas = tl.randn((nqubits, 1))
    CZ_layers = [BinaryGatesUnitary(nqubits, ncontraq, cz(), 0).forward(), BinaryGatesUnitary(nqubits, ncontraq, cz(), 1).forward()]
    dense_CZ = tl.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    dense_layer = dense_CZ
    for i in range(nqubits//2 - 2):
        dense_layer = tl.kron(dense_layer, dense_CZ)
    dense_layer1 = tl.kron(dense_layer, dense_CZ)
    dense_layer2 = tl.kron(tl.kron(tl.tensor([[1,0],[0,1]]), dense_layer), tl.tensor([[1,0],[0,0]])) + tl.kron(tl.kron(tl.tensor([[1,0],[0,-1]]), dense_layer), tl.tensor([[0,0],[0,1]]))
    RY = manual_rotY_unitary(thetas)
    matT = tl.transpose(tl.dot(dense_layer2, tl.dot(dense_layer1, TTMatrix(RY).to_matrix())))
    true_out = tl.dot(tl.transpose(dense_state), tl.dot(matT, dense_state))
    RY_T = []
    for factor in RY:
        RY_T.append(tt_transpose(factor))
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *CZ_layers[1], *CZ_layers[0], *RY_T, *state)
    assert_array_almost_equal(out, true_out[0], decimal=err_tol)
