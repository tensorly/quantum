import tensorly as tl
from tensorly.random import random_tt
from tensorly.tt_matrix import TTMatrix
from tensorly.testing import assert_array_almost_equal
from opt_einsum import contract

from ..tt_contraction import contraction_eq
from ..tt_state import tt_norm
from ..tt_gates import exp_pauli_y, RotY, cnot
from ..tt_operators import identity


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


def test_contraction_eq():
    nqubits = 4
    nlayers = 1
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [2 for i in range(nqubits-1)] + [1]
    state = random_tt(dims, rank=rank)
    state[0] = state[0]/tl.sqrt(tt_norm(state))

    rotY1, rotY2 = RotY(), RotY()
    thetas = tl.tensor([rotY1.theta, rotY2.theta])
    iden = TTMatrix([identity()]).to_matrix()
    true_CNOT = tl.kron(tl.kron(iden, TTMatrix([rotY1.forward()]).to_matrix()), tl.kron(TTMatrix([rotY2.forward()]).to_matrix(), tl.tensor([[1,0],[0,0]])))
    true_CNOT += tl.kron(tl.kron(tl.tensor([[0,1],[1,0]]), TTMatrix([rotY1.forward()]).to_matrix()), tl.kron(TTMatrix([rotY2.forward()]).to_matrix(), tl.tensor([[0,0],[0,1]])))
    layer = [cnot()[1].forward(), rotY1.forward(), rotY2.forward(), cnot()[0].forward()]
    eq = contraction_eq(nqubits, nlayers)
    out = contract(eq, *state, *layer, *state)
    state = state.to_tensor().reshape(-1,1)
    true_out = tl.dot(tl.transpose(state), tl.dot(true_CNOT, state))
    assert_array_almost_equal(out, true_out[0])
