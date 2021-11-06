import tensorly as tl
from opt_einsum import contract


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


def spins_to_tt_state(spins, device='cpu'):
    """Generates tt-tensor state of computational basis product space described by spins.

    Parameters
    ----------
    spins : List/tensor of ints (0's and 1's), spin values

    Returns
    -------
    tt-tensor product state in computational basis
    """
    state = list(map(_spin_to_qubit, spins[1:-1]))
    if spins[0] == 0:
        #Single spin 0 core in tt-tensor format at initial position
        state = [tl.tensor([[[1., 0.], [0., 1.]]])] + state
    else:
        #Single spin 1 core in tt-tensor format at initial position
        state = [tl.tensor([[[0., 1.], [1., 0.]]])] + state
    if spins[-1] == 0:
        #Single spin 0 core in tt-tensor format at final position
        state = state + [tl.tensor([[[1.], [0.]], [[0.], [0.]]])]
    else:
        #Single spin 1 core in tt-tensor format at final position
        state = state + [tl.tensor([[[0.], [1.]], [[0.], [0.]]])]
    if device=='cpu':
        return state
    else:
        return [core.to(device) for core in state]


def _spin_to_qubit(spin):
    """Matches computational basis spin (0 or 1) to corresponding tt-tensor core.

    Parameters
    ----------
    spin : int (0 or 1), spin value

    Returns
    -------
    tt-tensor core
    """
    if spin == 0:
        #Single spin 0 core in tt-tensor format at central position
        return tl.tensor([[[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]])
    else:
        #Single spin 1 core in tt-tensor format at central position
        return tl.tensor([[[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]])


def tt_norm(tensor):
    """The norm of a TT-tensor state.

    Parameters
    ----------
    state : tt-tensor, input state to be evolved by unitary
    
    Returns
    -------
    float, modulo squared of tensor
    """
    ncores, eq1, eq2, start = len(tensor.shape), [], [], ord('a')
    for n in range(ncores):
        idx = [start+2*n, start+1+2*n, start+2+2*n]
        eq1.append(''.join(chr(j) for j in idx))
        if n != 0:
            idx[0] += 2*ncores
        if n != ncores-1:
            idx[2] += 2*ncores
        eq2.append(''.join(chr(j) for j in idx))
    return tl.sqrt(contract(','.join(i for i in eq1+eq2), *tensor, *tensor))
