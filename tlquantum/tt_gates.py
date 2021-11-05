import tensorly as tl
tl.set_backend('pytorch')
from torch import randn, cos, sin
from torch.nn import Module, ModuleList, ParameterList, Parameter
from tensorly.tt_matrix import TTMatrix

from .tt_operators import identity
from .tt_precontraction import qubits_contract, _get_contrsets
from .tt_sum import tt_matrix_sum


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>
# Author: Jean Kossaifi <jkossaifi@nvidia.com>

# License: BSD 3 clause


class Unitary(Module):
    """A unitary for all qubits in a TTCircuit, using tensor ring tensors
    with PyTorch Autograd support.
    Can be defined with arbitrary gates or used as a base-class for set circuit
    types.

    Parameters
    ----------
    gates : list of TT gate classes, each qubit in the unitary
            to be involved in one gate.
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    Unitary
    """
    def __init__(self, gates, nqubits, ncontraq, contrsets=None, device=None):
        super().__init__()
        if contrsets is None:
            contrsets = _get_contrsets(nqubits, ncontraq)
        self.nqubits, self.ncontraq, self.contrsets, self.device = nqubits, ncontraq, contrsets, device
        self._set_gates(gates)


    def _set_gates(self, gates):
        """Sets the gate class instances as a PyTorch ModuleList for Unitary.

        """
        self.gates = ModuleList(gates)


    def forward(self):
        """Prepares the tensors of Unitary for forward contraction by calling the gate instances'
        forward method and doing qubit-wise (horizonal) pre-contraction.

        Returns
        -------
        List of pre-contracted gate tensors for general forward pass.
        """
        return qubits_contract([gate.forward() for gate in self.gates], self.ncontraq, contrsets=self.contrsets)


class BinaryGatesUnitary(Unitary):
    """A Unitary sub-class that generates a layer of a single two-qubit gates accross
    all qubits in a TTCircuit.

    Parameters
    ----------
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    q2gate : tuple of two gate instances, one for each qubit in gate.
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    BinaryGatesUnitary
    """
    def __init__(self, nqubits, ncontraq, q2gate, parity, contrsets=None):
        device = q2gate[0].device
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, device=device)
        self._set_gates(build_binary_gates_unitary(self.nqubits, q2gate, parity))


class UnaryGatesUnitary(Unitary):
    """A Unitary sub-class that generates a layer of unitary, single-qubit rotations.
    As simulation occurs in real-space, these rotations are about the Y-axis.

    Parameters
    ----------
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    UnaryGatesUnitary
    """
    def __init__(self, nqubits, ncontraq, contrsets=None, device=None):
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, device=device)
        self._set_gates([RotY(device=device) for i in range(self.nqubits)])


def build_binary_gates_unitary(nqubits, q2gate, parity):
    """Generate a layer of two-qubit gates.

    Parameters
    ----------
    nqubits : int, number of qubits
    q2gate : tt-tensor, 2-core, 2-qubit gates to use in layer
    parity : int, if even, apply first q2gate core to even qubits, if odd, to odd qubits.

    Returns
    -------
    Layer of two-qubit gates as list of tt-tensors
    """
    q2gate0, q2gate1 = q2gate
    layer, device = [], q2gate0.device
    for i in range(nqubits//2 - 1):
        layer += [q2gate0, q2gate1]
    if nqubits%2 == 0:
        if parity%2 == 0:
            return layer+[q2gate0, q2gate1]
        return [q2gate1]+layer+[q2gate0]
    if parity%2 == 0:
        return layer+[q2gate0, q2gate1, IDENTITY(device)]
    return [IDENTITY(device)]+layer+[q2gate0, q2gate1]


class RotY(Module):
    """Qubit rotations about the Y-axis with randomly initiated theta.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    RotY
    """
    def __init__(self, device=None):
        super().__init__()
        self.theta = Parameter(randn(1, device=device))
        self.iden, self.epy = identity(self.theta.device), exp_pauli_y(self.theta.device)


    def forward(self):
        """Prepares the RotY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*cos(self.theta/2)+self.epy*sin(self.theta/2)


class IDENTITY(Module):
    """Identity gate (does not change the state of the qubit on which it acts).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    IDENTITY
    """
    def __init__(self, device=None):
        super().__init__()
        self.core, self.device = identity(device=device), device


    def forward(self):
        """Prepares the left qubit of the IDENTITY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


def cnot(device=None):
    """Pair of CNOT class instances, one left (control) and one right (transformed).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    (CNOTL, CNOTR)
    """
    return CNOTL(device=device), CNOTR(device=device)


class CNOTL(Module):
    """Left (control-qubit) core of a CNOT gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Left core of CNOT gate.
    """
    def __init__(self, device=None):
        super().__init__()
        core, self.device = tl.zeros((1,2,2,2), device=device), device
        core[0,0,0,0] = core[0,1,1,1] = 1.
        self.core = core


    def forward(self):
        """Prepares the left qubit of the CNOT gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class CNOTR(Module):
    """Right (transformed qubit) core of a CNOT gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Right core of CNOT gate.
    """
    def __init__(self, device=None):
        super().__init__()
        core, self.device = tl.zeros((2,2,2,1), device=device), device
        core[0,0,0,0] = core[0,1,1,0] = 1.
        core[1,0,1,0] = core[1,1,0,0] = 1.
        self.core =  core


    def forward(self):
        """Prepares the right qubit of the CNOT gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


def cz(device=None):
    """Pair of CZ class instances, one left (control) and one right (transformed).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    (CZL, CZR)
    """
    return CZL(device=device), CZR(device=device)


class CZL(Module):
    """Left (control-qubit) core of a CZ gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Left core of CZ gate.
    """
    def __init__(self, device=None):
        super().__init__()
        core, self.device = tl.zeros((1,2,2,2), device=device), device
        core[0,0,0,0] = core[0,1,1,1] = 1.
        self.core = core


    def forward(self):
        """Prepares the left qubit of the CZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class CZR(Module):
    """Right (transformed qubit) core of a CZ gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Right core of CZ gate.
    """
    def __init__(self, device=None):
        super().__init__()
        core, self.device = tl.zeros((2,2,2,1), device=device), device
        core[0,0,0,0] = core[0,1,1,0] = core[1,0,0,0]  = 1.
        core[1,1,1,0] = -1.
        self.core = core

    def forward(self):
        """Prepares the right qubit of the CZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


def so4(state1, state2, device=None):
    """Pair of SO4 two-qubit rotation class instances, with rotations over
    different states.

    Parameters
    ----------
    state1 : int, the first of 4 quantum states to undergo the 2-qubit rotations
    state2 : int, the second of 4 quantum states to undergo the 2-qubit rotations
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4L, SO4R)
    """        
    R = SO4LR(state1, state2, 0, device=device)
    return R, SO4LR(state1, state2, 1, theta=R.theta, device=device)


class SO4LR(Module):
    """Left or right core of the two-qubit SO4 rotations gate.

    Parameters
    ----------
    state1 : int, the first of 4 quantum states to undergo the 2-qubit rotations
    state2 : int, the second of 4 quantum states to undergo the 2-qubit rotations
    position : int, if 0, then left core, if 1, then right core.
    device : string, device on which to run the computation.

    Returns
    -------
    if position == 0 --> SO4L
    if position == 1 --> SO4R
    """
    def __init__(self, state1, state2, position, theta=None, device=None):
        super().__init__()
        self.theta, self.position, self.device = Parameter(randn(1, device=device)), position, device
        if theta is not None:
            self.theta.data = theta.data
        ind1, ind2 = min(state1, state2), max(state1, state2)
        if (ind1, ind2) == (0,1):
            self.core_generator =  _so4_01
        elif (ind1, ind2) == (1,2):
            self.core_generator =  _so4_12
        elif (ind1, ind2) == (2,3):
            self.core_generator =  _so4_23
        else:
            raise IndexError('SO4 Rotation Gates have no state interaction pairs {}.\n'
                             'Valid state interactions pairs are (0,1), (1,2), and (2,3)'.format((state1, state2)))


    def forward(self):
        """Prepares the left or right qubit of the SO4 two-qubit rotation gate for forward contraction
        by calling the forward method and preparing the tt-factorized form of matrix representation.
        Update is based on theta (which is typically updated every epoch through backprop via Pytorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core_generator(self.theta, device=self.device)[self.position]


def _so4_01(theta, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along zeroth and first
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_01_L, SO4_01_R)
    """
    core1, core2 = tl.zeros((1,2,2,1), device=device), tl.zeros((1,2,2,1), device=device)
    core1[0,0,0,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T01I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,1), device=device), tl.zeros((1,2,2,1), device=device)
    core1[0,1,1,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T23I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,1), device=device), tl.zeros((1,2,2,1), device=device)
    core1[0,1,1,0] = core2[0,1,0,0] = 1
    core2[0,0,1,0] = -1
    R23I = [core1*sin(theta), core2]
    return [*tt_matrix_sum(TTMatrix(T01I), tt_matrix_sum(TTMatrix(T23I), TTMatrix(R23I)))]


def _so4_12(theta, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along first and second
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_12_L, SO4_12_R)
    """
    core1, core2 = tl.zeros((1,2,2,2), device=device), tl.zeros((2,2,2,1), device=device)
    core1[0,0,0,0] = core1[0,1,1,1] = core2[0,0,0,0] = core2[1,1,1,0] = 1
    T03I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,2), device=device), tl.zeros((2,2,2,1), device=device)
    core1[0,1,1,0] = core1[0,0,0,1] = core2[0,0,0,0] = core2[1,1,1,0] = 1
    T12I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,2), device=device), tl.zeros((2,2,2,1), device=device)
    core1[0,1,0,0] = core1[0,0,1,1] = core2[0,0,1,0] = 1
    core2[1,1,0,0] = -1
    R12I = [core1*sin(theta), core2]
    return [*tt_matrix_sum(TTMatrix(T03I), tt_matrix_sum(TTMatrix(T12I), TTMatrix(R12I)))]


def _so4_23(theta, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along second and third
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_23_L, SO4_23_R)
    """
    core1, core2 = tl.zeros((1,2,2,1), device=device), tl.zeros((1,2,2,1), device=device)
    core1[0,1,1,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T23I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,1), device=device), tl.zeros((1,2,2,1), device=device)
    core1[0,0,0,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T01I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,1), device=device), tl.zeros((1,2,2,1), device=device)
    core1[0,0,0,0] = core2[0,1,0,0] = 1
    core2[0,0,1,0] = -1
    R01I = [core1*sin(theta), core2]
    return [*tt_matrix_sum(TTMatrix(T23I), tt_matrix_sum(TTMatrix(T01I), TTMatrix(R01I)))]


def exp_pauli_y(device=None):
    """Matrix for sin(theta) component of Y-axis rotation in tt-tensor form.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) Y-rotation component.
    """
    return tl.tensor([[[[0],[-1]],[[1],[0]]]], device=device)
