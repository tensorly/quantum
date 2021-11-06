import tensorly as tl
tl.set_backend('pytorch')
from torch.nn import Module, ModuleList
from torch import transpose, randint
from itertools import chain
from opt_einsum import contract, contract_expression
from numpy import ceil

from .density_tensor import DensityTensor
from .tt_precontraction import qubits_contract, layers_contract
from .tt_contraction import contraction_eq


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


class TTCircuit(Module):
    """A simulator for variational quantum circuits using tensor ring tensors
    with PyTorch Autograd support.
    Can be used to compute: 1) the expectation value of an operator, 2) the single-qubit
    measurements of circuit's qubits; specifically for Multi-Basis Encoding [1], and 
    3) the partial trace of the circuit - all with Autograd support.
    [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, "Variational Quantum Optimization with Multi-Basis Encodings," (2021), arXiv:2106.13304.

    Parameters
    ----------
    unitaries : list of TT Unitaries, circuit operations
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    ncontral  : int, number of unitaries to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    equations : dictionary, accepts pre-computed/recycled equations for
               operator expectation values, single-qubit measurements,
               and partial traces.
    contractions : dictionary, accepts pre-computed/recycled paths for
               operator expectation values, single-qubit measurements,
               and partial traces.
    max_partial_trace_size : int, the maximum number of cores to keep in
               a single partial trace for single-qubit measurements.
    device : string, device on which to run the computation.

    Returns
    -------
    TTCircuit
    """
    def __init__(self, unitaries, ncontraq, ncontral, equations=None, contractions=None, max_partial_trace_size=4, device='cpu'): 
        super().__init__()
        self.nqubits, self.nlayers, self.ncontraq, self.ncontral = unitaries[0].nqubits, len(unitaries), ncontraq, ncontral
        self.nqsystems, self.nlsystems, self.layer_rep = int(ceil(self.nqubits/self.ncontraq)), int(ceil(self.nlayers/self.ncontral)), 2
        if equations is None:
            equations = {'expectation_value_equation': None, 'partial_trace_equation': None, 'partial_trace_equation_set': None}
        if contractions is None:
            contractions = {'expectation_value_contraction': None, 'partial_trace_contraction': None, 'partial_trace_contraction_set': None}
        self.equations, self.contractions = equations, contractions
        self.device, self.nparam_layers, contrsets = device, int(ceil(self.nlayers/self.layer_rep)), list(range(self.nqubits))
        self.contrsets = [contrsets[i:i+self.ncontraq] for i in range(0, self.nqubits, self.ncontraq)]
        self.max_partial_trace_size, segments = max_partial_trace_size, list(range(self.nqsystems))
        self.segments = [segments[0:self.max_partial_trace_size]] + [segments[i:i+self.max_partial_trace_size] for i in range(self.max_partial_trace_size, self.nqsystems, self.max_partial_trace_size)]
        self.unitaries = ModuleList(unitaries)


    def forward_expectation_value(self, state, operator, precontract_operator=True):
        """Full expectation value of self.measurement of the unitary evolved state.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        operator: tt-tensor, operator of which to get expectation value
        precontract_operator: bool, if true, the operator must be precontracted before main contraction pass

        Returns
        -------
        float, expectation value of self.measurement with unitary evolved state
        """
        if precontract_operator:
            operator = qubits_contract(operator, self.ncontraq, contrsets=self.contrsets)
        circuit = self._build_circuit(state, operator=operator)
        if self.contractions['expectation_value_contraction'] is None:
            if self.equations['expectation_value_equation'] is None:
                self.equations['expectation_value_equation'] = contraction_eq(self.nqsystems, 2*self.nlsystems+1)
            self.contractions['expectation_value_contraction'] = contract_expression(self.equations['expectation_value_equation'], *[core.shape for core in circuit])
        return self.contractions['expectation_value_contraction'](*circuit)


    def forward_single_qubit(self, state, op1, op2):
        """Expectation values of op for each qubit of state. Takes partial trace of subset of qubits and then
        takes single-operator measurements of these qubits.
        Specifically useful for Multi-Basis Encoding [1] (MBE).
        [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, "Variational Quantum Optimization with Multi-Basis Encodings," (2021), arXiv:2106.13304.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        op1 : tt-tensor, first single-measurement operator
        op2 : tt-tensor, second single-measurement operator

        Returns
        -------
        float, expectation value of self.measurement with unitary evolved state
        """
        circuit, expvals1, expvals2, count = self._build_circuit(state), tl.zeros((self.nqubits,), device=op1.device), tl.zeros((self.nqubits,), device=op1.device), 0
        if self.contractions['partial_trace_contraction_set'] is None:
            self._generate_partial_trace_contraction_set([core.shape for core in self._build_circuit(state)])
        for ind in range(len(self.segments)):
            partial = self.contractions['partial_trace_contraction_set'][ind](*circuit)
            partial_nqubits = int(tl.log2(tl.prod(tl.tensor(partial.shape)))/2)
            dims = [2 for i in range(partial_nqubits)]
            dims = [dims, dims]
            partial = DensityTensor(partial.reshape(sum(dims, [])), dims)
            for qubit_ind in range(partial_nqubits):
                qubit = partial.partial_trace(list(range(qubit_ind, qubit_ind+1)))[0].reshape(2,2)
                expvals1[count], expvals2[count], count = tl.sum(tl.diag(tl.dot(qubit, op1))), tl.sum(tl.diag(tl.dot(qubit, op2))), count+1
        return expvals1, expvals2


    def forward_partial_trace(self, state, kept_inds):
        """Partial trace for specified qubits in the output state of TTCircuit.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        kept_inds : list of ints, indices of the qubits to be kept in the partial trace

        Returns
        -------
        tensor in matrix form, partial trace of the circuit's output state
        """
        circuit = self._build_circuit(state)
        if self.contractions['partial_trace_contraction'] is None:
            if self.equations['partial_trace_equation'] is None:
                self.equations['partial_trace_equation'] = contraction_eq(self.nqsystems, 2*self.nlsystems, kept_inds=kept_inds)
            self.contractions['partial_trace_contraction'] = contract_expression(self.equations['partial_trace_equation'], *[core.shape for core in circuit])
        return self.contractions['partial_trace_contraction'](*circuit)


    def state_inner_product(self, state, compare_state):
        """Inner product of input state evolved in unitary with a comparison state.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        compare_state : tt-tensor, input state to be compared with evolved state

        Returns
        -------
        float, inner product of evolved state with compared state
        """
        eq = contraction_eq(self.nqsystems, self.nlsystems)
        built_layer = self._build_layer()
        circuit = compare_state + built_layer + state
        return contract(eq, *circuit)


    def _build_circuit(self, state, operator=[]):
        """Prepares the circuit gates and operators for forward pass of the tensor network.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        operators : tt-tensor, operator for which to calculate the expectation value, used by the
                    forward_expectation_value method.

        Returns
        -------
        list of tt-tensors, unitaries and operators of the TTCircuit, ready for contraction
        """
        built_layer = self._build_layer()
        built_layer_dagger = [tt_transpose(built_layer[i]) for n in range(self.nlsystems, 0, -1) for i in range((n-1)*self.nqsystems, n*self.nqsystems)]
        return state + built_layer + operator + built_layer_dagger + state


    def _build_layer(self):
        """Prepares the ket unitary gates gates for forward pass of the tensor network.

        Returns
        -------
        list of tt-tensors, unitaries of the TTCircuit, ready for contraction
        """
        built_layer = [unitary.forward() for unitary in self.unitaries]
        if self.nlayers % self.layer_rep > 0:
            built_layer = built_layer[:self.nlayers]
        return layers_contract(built_layer, self.ncontral)


    def _generate_partial_trace_contraction_set(self, shapes):
        """Populates the partial trace equations and contractions attributes for each of the the single-qubit
        measurements, as required by Multi-Basis Encoding.

        Parameters
        ----------
        shapes : list of shape tuples, the shapes of the tt-tensors to be contracted over
        """
        partial_trace_contraction_set = []
        if self.equations['partial_trace_equation_set'] is None:
            self._generate_partial_trace_equation_set()
        for equation in self.equations['partial_trace_equation_set']:
            partial_trace_contraction_set.append(contract_expression(equation, *shapes))
        self.contractions['partial_trace_contraction_set'] = partial_trace_contraction_set


    def _generate_partial_trace_equation_set(self):
        """Generates the partial trace equations for each of the the single-qubit measurements,
        as required by Multi-Basis Encoding.

        """
        partial_trace_equation_set = []
        for segment in self.segments:
            equation = contraction_eq(self.nqsystems, 2*self.nlsystems, kept_inds=segment)
            partial_trace_equation_set.append(equation)
        self.equations['partial_trace_equation_set'] = partial_trace_equation_set


def tt_transpose(tt):
    """Transpose single-qubit matrices in tt-tensor format.

    Parameters
    ----------
    tt : tt-tensor

    Returns
    -------
    Transpose of tt
    """
    return transpose(tt, 1, 2)
