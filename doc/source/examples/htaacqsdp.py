"""
HTAAC-QSDP
--------------------
Hadamard Test and Approximate Amplitude Constraint Quantum Semidefinite Programming ([1]_) (HTAAC-QSDP) quantum optimization algorithm for MaxCut using TensorLy-Quantum.
TensorLy-Quantum provides a Python interface 
to build TT-tensor network circuit simulator 
for large-scale simulation of variational quantum circuits
with full Autograd support similar to traditional PyTorch Neural Networks.
"""

import tlquantum as tlq
import torch
import itertools


torch.manual_seed(0)

device = 'cuda:0'
#device = 'cpu'
torch.cuda.set_device(device)
dtype = torch.complex64
constraint_dtype = torch.float32

ncontraq = 2 #number of qubits to pre-contract into single core
ncontral = 5 #number of layers to pre-contract into a single core

### hyperparameters simulation
nepochs = 300 # number of epochs per experiment
reps = 1 # number of repititions of experiment (full runs)

### hyperparamters optimizer
lr = 0.01 # learning rate

### circuit hyperparameters
nqubits = 10 # number of qubits
gate_rep = 120 # number of gates per qubit or circuit-depth
ps_order = 2
nterms = 800 # number of variables in SDP
W_divisor = 1 # divisor coefficient, not relevant for GSET graphs

### graph hyperparameters
alpha = 0.01 # unitary phase
U_P_multiplier = 1.0

### G11, G12, and G13 Graph Parameters
#name = 'Graphs/G11' # graph name
#sdp_value = 542 # best-known classical SDP value
#coeff_base = 100 #10 # size of coefficient
#reg = 1.2

### G14 and G15 Graph Parameters
#name = 'Graphs/G14' # graph name
#sdp_value = 2922 # best-known classical SDP value
#coeff_base = 100 #10 # size of coefficient
#reg = 3.0

### G20 and G21 Graph Parameters
name = 'Graphs/G20' # graph name
sdp_value = 838 # best-known classical SDP value
coeff_base = 50 #10 # size of coefficient
reg = 3.0

### build trivial input state as a tensor
state = tlq.spins_to_tt_state([0 for i in range(nqubits)], device=device, dtype=dtype)
state = tlq.qubits_contract(state, ncontraq)


### Alternative constraint vectors (dot product) instead of matrices (inner product with state)
pauli_obs_vec = []
for i in range(nqubits):
    vec = torch.zeros(1, 2**nqubits, dtype=constraint_dtype, device=device)
    for j in range(0, 2**nqubits, 2**(nqubits-1-i)):
        vec[0, j:j+2**(nqubits-1-i)] = (-1)**(j//2**(nqubits-1-i))
    pauli_obs_vec.append(vec)
reset = True
for i in range(2,ps_order+1):
    for gate_indices in list(itertools.combinations(list(range(nqubits)), i)):
        for ind in range(nqubits):
            if ind in gate_indices:
                if reset:
                    op = pauli_obs_vec[ind]
                    reset = False
                else:
                    op = op*pauli_obs_vec[ind]
        pauli_obs_vec.append(op)
        reset = True
nconstraints = len(pauli_obs_vec)
coeff = coeff_base*alpha/nconstraints

constraint_matrix = torch.cat(pauli_obs_vec, dim=0)
del pauli_obs_vec, vec, op


### build the two unitary matrices

vertices1 = torch.load(name+'_wspins1.pt').to(torch.int64).to(device)
vertices2 = torch.load(name+'_wspins2.pt').to(torch.int64).to(device)
weights = torch.load(name+'_weights.pt').to(device)

weights = weights/W_divisor
# half of the terms are missing because we don't add the reflection, but we save memory and just multiply by 2 below
U_W_sparse = alpha*torch.sparse_coo_tensor([vertices1.tolist(), vertices2.tolist()], weights, (2**nqubits, 2**nqubits), dtype=constraint_dtype)
### V is generator of the population balancing unitary U_P
bins = torch.bincount(torch.cat((vertices1, vertices2)))
max_bins = torch.max(bins)
U_P_vectorized = torch.zeros((1, 2**nqubits), dtype=dtype, device=device)
for i in range(2**nqubits):
    if i < nterms:
        U_P_vectorized[0,i] = (-(max_bins-bins[i])/reg)
    else:
        U_P_vectorized[0,i] = (-max_bins/reg)

U_P_vectorized = torch.imag(torch.exp(1j*alpha*U_P_vectorized)) * U_P_multiplier


for rep in range(reps):

    ### build the unitary gates layer by layer
    ### rebuild each rep to get new circuit/random initialization
    CZ0 = tlq.BinaryGatesUnitary(nqubits, ncontraq, tlq.cz(device=device, dtype=dtype), 0)
    CZ1 = tlq.BinaryGatesUnitary(nqubits, ncontraq, tlq.cz(device=device, dtype=dtype), 1)
    unitaries = []
    for r in range(gate_rep):
        unitaries += [tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device, dtype=dtype), CZ0, tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device, dtype=dtype), CZ1]
    circuit = tlq.TTCircuit(unitaries, ncontraq, ncontral)
    opt = torch.optim.Adam(circuit.parameters(), lr=lr, amsgrad=True) # basic ADAM AMSGrad optimizer

    for epoch in range(nepochs):

        output_state = circuit.to_ket(state).real

        ### loss from the objective function and population balancing unitaries
        loss = torch.sparse.mm(U_W_sparse, output_state)
        loss = 2*torch.mm(output_state.T, loss) # multiply by 2 because we don't have the reflected terms

        with torch.no_grad():
            cut_vec = tlq.calculate_cut(torch.sign(torch.real(output_state[0:nterms])), vertices1, vertices2, weights, get_cut=True).data
            print('Epoch Number ' + str(epoch), flush=True)
            print('Ratio of HTAAC-QSDP cut to classical SDP cut ' + str(cut_vec.item()/sdp_value), flush=True)
            print()

        ### square the state (abs val but real as these are all real states)
        output_state = output_state**2
        U_P_result = torch.mm(U_P_vectorized, output_state)
        constraints = torch.sum(torch.mm(constraint_matrix, output_state)**2)

        ### add the constraint loss for full backprop
        loss = loss + U_P_result + coeff*constraints

        loss.backward()
        opt.step()
        opt.zero_grad()


# %%
# References
# ----------
# .. [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, "Quantum Semidefinite Programming with the Hadamard Test and Approximate Amplitude Constraints", (2022), arXiv:2206.14999.
