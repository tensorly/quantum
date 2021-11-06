"""
Variational Quantum Eigensolver
-------------------------------

Variational Quantum Eigensolver ([1]_) (VQE) 
with Transverse Field Ising Model Hamiltonian using TensorLy-Quantum.
TensorLy-Quantum provides a Pythonic API to TT-tensor network circuit simulation
for large-scale simulation of variational quantum circuits,
with full Autograd support and an interface similar to PyTorch Neural Networks.
"""

import tensorly as tl
import tlquantum as tlq
from tensorly.tt_matrix import TTMatrix
from torch import randint, rand, arange
from torch.optim import Adam
import matplotlib.pyplot as plt


# %% Set up simulation parameters
# Uncomment the line below to use the GPU

#device = 'cuda' 
device = 'cpu' 

nepochs = 80 #number of training epochs

nqubits = 5 #number of qubits
ncontraq = 2 #2 #number of qubits to pre-contract into single core
ncontral = 2 #2 #number of layers to pre-contract into a single core
nterms = 10
lr = 0.5


# %% Generate an input state. For each qubit, 0 --> |0> and 1 --> |1>
state = tlq.spins_to_tt_state([0 for i in range(nqubits)], device) # generate generic zero state |00000>
state = tlq.qubits_contract(state, ncontraq)


# %% Build a transverse field Ising model Hamiltonian. 
# Here we build a random spin-spin and transverse field weights.
# two-qubit terms
qubits1 = randint(nqubits, (nterms,), device=device) # randomly generated first qubits of each two-qubit term
qubits2 = randint(nqubits, (nterms,), device=device) # randomly generated second qubits of each two-qubit term
qubits2[qubits2==qubits1] += 1 # because qubits in this Hamiltonian randomly generated, eliminate self-interacting terms
qubits2[qubits2 >= nqubits] = 0
weights = rand((nterms,), device=device) # randomly generated coefficients of each two-qubit interaction in Hamiltonian
binary_H = tlq.binary_hamiltonian(tlq.pauli_z(device), nqubits, qubits1, qubits2, weights) # build the spin-spin Hamiltonian H

# %% transverse field (one-qubit) terms
qubits = arange(nqubits, device=device) # specify that each qubit will have a transverse field term
weights = rand((nqubits,), device=device) # randomly generated coefficients for the transverse field felt by each qubit
unary_H = tlq.unary_hamiltonian(tlq.pauli_x(device), nqubits, qubits, weights) #build the transverse field Hamiltonian

# %% build the transverse field Ising model Hamiltonian
Ising_H = tlq.tt_matrix_sum(binary_H, unary_H)


# %% Build unitary gates in TT tensor form

# %% the gate of each qubit can be specified as a custom unitary
custom_U = tlq.Unitary([tlq.RotY(device), *tlq.so4(0,1, device), tlq.RotY(device), *tlq.so4(2, 3, device)], nqubits, ncontraq)

# %% or entire layers of gates (one gate per each qubit) can be specified using Unary/BinaryGatesUnitary
RotY = tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device) # one Y-axis rotation gate applied to each qubit of the circuit
parity = 0
CZ0 = tlq.BinaryGatesUnitary(nqubits, ncontraq, tlq.cz(device=device), parity) # one controlled-z gate for each pair of qubits using even parity (even qubits control)
parity = 1
SO4_01 = tlq.BinaryGatesUnitary(nqubits, ncontraq, tlq.so4(2,3, device=device), parity) # one SO4 rotation about two-qubit states |2> and |3> with odd parity


# %% Combine layers of unitary gates for use in quantum circuit

# %% specify circuit order unitary by unitary in circuit list
unitaries = [RotY, SO4_01, tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device), CZ0]

# %% or build circuit block by block
repeat_block, unitaries_automatic = 3, []
for i in range(repeat_block):
    unitaries_automatic += unitaries


# %% VQE Simulation of the transverse field Ising model

# %% build TTCircuit using specified unitaries
circuit = tlq.TTCircuit(unitaries, ncontraq, ncontral)
opt = Adam(circuit.parameters(), lr=lr, amsgrad=True) # define PyTorch optimizer
energy_vec = tl.zeros(nepochs)

for epoch in range(nepochs):
    # TTCircuit forward pass computes expectation value of Ising_H
    energy = circuit.forward_expectation_value(state, Ising_H)
    print('Energy (loss) at epoch ' + str(epoch) + ' is ' + str(energy[0].item()) + '. \n')

    # PyTorch Autograd attends to backwards pass and parameter update
    energy.backward()
    opt.step()
    opt.zero_grad(epoch)
    energy_vec[epoch] = energy


# %% VIsualize the results
Ising_H = TTMatrix(Ising_H).to_matrix()
true_energies, _ = tl.eigh(Ising_H)
ground_state_energy = true_energies[0]
plt.figure()
plt.plot(energy_vec.detach().numpy(), color='r')
plt.hlines(ground_state_energy, 0, nepochs, color='k', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Energy')
plt.xticks()
plt.yticks()
plt.legend(['Variational Solution', 'Ground Truth'])
plt.show()

# %% 
# References
# ----------
# .. [1] Peruzzo, A., McClean, J., Shadbolt, P. et al. A variational eigenvalue solver on a photonic quantum processor. Nat Commun 5, 4213 (2014). 
