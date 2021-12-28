"""
Multi-Basis Encoding
--------------------

Multi-Basis Encoding ([1]_) (MBE) quantum optimization algorithm for MaxCut using TensorLy-Quantum.
TensorLy-Quantum provides a Python interface 
to build TT-tensor network circuit simulator 
for large-scale simulation of variational quantum circuits
with full Autograd support similar to traditional PyTorch Neural Networks.
"""


import tensorly as tl
import tlquantum as tlq
from torch import randint, rand, arange, cat, tanh, no_grad, float32
from torch.optim import Adam
import matplotlib.pyplot as plt


# %% Set up simulation parameters
# Uncomment the line below to use the GPU

device = 'cuda' 
#device = 'cpu'

dtype = float32

nepochs = 40 #number of training epochs

nqubits = 20 #number of qubits
ncontraq = 2 #2 #number of qubits to pre-contract into single core
ncontral = 2 #2 #number of layers to pre-contract into a single core
nterms = 20
lr = 0.7


# %% Generate an input state. For each qubit, 0 --> |0> and 1 --> |1>
state = tlq.spins_to_tt_state([0 for i in range(nqubits)], device=device, dtype=dtype) # generate generic zero state |00000>
state = tlq.qubits_contract(state, ncontraq)


# %% Generate the graph vertices/edges. Each pair of qubits represents two vertices with an edge between them.
# Here we build a random graph with randomly weighted edges.
# Note: MBE allows us to encode two vertices (typically two qubits) into a single qubit using the z and x-axes.
# If y-axis included, we can encode three vertices per qubit.
vertices1 = randint(2*nqubits, (nterms,), device=device) # randomly generated first qubits (vertices) of each two-qubit term (edge)
vertices2 = randint(2*nqubits, (nterms,), device=device) # randomly generated second qubits (vertices) of each two-qubit term (edge)
vertices2[vertices2==vertices1] += 1 # because qubits in this graph are randomly generated, eliminate self-interacting terms
vertices2[vertices2 >= nqubits] = 0
weights = rand((nterms,), device=device) # randomly generated edge weights


# %% Build unitary gates in TT tensor form
RotY1 = tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device, dtype=dtype) #single-qubit rotations about the Y-axis
RotY2 = tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device, dtype=dtype)
CZ0 = tlq.BinaryGatesUnitary(nqubits, ncontraq, tlq.cz(device=device, dtype=dtype), 0) # one controlled-z gate for each pair of qubits using even parity (even qubits control)
unitaries = [RotY1, CZ0, RotY2]


# %% Multi-Basis Encoding (MBE) Simulation for MaxCut optimization

circuit = tlq.TTCircuit(unitaries, ncontraq, ncontral) # build TTCircuit using specified unitaries
opz, opx = tl.tensor([[1,0],[0,-1]], device=device, dtype=dtype), tl.tensor([[0,1],[1,0]], device=device, dtype=dtype) # measurement operators for MBE
print(opz)
opt = Adam(circuit.parameters(), lr=lr, amsgrad=True) # define PyTorch optimizer
loss_vec = tl.zeros(nepochs)
cut_vec = tl.zeros(nepochs)

for epoch in range(nepochs):
    # TTCircuit forward pass computes expectation value of single-qubit pauli-z and pauli-x measurements
    spinsz, spinsx = circuit.forward_single_qubit(state, opz, opx)
    spins = cat((spinsz, spinsx))
    nl_spins = tanh(spins) # apply non-linear activation function to measurement results
    loss = tlq.calculate_cut(nl_spins, vertices1, vertices2, weights) # calculate the loss function using MBE
    print('Relaxation (raw) loss at epoch ' + str(epoch) + ': ' + str(loss.item()) + '. \n')
    with no_grad():
        cut_vec[epoch] = tlq.calculate_cut(tl.sign(spins), vertices1, vertices2, weights, get_cut=True) #calculate the rounded MaxCut estimate (algorithm's result)
        print('Rounded MaxCut value (algorithm\'s solution): ' + str(cut_vec[epoch]) + '. \n')

    # PyTorch Autograd attends to backwards pass and parameter update
    loss.backward()
    opt.step()
    opt.zero_grad()
    loss_vec[epoch] = loss


# %% VIsualize the result
plt.rc('xtick')
plt.rc('ytick')
fig, ax1 = plt.subplots()
ax1.plot(loss_vec.detach().numpy(), color='k')
ax2 = ax1.twinx()
ax2.plot(cut_vec.detach().numpy(), color='g')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='k')
ax2.set_ylabel('Cut', color='g')
plt.show()

# %%
# References
# ----------
# .. [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, "Variational Quantum Optimization with Multi-Basis Encodings," (2021), arXiv:2106.13304.
