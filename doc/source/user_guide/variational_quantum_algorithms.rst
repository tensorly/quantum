Variational Quantum Algorithms with TensorLy-Quantum
====================================================

TensorLy-Quantum provides the tools to use quantum circuit simulation to solve hard problems with tensor-based simulation of variational quantum algorithms. For instance, TensorLy-Quantum users can both find the ground states of quantum Ising models and construct quantum algorithms for NP-hard classical problems, such as MaxCut. In addition to flexibile circuit ansatze and operator (Hamiltonian) building functions, we also provide the framework for novel quantum algorithms, such as Multi-Basis Encoding.

Solving the Transverse Field Ising Model
----------------------------------------

Users can easily build quantum Hamiltonians of interest and solve them
using the flexible circuit ansatze of TensorLy-Quantum. 
In :ref:`sphx_glr_auto_examples_plot_vqe_transverse_field_Ising.py`, 
we provide an example how to execute such an algorithm.

Multi-Basis Encoding
--------------------

To be able to scale to larger number of qubits, we developed a new technique called Multi-Basis Encoding [1]_ (MBE).
In :ref:`sphx_glr_auto_examples_plot_mbe_maxcut.py`, we provide an example of the MBE
quantum optimization algorithm for MaxCut via PyTorch Autograd supported TT-tensors with TensorLy-Quantum.
    

Solving MaxCut
--------------

We provide all the tools to solve MaxCut problems at scale, :mod:`tlquantum.maxcut`


References
----------

.. [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, 
       "Variational Quantum Optimization with Multi-Basis Encodings," (2021), arXiv:2106.13304.
