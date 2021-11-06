Solving MaxCut
==============

TensorLy-Quantum provides the tools to use quantum circuit simulation to solve hard problems, for instance, MaxCut.

Multi-Basis Encoding
--------------------

To be able to scale to larger number of qubits, we developed a new technique called Multi-Basis Encoding [1]_ (MBE).
In :ref:`sphx_glr_auto_examples_plot_mbe_maxcut.py`, we probide an example of the MBE
quantum optimization algorithm for MaxCut via PyTorch Autograd supported TT-tensors with TensorLy-Quantum.
    

Solving MaxCut
--------------

We provide all the tools to solve MaxCut problems at scale, :mod:`tlquantum.maxcut`


References
----------

.. [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, 
       "Variational Quantum Optimization with Multi-Basis Encodings," (2021), arXiv:2106.13304.
