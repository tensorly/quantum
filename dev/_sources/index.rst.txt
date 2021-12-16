:no-toc:
:no-localtoc:
:no-pagination:

.. only:: latex

   TensorLy-Quantum
   ================


.. TensorLy-Quantum documentation

.. only:: html

   .. raw:: html

      <div class="container content">
      <br/><br/>

.. image:: _static/logos/tensorly-quantum-logo.png
   :align: center
   :width: 1000

.. only:: html

   .. raw:: html 
   
      <div class="has-text-centered">
         <h3> Tensor-Based Quantum Machine Learning </h3>
      </div>
      <br/><br/>

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   user_guide/index
   modules/api
   auto_examples/index
   about

TensorLy-Quantum is a Python library for Tensor-Based Quantum Machine Learning that
builds on top of `TensorLy <https://github.com/tensorly/tensorly/>`_
and `PyTorch <https://pytorch.org/>`_.

With TensorLy-Quantum, you can easily: 

- **Create quantum circuit**: .
- **Leverage tensor methods**: the state vectors are efficiently represented in factorized form as Tensor-Rings (MPS) and the operators as TT-Matrices (MPO)
- **Efficient simulation**: tensorly-quantum leverages the factorized structure to efficiently perform quantum simulation without ever forming the full, dense operators and state-vectors
- **Multi-Basis Encoding**: we provide multi-basis encoding out-of-the-box for scalable experimentation
- **Solve hard problems**: we provide all the tools to solve the MaxCut problem for an unprecendented number of qubits / vertices


.. only:: html

   .. raw:: html

      <br/> <br/>
      <br/>

      <div class="container has-text-centered">
      <a class="button is-large is-dark is-primary" href="install.html">
         Get Started!
      </a>
      </div>
      
      </div>
