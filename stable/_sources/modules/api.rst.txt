=============
API reference
=============

:mod:`tlquantum`: Quantum ML

Density Tensors
===============

.. automodule:: tlquantum.density_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.density_tensor

TensorLy-Quantum provides a convenient class for representing and manipulating density tensors: 

.. autosummary::
    :toctree: generated
    :template: class.rst

    DensityTensor

Tensor-Trains
=============

Also known as MPO, MPS, tensor-train is an efficient way to represent state-vectors and operators in factorized form.
In TensorLy-Quantum, we provide out-of-the-box layers and circuits in the TT format.

.. automodule:: tlquantum.tt_gates
    :no-members:
    :no-inherited-members:


Gates in TT-form
----------------

.. currentmodule:: tlquantum.tt_gates


.. autosummary::
    :toctree: generated
    :template: class.rst

   Unitary
   IDENTITY
   RotY
   UnaryGatesUnitary
   BinaryGatesUnitary
   CZL
   CZR
   CNOTL
   CNOTR
   SO4LR


We also provide some convenience functions to facilitate creation of some of the gates:

.. autosummary::
    :toctree: generated
    :template: function.rst

    cz
    cnot
    so4


Operators in TT-form
--------------------

.. automodule:: tlquantum.tt_operators
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.tt_operators

.. autosummary::
    :toctree: generated
    :template: function.rst

    unary_hamiltonian
    binary_hamiltonian
    identity
    pauli_x
    pauli_z


States in TT-form
-----------------

.. automodule:: tlquantum.tt_state
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.tt_state

.. autosummary::
    :toctree: generated
    :template: function.rst

    spins_to_tt_state
    tt_norm


Creating circuits
-----------------

.. automodule:: tlquantum.tt_circuit
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.tt_circuit

.. autosummary::
    :toctree: generated
    :template: class.rst

    TTCircuit


Adding TT/MPS/MPOs
------------------

.. automodule:: tlquantum.tt_sum
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.tt_sum

.. autosummary::
    :toctree: generated
    :template: function.rst

    tt_sum
    tt_matrix_sum


Contracting Tensor Networks
---------------------------


.. automodule:: tlquantum.tt_contraction
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.tt_contraction

.. autosummary::
    :toctree: generated
    :template: function.rst

   contraction_eq

Precontraction
--------------

.. automodule:: tlquantum.tt_precontraction
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.tt_precontraction

.. autosummary::
    :toctree: generated
    :template: function.rst

   qubits_contract
   layers_contract

Solving MaxCut
==============


.. automodule:: tlquantum.maxcut
    :no-members:
    :no-inherited-members:

.. currentmodule:: tlquantum.maxcut

.. autosummary::
    :toctree: generated
    :template: function.rst

   calculate_cut
   brute_force_calculate_maxcut
