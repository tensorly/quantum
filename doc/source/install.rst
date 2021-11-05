Installation
============

Getting the requirements
------------------------

We recommend you get the latest version of TensorLy and TensorLy-Torch directly from Github.
Alternatively, you can directly get the requirements with pip::

   pip install -r requirements.txt

Installing from source
----------------------

The recommended way to install TensorLy-Quantum is from source: first clone the repository and cd there::

   git clone https://github.com/tensorly/quantum
   cd quantum

Then install the package (here in editable mode with `-e` or equivalently `--editable`::

   pip install -e .

Installing from pip
-------------------

Alternatively, you can direclty get TensorLy-Quantum direclty from pypi::

   pip install tensorly-quantum


Testing that the package was correctly installed. First open a python interpreter::

   python

Then check that you can import tensorly-quantum and verify you have the correct version:

.. code-block:: python

   >>> import tlquantum
   >>> print(tlquantum.__version__)
   0.1.0

Making the documentation
------------------------

Our documentation is built with sphinx, you can make it easily. Once you're in the tensorly-quantum repository folder (see install from source, above)::

   cd doc
   pip install -r requirements_doc.txt
   make html

