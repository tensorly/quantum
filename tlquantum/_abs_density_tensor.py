import tensorly as tl
from collections.abc import MutableMapping
from abc import ABCMeta
from itertools import product


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


class AbsDensityTensor(MutableMapping, metaclass=ABCMeta):
    """
    Base Class for Density Tensors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def partial_trace(self, kept_indices):
        if self.op_type == 'rho':
            return self.partial_trace_dm(kept_indices)

        elif len(kept_indices) == len(self.subsystems[0]) - 1:
        # Can only trace over a single subsystem of standard tensor before we generally require entire density matrix.
            return self.partial_trace_state(kept_indices)

        else:
        # Eventually add transformation for standard tensor to density matrix for bra/ket with multisubsystem trace.
            return NotImplementedError


    def partial_trace_dm(self, kept_indices):
        return NotImplementedError


    def partial_trace_state(self, kept_indices):
        return NotImplementedError


    def _validate_subsystems(self, subsystems):
            if len(subsystems) != 2:
                raise ValueError('Physically, DensityTensor subsystems should have two sets of indices.\n'
                                 'DensityTensor subsystems represent partitions of square density matrices or row/column vectors.')

            if len(subsystems[0]) != len(subsystems[1]):
                raise ValueError('Physically, DensityTensor subsystems should have equal length index sets.\n'
                                 'DensityTensor represent partitions of square density matrices or row/column vectors.')

            if all(index == 1 for index in subsystems[0]):
            #check if row vector
                return subsystems, 'bra'

            if all(index == 1 for index in subsystems[1]):
            #check if column vector
                return subsystems, 'ket'

            if subsystems[0] == subsystems[1]:
            #check if square
                return subsystems, 'rho'

            raise ValueError('DensityTensor indices can be equal or one of them can be all ones.\n'
                             'DensityTensor represent partitions of square density matrices or row/column vectors.')
