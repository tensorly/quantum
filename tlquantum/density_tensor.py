from ._abs_density_tensor import AbsDensityTensor
import tensorly as tl
import tensorly.metrics as tlm


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


class DensityTensor(AbsDensityTensor):
    """A quantum state container for state and density matrix operations, including partial
    traces and quantum information metric calculations.

    Parameters
    ----------
    tensor : tt-tensor, state or density matrix
    subsystems : list of ints, system sizes in correspondence of the size of tensor

    Returns
    -------
    DensityTensor
    """
    def __init__(self, tensor, subsystems):
        super().__init__()

        self.tensor = tensor
        self.subsystems, self.op_type = self._validate_subsystems(subsystems)

    def __getitem__(self, index):
        if index == 0:
            return self.tensor

        elif index == 1:
            return self.subsystems

        elif index == 2:
            return self.op_type

        else:
            raise IndexError('DensityTensor has no index {}.\n'
                             'DensityTensor is object of two indices: tensor and subsystems.'.format(index))


    def __setitem__(self, index, value):
        if index == 0:
            self.tensor = value

        elif index == 1:
            self.subsystems, self.op_type = self._validate_subsystems(subsystems) #_validate_subsystems(value)

        elif index == 2:
            raise IndexError('Operator type is determined by subsystems and can be updated by updating subsystems.')

        else:
            raise IndexError('DensityTensor has no index {}.\n'
                             'DensityTensor is object of two indices: tensor and subsystems.'.format(index))


    def __delitem__(self, index, value):
        if index == 0:
            self.tensor = None

        if index == 1:
            self.subsystems = None
            self.op_type = None

        if index == 2:
            raise IndexError('Operator type is determined by subsystems and can be updated by updating subsystems.')

        else:
            raise IndexError('DensityTensor has no index {}.\n'
                             'DensityTensor is object of two indices: tensor and subsystems.'.format(index))


    def __iter__(self):
        yield self.tensor
        yield self.subsystems
        yield self.op_type


    def __len__(self):
        return 2


    def __repr__(self):
        message = 'DensityTensor in {} form of dimensions {} and subsystems {}.'.format(self.op_type, self.tensor.shape, self.subsystems)
        return message


    def partial_trace_dm(self, kept_indices):
        """Partial trace for density matrix.

        Parameters
        ----------
        kept_indices : list of int, indices to not trace over and keep for ptrace

        Returns
        -------
        DensityTensor, result of partial trace
        """
        num_subsystems = len(self.subsystems[0])
        traced_indices = [index for index in range(num_subsystems) if index not in kept_indices]
        new_subsystems = [self.subsystems[0][index] for index in range(num_subsystems) if index in kept_indices]
        rho = _partial_trace(self.tensor, traced_indices, num_subsystems)
        rho = tl.reshape(rho, tuple(new_subsystems*2))
        return DensityTensor(rho, [new_subsystems, new_subsystems])


    def partial_trace_state(self, kept_indices):
        """Partial trace for state vector.

        Parameters
        ----------
        kept_indices : list of int, indices to not trace over and keep for partial trace

        Returns
        -------
        DensityTensor, result of partial trace
        """
        traced_index = [index for index in range(len(self.subsystems[0])) if index not in kept_indices][0]
        temp_ket = self.tensor
        temp_subsystems = self.subsystems
        if self.op_type == 'bra':
            temp_ket = tl.transpose(tl.transpose(temp_ket))
            temp_subsystems = temp_subsystems[::-1]
        temp_ket = tl.reshape(temp_ket, tuple(temp_subsystems[0])) #because we don't enforce that tensor shape match subsystems
        temp_ket = tl.unfold(temp_ket, traced_index)
        temp_ket = tl.dot(tl.transpose(temp_ket), tl.conj(temp_ket)) #reduced_row, update to make DensityTensor of proper dimensions
        new_subsystems = [temp_subsystems[0][index] for index in kept_indices]
        temp_ket = tl.reshape(temp_ket, tuple(new_subsystems*2))
        return DensityTensor(temp_ket, [new_subsystems, new_subsystems])


    def vonneumann_entropy(self, kept_indices):
        """Von Neumann entropy for the partial trace of the density tensor.

        Parameters
        ----------
        kept_indices : list of int, indices to not trace over and keep for partial trace

        Returns
        -------
        float, Von Neumann entropy of specified partial trace
        """
        dm = self.partial_trace(kept_indices)[0]
        square_dim = int(tl.sqrt(tl.prod(tl.tensor(dm.shape))))
        return tlm.vonneumann_entropy(tl.reshape(dm, (square_dim, square_dim)))


    def mutual_information(self, kept_indices1, kept_indices2):
        """Mutual Information for the partial traces of the specified kept indices.

        Parameters
        ----------
        kept_indices1 : list of int, indices to not trace over and keep for partial trace
        kept_indices2 : list of int, indices to not trace over and keep for partial trace

        Returns
        -------
        float, Mutual Information of specified partial traces
        """
        return self.vonneumann_entropy(kept_indices1) + self.vonneumann_entropy(kept_indices2) - self.vonneumann_entropy(kept_indices1+kept_indices2)


def _partial_trace(tensor, traced_dims, num_subsystems):
    """Traces over density matrix of qubits producing density matrix of partial trace.

    Parameters
    ----------
    tensor : torch tensor of floats, density matrix to be partially traced over
    traced_dims : list of ints, qubit indices to trace over
    num_subsystems : int, number of subsystems in tensor

    Returns
    -------
    torch tensor of floats, partial trace of tensor
    """
    start = ord('a')
    all_modes1 = [chr(start+i) for i in range(num_subsystems)]
    all_modes2 = [chr(start+i+num_subsystems) for i in range(num_subsystems)]
    for dim in range(num_subsystems):
        if dim in traced_dims:
            all_modes2[dim] = all_modes1[dim]
    all_modes1 = ''.join(all_modes1)
    all_modes2 = ''.join(all_modes2)
    inds_initial = all_modes1+all_modes2 
    inds_final = [all_modes1[iii] for iii in range(num_subsystems) if iii not in traced_dims]
    inds_final = ''.join(inds_final + [all_modes2[iii] for iii in range(num_subsystems) if iii not in traced_dims])
    return tl.einsum(inds_initial + '->' + inds_final, tensor)
