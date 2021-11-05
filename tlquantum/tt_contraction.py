from opt_einsum.parser import get_symbol
from collections import Counter


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>
# Author: Jean Kossaifi <jkossaifi@nvidia.com>

# License: BSD 3 clause


def contraction_eq(nqubits, nlayers, kept_inds=None):
    """Generates einsum contraciton equation.

    Parameters
    ----------
    nqubits : int, number of qubits to contract over
    nlayers : int, number of layers to contract over
    kept_inds : list of inds, qubit indices to keep. If not None, then ptrace equation is generated
    
    Returns
    -------
    string of the contraction equation.
    """
    start = 1
    tt_idx = []
    for i in range(nqubits):
        idx = [start+2*i, start+2*i+1, start+2*i+2]
        tt_idx.append(''.join(get_symbol(j) for j in idx))

    start2 = start+2+2*i
    max_ind = 2*start2 + (nlayers+1)*nqubits + 2
    factors_idx = []
    for tt in range(nlayers):
        for i in range(nqubits):
            if i==0:
                idx = [start2+2*tt*nqubits, start2+1+2*tt*nqubits, start+1+2*tt*nqubits, start2+2+2*tt*nqubits]
                if tt==0:
                    idx[0] = 0
            elif i==nqubits-1:
                idx = [start2+2*i+2*tt*nqubits, start2+2*i+1+2*tt*nqubits, start+1+2*i+2*tt*nqubits, start2+2*tt*nqubits]
                if tt==0:
                    idx[-1] = 0
            else:
                idx = [start2+2*i+2*tt*nqubits, start2+2*i+1+2*tt*nqubits, start+1+2*i+2*tt*nqubits, start2+2*i+2+2*tt*nqubits]
            if (kept_inds is not None) and (tt == int(nlayers/2)) and (i in kept_inds):
                idx[2] += max_ind
            factors_idx.append(''.join(get_symbol(j) for j in idx))

    start_phys = start2+1+2*(nlayers-1)*nqubits
    start_virt = start2+2*(nqubits-1)+1+2*(nlayers-1)*nqubits + 1
    measure_idx = []
    for i in range(nqubits):
        idx = [start_virt+i-1, start_phys+2*i, start_virt+i]
        if i==0:
            idx[0] = start
        if i==nqubits-1:
            idx[-1] = start2
        measure_idx.append(''.join(get_symbol(j) for j in idx))

    if kept_inds is not None:
        out_idx = ''.join(tt_idx)+''.join(factors_idx)+''.join(measure_idx)
        counts = Counter(out_idx)
        out_idx = ''.join(ind for ind, count in counts.items() if count == 1)
        return ','.join(i for i in measure_idx) + ',' + ','.join(i for i in factors_idx) + ',' + ','.join(i for i in tt_idx) + '->' + out_idx

    return ','.join(i for i in measure_idx) + ',' + ','.join(i for i in factors_idx) + ',' + ','.join(i for i in tt_idx) + '-> b'

