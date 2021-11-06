import tensorly as tl


# Author: Jean Kossaifi <jkossaifi@nvidia.com>
# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


def tt_sum(t1, t2):
    """Sums two TT tensors in decomposed form

    Parameters
    ----------
    t1 : tt-tensor
    t2 : tt-tensor

    Returns
    -------
    tt-tensor sum of t1 and t2

    Notes
    -----
    The solution can be easily seen by writing the element-wise expression.
    The sum of two third order cores A and B becomes a new core::
       | A(i)  0  |
       |  0   B(i)|
    
    In the code, we first form the two columns which we then concatenate::
       | A(i) | |  0  |
       |  0   | | B(i)|
    
    """
    new_tt, n_cores, device = [], len(t1), t1[0].device
    for i, (core1, core2) in enumerate(zip(t1, t2)):
        if i == 0: # First core is (1, I_1, R_0)
            core = tl.concatenate((core1, core2), axis=2)
        elif i == (n_cores - 1): # Last core is (I_N, R_N, 1)
            core = tl.concatenate((core1, core2), axis=0)
        else: # 3rd order cores (R_k, I_k, R_{k+1})
            padded_c1 = tl.concatenate(
                (core1, tl.zeros((t2.rank[i], core1.shape[1], t1.rank[i+1]), device=device)),
                axis=0
            )
            padded_c2 = tl.concatenate(
                (tl.zeros((t1.rank[i], core1.shape[1], t2.rank[i+1]), device=device), core2),
                axis=0
            )
            core = tl.concatenate((padded_c1, padded_c2), axis=2)
        new_tt.append(core)
    return tl.tt_tensor.TTTensor(new_tt)


def tt_matrix_sum(t1, t2):
    """Sums two TT matrices in decomposed form

    Parameters
    ----------
    t1 : tt-tensor matrix
    t2 : tt-tensor matrix

    Returns
    -------
    tt-tensor matrix sum of t1 and t2
    """
    if t1 == []:
        return tl.tt_matrix.TTMatrix(t2)
    if t2 == []:
        return tl.tt_matrix.TTMatrix(t1)
    t1, t2 = tl.tt_matrix.TTMatrix(t1), tl.tt_matrix.TTMatrix(t2)
    new_tt, n_cores, device = [], len(t1), t1[0].device
    for i, (core1, core2) in enumerate(zip(t1, t2)):
        if i == 0:
            core = tl.concatenate((core1, core2), axis=3)
        elif i == (n_cores - 1):
            core = tl.concatenate((core1, core2), axis=0)
        else:
            padded_c1 = tl.concatenate(
                (core1, tl.zeros((t2.rank[i], core1.shape[1], core1.shape[2], t1.rank[i+1]), device=device)),
                axis=0
            )
            padded_c2 = tl.concatenate(
                (tl.zeros((t1.rank[i], core1.shape[1], core1.shape[2], t2.rank[i+1]), device=device), core2),
                axis=0
            )
            core = tl.concatenate((padded_c1, padded_c2), axis=3)
        new_tt.append(core)
    return tl.tt_matrix.TTMatrix(new_tt)
