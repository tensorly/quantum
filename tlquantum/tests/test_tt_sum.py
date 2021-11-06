import tensorly as tl
from tensorly.random import random_tt, random_tt_matrix
from tensorly.tt_matrix import TTMatrix
from tensorly.testing import assert_array_almost_equal

from ..tt_sum import tt_sum, tt_matrix_sum


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>
# Author: Jean Kossaifi <jkossaifi@nvidia.com>


def test_tt_sum():
    t1 = random_tt((3, 4, 3, 2), 0.5)
    t2 = random_tt((3, 4, 3, 2), 0.5)
    true_res = t1.to_tensor() + t2.to_tensor()
    res = tt_sum(t1, t2).to_tensor()
    assert_array_almost_equal(true_res, res)


def test_one_core_mpo_sum():
    t1 = random_tt_matrix((2,2), 2)
    t2 = random_tt_matrix((2,2), 2)
    true_res = t1.to_matrix() + t2.to_matrix()
    res = TTMatrix([t1[0] + t2[0]]).to_matrix()
    assert_array_almost_equal(true_res, res)


def test_tt_matrix_sum():
    t1 = random_tt_matrix((2,2,2,2), 4)
    t2 = random_tt_matrix((2,2,2,2), 4)
    assert_array_almost_equal(tt_matrix_sum(t1, t2).to_matrix(), tt_matrix_sum(t1, t2).to_matrix())

    t1 = random_tt_matrix((2,2,2,2,2,2), 4)
    t2 = random_tt_matrix((2,2,2,2,2,2), 4)
    assert_array_almost_equal(tt_matrix_sum(t1, t2).to_matrix(), tt_matrix_sum(t1, t2).to_matrix())

    t1 = random_tt_matrix((3,8,5,7), 12)
    t2 = random_tt_matrix((3,8,5,7), 11)
    assert_array_almost_equal(tt_matrix_sum(t1, t2).to_matrix(), tt_matrix_sum(t1, t2).to_matrix())
