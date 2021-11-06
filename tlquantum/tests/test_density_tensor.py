import tensorly as tl
from tensorly.testing import assert_array_almost_equal
from numpy import array, matmul, transpose
from numpy.linalg import norm

from ..density_tensor import DensityTensor


err_tol = 4


def test_validate_subsystems():
    square_subsystems = [[3, 2, 5], [3, 2, 5]]
    column_subsystems = [[1, 7, 4], [1, 1, 1]]
    row_subsystems = [[1, 1, 1], [2, 2, 5]]
    dummy_tensor = tl.randn((4, 4))

    DT = DensityTensor(dummy_tensor, square_subsystems)
    assert DT._validate_subsystems(square_subsystems) == (square_subsystems, 'rho')
    assert DT._validate_subsystems(column_subsystems) == (column_subsystems, 'ket')
    assert DT._validate_subsystems(row_subsystems) == (row_subsystems, 'bra')

def test_partial_trace_state():
    # For GHZ state
    ket = tl.zeros((8, 1))
    ket[0] = ket[-1] = 1/tl.sqrt(tl.tensor(2))
    ket = DensityTensor(ket, [[2, 2, 2], [1, 1, 1]])
    ptrace = tl.zeros((4, 4))
    ptrace[0, 0] = ptrace[-1, -1] = 0.5
    ptrace = tl.reshape(ptrace, (2,2,2,2))
    assert_array_almost_equal(ket.partial_trace([0,1])[0], ptrace, decimal=err_tol)
    assert ket.partial_trace([0,1])[1] == [[2, 2], [2, 2]]
    assert_array_almost_equal(ket.partial_trace([0,2])[0], ptrace, decimal=err_tol)
    assert ket.partial_trace([0,2])[1] == [[2, 2], [2, 2]]
    assert_array_almost_equal(ket.partial_trace([1,2])[0], ptrace, decimal=err_tol)
    assert ket.partial_trace([0,1])[1] == [[2, 2], [2, 2]]

    # For product state
    ket = tl.zeros((8, 1))
    ket[0] = 1
    ket = DensityTensor(ket, [[2, 2, 2], [1, 1, 1]])
    ptrace = tl.zeros((4, 4))
    ptrace[0, 0] = 1
    ptrace = tl.reshape(ptrace, (2,2,2,2))
    assert_array_almost_equal(ket.partial_trace([0,1])[0], ptrace, decimal=err_tol)
    assert_array_almost_equal(ket.partial_trace([0,2])[0], ptrace, decimal=err_tol)
    assert_array_almost_equal(ket.partial_trace([1,2])[0], ptrace, decimal=err_tol)

    # For random state
    ket = array([-1.83550206, -1.02964279, -0.680801,    1.14528818,  0.01097224, -1.46404358,
  0.65055477, -1.13207049,  0.70687065,  0.19446055, -2.35153163,  0.30000451,
  0.75665108,  0.2342452,  -0.15429997,  0.45617948])
    ket = ket.reshape(16, 1)/norm(ket)
    ket = tl.tensor(ket)
    ket = DensityTensor(ket, [[2, 2, 4], [1, 1, 1]])
    true_ptrace_01 = tl.tensor([[[[ 0.3635, -0.0148],
          [ 0.0262, -0.0587]],

         [[-0.0148,  0.2255],
          [-0.1257, -0.0557]]],


        [[[ 0.0262, -0.1257],
          [ 0.3607,  0.0633]],

         [[-0.0587, -0.0557],
          [ 0.0633,  0.0503]]]])
    assert_array_almost_equal(ket.partial_trace([0,1])[0], true_ptrace_01, decimal=err_tol)
    assert ket.partial_trace([0,1])[1] == [[2, 2], [2, 2]]
    true_ptrace_02 = tl.tensor([[[[ 0.1974,  0.1098,  0.0736, -0.1239],
          [-0.0755, -0.0208,  0.2528, -0.0320]],

         [[ 0.1098,  0.1877, -0.0147,  0.0280],
          [-0.1075, -0.0318,  0.1551, -0.0572]],

         [[ 0.0736, -0.0147,  0.0519, -0.0888],
          [ 0.0006,  0.0012,  0.0879,  0.0054]],

         [[-0.1239,  0.0280, -0.0888,  0.1519],
          [-0.0028, -0.0025, -0.1475, -0.0101]]],


        [[[-0.0755, -0.1075,  0.0006, -0.0028],
          [ 0.0628,  0.0184, -0.1042,  0.0326]],

         [[-0.0208, -0.0318,  0.0012, -0.0025],
          [ 0.0184,  0.0054, -0.0289,  0.0097]],

         [[ 0.2528,  0.1551,  0.0879, -0.1475],
          [-0.1042, -0.0289,  0.3254, -0.0455]],

         [[-0.0320, -0.0572,  0.0054, -0.0101],
          [ 0.0326,  0.0097, -0.0455,  0.0175]]]])
    assert_array_almost_equal(ket.partial_trace([0,2])[0], true_ptrace_02, decimal=err_tol)
    assert ket.partial_trace([0,2])[1] == [[2, 4], [2, 4]]
    true_ptrace_12 = tl.tensor([[[[ 0.2266,  0.1188, -0.0242, -0.1107],
          [ 0.0302,  0.1671, -0.0763,  0.1406]],

         [[ 0.1188,  0.0643,  0.0143, -0.0657],
          [ 0.0080,  0.0910, -0.0410,  0.0735]],

         [[-0.0242,  0.0143,  0.3511, -0.0870],
          [-0.1047,  0.0261, -0.0047, -0.0177]],

         [[-0.1107, -0.0657, -0.0870,  0.0821],
          [ 0.0140, -0.0941,  0.0409, -0.0679]]],


        [[[ 0.0302,  0.0080, -0.1047,  0.0140],
          [ 0.0335,  0.0094, -0.0064,  0.0195]],

         [[ 0.1671,  0.0910,  0.0261, -0.0941],
          [ 0.0094,  0.1288, -0.0579,  0.1034]],

         [[-0.0763, -0.0410, -0.0047,  0.0409],
          [-0.0064, -0.0579,  0.0262, -0.0473]],

         [[ 0.1406,  0.0735, -0.0177, -0.0679],
          [ 0.0195,  0.1034, -0.0473,  0.0873]]]])
    assert_array_almost_equal(ket.partial_trace([1,2])[0], true_ptrace_12, decimal=err_tol)
    assert ket.partial_trace([1,2])[1] == [[2, 4], [2, 4]]


def test_ptrace_dm():
    # For three dimensions, trace 2
    state = array([-0.00249035,  0.53490991,  1.01977248,  1.30321645,  1.7016896,  -0.82882385,
  0.00619055,  0.05851384,  0.48394437,  0.07139605,  1.20778799, -0.418087,
  0.92517927, -0.24892964,  0.48046415,  0.5655858,  -1.26665663,  1.13119857,
 -0.77152629, -1.02206745, -2.00345108, -0.7046856,  -1.47679974, -0.65070436,
 -0.25004916,  1.16761081,  1.59137088, -0.92458317, -0.08237956, -0.56372749,
 -1.17470156,  1.1172573 ])
    state = state.reshape(32, 1)/norm(state)
    dm = matmul(state, transpose(state))
    dm = (dm + matmul(state, transpose(state)))/2
    dm = tl.reshape(tl.tensor(dm), (2,4,4,2,4,4))
    dm = DensityTensor(dm, [[2, 4, 4], [2, 4, 4]])
    rho0 = dm.partial_trace([0])
    rho1 = dm.partial_trace([1])
    rho2 = dm.partial_trace([2])
    true_patrial_trace_0 = tl.tensor([[ 0.3390, -0.0675],[-0.0675,  0.6610]])
    assert_array_almost_equal(rho0[0], true_patrial_trace_0, decimal=err_tol)
    assert rho0[1] == [[2], [2]]
    true_patrial_trace_1 = tl.tensor([[ 0.2571,  0.1083,  0.0708,  0.0110],
        [ 0.1083,  0.3645, -0.0451,  0.1154],
        [ 0.0708, -0.0451,  0.2278, -0.0942],
        [ 0.0110,  0.1154, -0.0942,  0.1506]])
    assert_array_almost_equal(rho1[0], true_patrial_trace_1, decimal=err_tol)
    assert rho1[1] == [[4], [4]]
    true_partial_trace_2 = tl.tensor([[ 0.3295, -0.0638,  0.1591,  0.1075],
        [-0.0638,  0.1532,  0.1088, -0.0657],
        [ 0.1591,  0.1088,  0.3208,  0.0021],
        [ 0.1075, -0.0657,  0.0021,  0.1965]])
    assert_array_almost_equal(rho2[0], true_partial_trace_2, decimal=err_tol)
    assert rho2[1] == [[4], [4]]


def test_vonneumann_entropy():
    state = array([-1.83550206, -1.02964279, -0.680801,    1.14528818,  0.01097224, -1.46404358,
  0.65055477, -1.13207049,  0.70687065,  0.19446055, -2.35153163,  0.30000451,
  0.75665108,  0.2342452,  -0.15429997,  0.45617948])
    state = state.reshape(16, 1)/norm(state)
    dm = matmul(state, transpose(state))
    dm = (dm + matmul(state, transpose(state)))/2
    dm = tl.reshape(tl.tensor(dm), (2,8,2,8))
    dm = DensityTensor(dm, [[2, 8], [2, 8]])
    vne0 = dm.vonneumann_entropy([0])
    vne1 = dm.vonneumann_entropy([1])
    qt_vne0 = 0.9745033633882318
    qt_vne1 = 0.9745033633882336
    assert_array_almost_equal(vne0, qt_vne0, decimal=err_tol)
    assert_array_almost_equal(vne1, qt_vne1, decimal=err_tol)


def test_mutual_information():
    # For three dimensions, trace 2
    state = array([-0.00249035,  0.53490991,  1.01977248,  1.30321645,  1.7016896,  -0.82882385,
  0.00619055,  0.05851384,  0.48394437,  0.07139605,  1.20778799, -0.418087,
  0.92517927, -0.24892964,  0.48046415,  0.5655858,  -1.26665663,  1.13119857,
 -0.77152629, -1.02206745, -2.00345108, -0.7046856,  -1.47679974, -0.65070436,
 -0.25004916,  1.16761081,  1.59137088, -0.92458317, -0.08237956, -0.56372749,
 -1.17470156,  1.1172573 ])    
    state = state.reshape(32, 1)/norm(state)
    dm = matmul(state, transpose(state))
    dm = (dm + matmul(state, transpose(state)))/2
    dm = tl.reshape(tl.tensor(dm), (2,4,4,2,4,4))
    dm = DensityTensor(dm, [[2, 4, 4], [2, 4, 4]])
    vne0 = dm.mutual_information([0], [1])
    vne1 = dm.mutual_information([1], [0,2])
    vne2 = dm.mutual_information([2], [0])
    qt_vne0 = 1.0279272502071324
    qt_vne1 = 3.3537147522107285
    qt_vne2 = 0.7924815530809517
    assert_array_almost_equal(vne0, qt_vne0, decimal=err_tol)
    assert_array_almost_equal(vne1, qt_vne1, decimal=err_tol)
    assert_array_almost_equal(vne2, qt_vne2, decimal=err_tol)
