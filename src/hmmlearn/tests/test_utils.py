import numpy as np
from numpy.testing import assert_allclose

from hmmlearn.utils import normalize


def test_normalize():
    A = np.random.normal(42.0, size=128)
    A[np.random.choice(len(A), size=16)] = 0.0
    assert (A == 0.0).any()
    normalize(A)
    assert_allclose(A.sum(), 1.0)


def test_normalize_along_axis():
    A = np.random.normal(42.0, size=(128, 4))
    for axis in range(A.ndim):
        A[np.random.choice(len(A), size=16), axis] = 0.0
        assert (A[:, axis] == 0.0).any()
        normalize(A, axis=axis)
        assert_allclose(A.sum(axis=axis), 1.0)
