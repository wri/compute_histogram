import numpy as np
from compute_histogram.main import _compute_histogram, _add_histogram


a = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
mask_layer = np.array([[False, True, True], [True, False, True], [True, True, False]])


def test_make_histogram():
    HISTO_RANGE = (0, 6)
    BINS = len(range(HISTO_RANGE[0], HISTO_RANGE[1]))

    histo = _compute_histogram(a, BINS, HISTO_RANGE)
    assert np.all(histo == np.array([0, 1, 2, 3, 2, 1]))

    HISTO_RANGE = (0, 9)
    BINS = len(range(HISTO_RANGE[0], HISTO_RANGE[1]))

    histo = _compute_histogram(a, BINS, HISTO_RANGE)
    assert np.all(histo == np.array([0, 1, 2, 3, 2, 1, 0, 0, 0]))


def test_add_histogram():
    h1 = np.array([0, 1, 2, 3, 2, 1, 0, 0, 0])
    h2 = np.array([1, 2, 1, 2, 0, 0, 1, 1, 1])

    histo = _add_histogram(h1, h2)
    assert np.all(histo == np.array([1, 3, 3, 5, 2, 1, 1, 1, 1]))
