import numpy as np
from compute_histogram.main import (
    _compute_histogram,
    _add_histogram,
    _get_mask,
    _apply_mask,
)


a = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
mask_layer = np.array([[False, True, True], [True, False, True], [True, True, False]])


def test_get_mask():

    mask = _get_mask(a, 2)
    assert np.all(
        mask
        == np.array([[False, False, True], [False, True, True], [True, True, True]])
    )

    mask = _get_mask(a, 2, mask_layer)
    assert np.all(
        mask
        == np.array([[False, False, True], [False, False, True], [True, True, False]])
    )


def test_apply_mask():
    w = _apply_mask(mask_layer, a)
    assert np.all(w == np.array([2, 3, 2, 4, 3, 4]))


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
