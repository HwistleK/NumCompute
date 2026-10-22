import numpy as np
import pytest

from numcompute.sort_search import (stable_sort, stable_argsort, multikey_sort,
                                    topk, quickselect, binary_search)


# ---------- stable sort ----------

def test_stable_sort_matches_numpy():
    a = np.array([3.0, -1.0, 2.0, 2.0, 0.5])
    assert np.array_equal(stable_sort(a), np.sort(a))


def test_stable_argsort_descending_keeps_tie_order():
    a = np.array([1, 3, 2, 3, 1])
    # Ties (3 at idx 1 and 3; 1 at idx 0 and 4) keep original order.
    assert stable_argsort(a, descending=True).tolist() == [1, 3, 2, 0, 4]


def test_stable_sort_descending_nan_last():
    a = np.array([1.0, np.nan, 3.0])
    out = stable_sort(a, descending=True)
    assert out[:2].tolist() == [3.0, 1.0] and np.isnan(out[2])


def test_stable_sort_rejects_strings():
    with pytest.raises(TypeError):
        stable_sort(np.array(["b", "a"]))


# ---------- multi-key sort ----------

def test_multikey_sort_primary_then_secondary():
    X = np.array([[2, 9], [1, 5], [2, 1], [1, 7]])
    out = multikey_sort(X, keys=[0, 1])
    assert out.tolist() == [[1, 5], [1, 7], [2, 1], [2, 9]]


def test_multikey_sort_mixed_direction():
    X = np.array([[2, 9], [1, 5], [2, 1], [1, 7]])
    out, order = multikey_sort(X, keys=[0, 1], ascending=[True, False],
                               return_indices=True)
    assert out.tolist() == [[1, 7], [1, 5], [2, 9], [2, 1]]
    assert order.tolist() == [3, 1, 0, 2]


def test_multikey_sort_invalid_key():
    with pytest.raises(ValueError):
        multikey_sort(np.zeros((3, 2)), keys=[5])


# ---------- top-k ----------

def test_topk_largest_and_smallest():
    a = np.array([5, 1, 9, 3, 7])
    vals, idx = topk(a, 3)
    assert vals.tolist() == [9, 7, 5] and idx.tolist() == [2, 4, 0]
    assert topk(a, 2, largest=False, return_indices=False).tolist() == [1, 3]


def test_topk_extreme_k():
    a = np.array([4, 2, 8])
    assert topk(a, 0, return_indices=False).shape == (0,)
    assert topk(a, 3, return_indices=False).tolist() == [8, 4, 2]
    with pytest.raises(ValueError):
        topk(a, 4)
    with pytest.raises(ValueError):
        topk(a, -1)


def test_topk_ties_ordered_by_index():
    a = np.array([5, 5, 5, 5])
    vals, idx = topk(a, 4)
    assert idx.tolist() == [0, 1, 2, 3]


def test_topk_nan_selected_last():
    a = np.array([np.nan, 2.0, 1.0])
    vals, idx = topk(a, 2)
    assert vals.tolist() == [2.0, 1.0]
    assert np.isnan(topk(a, 3, return_indices=False)[-1])


def test_topk_2d_along_axis():
    X = np.array([[1, 4, 2], [9, 0, 3]])
    assert topk(X, 2, axis=1, return_indices=False).tolist() == [[4, 2], [9, 3]]
    assert topk(X, 1, axis=0, return_indices=False).tolist() == [[9, 4, 3]]


def test_topk_non_integer_k():
    with pytest.raises(TypeError):
        topk([1, 2, 3], 1.5)


# ---------- quickselect ----------

def test_quickselect_matches_sort_random():
    rng = np.random.default_rng(0)
    a = rng.integers(-50, 50, size=200)   # many duplicates
    s = np.sort(a)
    for k in [0, 1, 57, 100, 199, -1]:
        assert quickselect(a, k, seed=1) == s[k]


def test_quickselect_all_equal_and_single():
    assert quickselect(np.full(10, 7.0), 4) == 7.0
    assert quickselect([3], 0) == 3


def test_quickselect_errors():
    with pytest.raises(ValueError):
        quickselect([], 0)
    with pytest.raises(ValueError):
        quickselect([1.0, np.nan], 0)
    with pytest.raises(ValueError):
        quickselect([1, 2, 3], 3)


def test_quickselect_non_contiguous():
    a = np.arange(20)[::-2]               # strided view: 19, 17, ..., 1
    assert not a.flags["C_CONTIGUOUS"]
    assert quickselect(a, 0) == 1


# ---------- binary search ----------

def test_binary_search_found_and_missing():
    arr = np.array([1, 3, 5, 7])
    assert binary_search(arr, 5) == (2, True)
    assert binary_search(arr, 4) == (2, False)
    assert binary_search(arr, 0) == (0, False)
    assert binary_search(arr, 9) == (4, False)


def test_binary_search_vectorised_with_duplicates():
    arr = np.array([1, 2, 2, 2, 5])
    idx, found = binary_search(arr, [2, 3, 5], side="right")
    assert idx.tolist() == [4, 4, 5]
    assert found.tolist() == [True, False, True]


def test_binary_search_empty_and_nan():
    assert binary_search(np.array([]), 1.0) == (0, False)
    assert binary_search(np.array([1.0, 2.0]), np.nan)[1] is False


def test_binary_search_unsorted_check():
    with pytest.raises(ValueError):
        binary_search([3, 1, 2], 1, check_sorted=True)
