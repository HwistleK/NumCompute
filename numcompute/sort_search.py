"""Sorting, top-k selection and searching utilities.

All functions accept array-like input and return NumPy arrays.
Core computations are vectorised; the only Python loop is the
partition loop inside ``quickselect``, which is inherent to the algorithm.
"""

import numpy as np

__all__ = ["stable_sort", "stable_argsort", "multikey_sort", "topk",
           "quickselect", "binary_search"]


def _as_numeric(a, name="a"):
    """Convert input to a numeric ndarray or raise TypeError."""
    arr = np.asarray(a)
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    return arr


def stable_argsort(a, axis=-1, descending=False):
    """Return indices that stably sort ``a`` along ``axis``.

    Equal elements keep their original relative order, in both ascending
    and descending mode. NaNs are placed last in both modes.

    Parameters
    ----------
    a : array_like, shape (..., n, ...)
        Numeric input array.
    axis : int, default -1
        Axis to sort along.
    descending : bool, default False
        Sort from largest to smallest if True.

    Returns
    -------
    indices : ndarray of int, same shape as ``a``

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    numpy.exceptions.AxisError
        If ``axis`` is out of range.

    Complexity
    ----------
    Time O(n log n) per slice, space O(n).
    """
    arr = _as_numeric(a).astype(float, copy=False)
    # Negating keeps ties in original order (flipping the result would not).
    key = -arr if descending else arr
    return np.argsort(key, axis=axis, kind="stable")


def stable_sort(a, axis=-1, descending=False):
    """Return a stably sorted copy of ``a`` along ``axis``.

    Parameters
    ----------
    a : array_like
        Numeric input array.
    axis : int, default -1
        Axis to sort along.
    descending : bool, default False
        Sort from largest to smallest if True. NaNs are placed last.

    Returns
    -------
    sorted : ndarray, same shape and dtype as ``a``

    Raises
    ------
    TypeError
        If ``a`` is not numeric.

    Complexity
    ----------
    Time O(n log n) per slice, space O(n).
    """
    arr = _as_numeric(a)
    if not descending:
        return np.sort(arr, axis=axis, kind="stable")
    idx = stable_argsort(arr, axis=axis, descending=True)
    return np.take_along_axis(arr, idx, axis=axis)


def multikey_sort(X, keys, ascending=True, return_indices=False):
    """Sort the rows of a 2-D array by several columns.

    Parameters
    ----------
    X : array_like, shape (n_rows, n_cols)
        Numeric 2-D array.
    keys : sequence of int
        Column indices in priority order; ``keys[0]`` is the primary key.
    ascending : bool or sequence of bool, default True
        Sort direction for all keys, or one flag per key.
    return_indices : bool, default False
        Also return the row order if True.

    Returns
    -------
    X_sorted : ndarray, shape (n_rows, n_cols)
    order : ndarray of int, shape (n_rows,)
        Only returned if ``return_indices`` is True.

    Raises
    ------
    TypeError
        If ``X`` is not numeric.
    ValueError
        If ``X`` is not 2-D, ``keys`` is empty, a key is out of range,
        or ``ascending`` has the wrong length.

    Complexity
    ----------
    Time O(m * n log n) for m keys, space O(m * n).
    """
    X = _as_numeric(X, "X")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    keys = list(keys)
    if len(keys) == 0:
        raise ValueError("keys must contain at least one column index")
    n_cols = X.shape[1]
    if any(not -n_cols <= k < n_cols for k in keys):
        raise ValueError(f"keys {keys} out of range for {n_cols} columns")

    if isinstance(ascending, (bool, np.bool_)):
        asc = np.full(len(keys), bool(ascending))
    else:
        asc = np.asarray(ascending, dtype=bool)
        if asc.shape != (len(keys),):
            raise ValueError("ascending must be a bool or one bool per key")

    cols = X[:, keys].astype(float).T                     # (m, n_rows)
    cols = np.where(asc[:, None], cols, -cols)            # flip descending keys
    # np.lexsort treats the LAST key as primary, so reverse the priority.
    order = np.lexsort(cols[::-1])
    X_sorted = X[order]
    return (X_sorted, order) if return_indices else X_sorted


def topk(values, k, largest=True, return_indices=True, axis=-1):
    """Select the k largest (or smallest) elements along an axis.

    Uses ``np.argpartition`` (average O(n)) and then sorts only the
    k selected elements. Output is ordered best-first; equal values are
    ordered by their original index. NaNs are treated as missing and are
    selected only after every finite value.

    Parameters
    ----------
    values : array_like, shape (..., n)
        Numeric input array.
    k : int
        Number of elements to select, ``0 <= k <= n``.
    largest : bool, default True
        Select the largest values if True, otherwise the smallest.
    return_indices : bool, default True
        Also return the indices of the selected elements.
    axis : int, default -1
        Axis to select along.

    Returns
    -------
    top_values : ndarray, shape (..., k)
        Selected values with ``axis`` replaced by length k.
    top_indices : ndarray of int, shape (..., k)
        Positions in ``values`` along ``axis``. Only if ``return_indices``.

    Raises
    ------
    TypeError
        If ``values`` is not numeric or ``k`` is not an integer.
    ValueError
        If ``values`` is 0-d or ``k`` is outside ``[0, n]``.

    Complexity
    ----------
    Time O(n + k log k) per slice, space O(n).
    """
    arr = _as_numeric(values, "values")
    if arr.ndim == 0:
        raise ValueError("values must have at least one dimension")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    arr = np.moveaxis(arr, axis, -1)
    n = arr.shape[-1]
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, {n}], got {k}")

    # Score so that "best" is always the smallest; NaN becomes worst (+inf).
    score = arr.astype(float)
    if largest:
        score = -score
    score = np.where(np.isnan(score), np.inf, score)

    if k == 0:
        idx = np.empty(arr.shape[:-1] + (0,), dtype=np.intp)
    else:
        part = np.argpartition(score, k - 1, axis=-1)[..., :k]
        part_score = np.take_along_axis(score, part, axis=-1)
        # Order selected items by score, then by original index for ties.
        order = np.lexsort((part, part_score), axis=-1)
        idx = np.take_along_axis(part, order, axis=-1)

    top = np.take_along_axis(arr, idx, axis=-1)
    top, idx = np.moveaxis(top, -1, axis), np.moveaxis(idx, -1, axis)
    return (top, idx) if return_indices else top


def quickselect(a, k, seed=None):
    """Return the k-th smallest element (0-based) using quickselect.

    Educational implementation: each iteration picks a random pivot and
    splits the remaining candidates into less / equal / greater groups
    with vectorised boolean masks, then keeps only the group containing
    the answer. The three-way split makes all-equal input finish in one
    iteration.

    Parameters
    ----------
    a : array_like, shape (n,)
        Numeric 1-D array without NaNs.
    k : int
        Order statistic to return, ``0 <= k < n`` (0 = minimum).
        Negative k counts from the end (-1 = maximum).
    seed : int or None, default None
        Seed for the pivot generator, for reproducible runs.

    Returns
    -------
    value : scalar
        Same value as ``np.sort(a)[k]``.

    Raises
    ------
    TypeError
        If ``a`` is not numeric or ``k`` is not an integer.
    ValueError
        If ``a`` is not 1-D, is empty, contains NaN, or ``k`` is out of range.

    Complexity
    ----------
    Time O(n) on average, O(n^2) worst case. Space O(n).
    """
    arr = _as_numeric(a)
    if arr.ndim != 1:
        raise ValueError(f"a must be 1-D, got shape {arr.shape}")
    n = arr.size
    if n == 0:
        raise ValueError("quickselect of an empty array is undefined")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if not -n <= k < n:
        raise ValueError(f"k must be in [{-n}, {n - 1}], got {k}")
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError("a contains NaN; remove or impute NaNs first")

    k = int(k) % n
    rng = np.random.default_rng(seed)
    candidates = arr
    while True:
        pivot = candidates[rng.integers(candidates.size)]
        less = candidates[candidates < pivot]
        n_equal = np.count_nonzero(candidates == pivot)
        if k < less.size:
            candidates = less
        elif k < less.size + n_equal:
            return pivot
        else:
            k -= less.size + n_equal
            candidates = candidates[candidates > pivot]


def binary_search(sorted_array, x, side="left", check_sorted=False):
    """Find insertion indices of ``x`` in a sorted 1-D array.

    Parameters
    ----------
    sorted_array : array_like, shape (n,)
        Numeric array sorted in ascending order.
    x : scalar or array_like, shape (m,) or any shape
        Value(s) to search for.
    side : {'left', 'right'}, default 'left'
        'left' returns the first valid insertion point, 'right' the last.
    check_sorted : bool, default False
        Verify that ``sorted_array`` is ascending (costs O(n)).

    Returns
    -------
    index : int or ndarray of int, same shape as ``x``
        Insertion index that keeps the array sorted.
    found : bool or ndarray of bool, same shape as ``x``
        True where ``x`` is present in ``sorted_array``. NaN is never found.

    Raises
    ------
    TypeError
        If inputs are not numeric.
    ValueError
        If ``sorted_array`` is not 1-D, ``side`` is invalid, or
        ``check_sorted`` is True and the array is not sorted.

    Complexity
    ----------
    Time O(m log n), space O(m).
    """
    arr = _as_numeric(sorted_array, "sorted_array")
    xs = _as_numeric(x, "x")
    if arr.ndim != 1:
        raise ValueError(f"sorted_array must be 1-D, got shape {arr.shape}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if check_sorted and arr.size > 1 and np.any(arr[1:] < arr[:-1]):
        raise ValueError("sorted_array is not sorted in ascending order")

    idx = np.searchsorted(arr, xs, side=side)
    if arr.size == 0:
        found = np.zeros(xs.shape, dtype=bool)
    else:
        # The match sits at idx for 'left', at idx-1 for 'right'.
        probe = idx if side == "left" else idx - 1
        safe = np.clip(probe, 0, arr.size - 1)
        found = (probe >= 0) & (probe < arr.size) & (arr[safe] == xs)

    if xs.ndim == 0:
        return int(idx), bool(found)
    return idx, found
