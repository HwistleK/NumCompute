"""Basic descriptive statistics."""

import numpy as np

__all__ = [
    "mean",
    "median",
    "std",
    "minimum",
    "maximum",
    "histogram",
    "quantiles",
]


def _as_numeric(data):
    """Convert data to a numeric NumPy array."""
    arr = np.asarray(data)
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {arr.dtype}")
    return arr


def mean(data, axis=None):
    """Calculate the mean, ignoring NaN values."""
    return np.nanmean(_as_numeric(data), axis=axis)


def median(data, axis=None):
    """Calculate the median, ignoring NaN values."""
    return np.nanmedian(_as_numeric(data), axis=axis)


def std(data, axis=None, ddof=0):
    """Calculate the standard deviation, ignoring NaN values."""
    return np.nanstd(_as_numeric(data), axis=axis, ddof=ddof)


def minimum(data, axis=None):
    """Find the minimum, ignoring NaN values."""
    return np.nanmin(_as_numeric(data), axis=axis)


def maximum(data, axis=None):
    """Find the maximum, ignoring NaN values."""
    return np.nanmax(_as_numeric(data), axis=axis)


def histogram(data, bins=10, range=None):
    """Create a histogram after removing NaN values."""
    arr = _as_numeric(data)
    valid = arr[~np.isnan(arr)]
    return np.histogram(valid, bins=bins, range=range)


def quantiles(data, q, axis=None):
    """Calculate quantiles between 0 and 1, ignoring NaN values."""
    arr = _as_numeric(data)
    q_arr = np.asarray(q, dtype=float)
    if np.isnan(q_arr).any() or np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be in the interval [0, 1]")
    return np.nanquantile(arr, q_arr, axis=axis)
