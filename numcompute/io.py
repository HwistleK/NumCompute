"""CSV loading with delimiter support and missing value handling: Load data in Numpy arrays
"""
import numpy as np
 
__all__ = ["load_csv"]

def load_csv(filepath, delimiter=",", skiprows=0,  missing_values="", filling_values=np.nan, dtype=float):
    """Load a CSV file into a Numpy array:

    Args:
        filepath (str or path-like): Path to the CSV file to load,
        delimiter (str, default ','): Delimiter used to seperate values,
        skiprows (int, default 0):Number of lines to skip in the beginning of the file,
        missing_values (str, sequence or None, default ''): Elements to recognise as missing values,
        filling_values (scalar, default np.nan): Value used to fill missing occurences,
        dtype (data-type, default float): Data type of the returned Numpy array.

    Raises:
        TypeError: If the filepath is not a valid string or path-like object
        ValueError: If the file cannot be parsed correctly

    Returns:
        arr: ndarray: Loaded numeric data as Numpy arrays
        
    Complexity:
        Time : O(n_rows * n_cols)
        Space : O(n_rows * n_cols)
    """
    
    if not isinstance(filepath, (str, bytes)):
        raise TypeError(f"Filepath must be a string or path-like, got {type(filepath).__name__} instead.")
    try:
        arr = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=skiprows,
            missing_values=missing_values,
            filling_values=filling_values,
            dtype=dtype
        )
    except Exception as e:
        raise ValueError(f"Failed to load CSV file from {filepath!r}: {e}")
    
    return arr