from pathlib import Path
import numpy as np
import os

"""
What:
    Handle All validation for the dataset_loader module.
"""

def _validate_path(dataset_file_path: Path):

    if not isinstance(dataset_file_path, Path):
        raise TypeError("dataset_file_path must be a pathlib.Path.")
    
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file_path}")

    if not dataset_file_path.is_file():
        raise ValueError( f"Expected a file: {dataset_file_path}")
    
    if dataset_file_path.suffix.lower() != ".npz".lower():
        raise ValueError( f"Expected a '.npz' file, got '{dataset_file_path.suffix}'.")
    
    if dataset_file_path.stat().st_size == 0:
        raise ValueError(f"File is empty: {dataset_file_path}")
    
    if not os.access(dataset_file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: {dataset_file_path}")

def _validate_X_y(X: np.ndarray, y: np.ndarray) -> None:

    # Validate types
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be np.ndarray, got {type(X)}")

    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be np.ndarray, got {type(y)}")

    # Validate empty dataset
    if X.size == 0:
        raise ValueError("X dataset is empty.")

    if y.size == 0:
        raise ValueError("y dataset is empty.")

    # Validate sample count
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y sample count mismatch: "
            f"X has {X.shape[0]}, y has {y.shape[0]}"
        )

    # Validate X dimensions
    if X.ndim != 2:
        raise ValueError( f"X must be 2D array. Got shape {X.shape}")

    # Validate y dimensions
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array. Got shape {y.shape}")

    # Validate numeric features
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(f"X must contain numeric values. Got {X.dtype}")


    # Validate labels are not missing
    if np.any(y != y):
        raise ValueError("y contains missing labels.")
