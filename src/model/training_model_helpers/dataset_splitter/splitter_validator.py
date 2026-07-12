import numpy as np

"""
What:
    Handle All validation for the dataset_splitter module.
"""

def _validate_input(X: np.ndarray, y: np.ndarray, config_test_size: float, config_random_state: int):
    
    #Check data types.
    if not isinstance(X, np.ndarray):
        raise TypeError("Landmark (X) must be  Numpy array.")
    if not isinstance(y, np.ndarray):
        raise TypeError("Labels (y) must be  Numpy array")
    if not isinstance(config_test_size, (float,int)):
        raise TypeError("Test size must be float or int.")
    if not isinstance(config_random_state, int):
        raise TypeError("Random state must be int.")
    
    #Check if it's empty.
    if X.size == 0:
        raise ValueError("X cannot be empty.")
    if y.size == 0:
        raise ValueError("y cannot be empty.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    
    #Validate ranges
    if not 0 < config_test_size < 1:
        raise ValueError("Float test_size must be between 0 and 1.")

    if config_random_state < 0:
        raise ValueError("random_state cannot be negative.")
    
    #Check landmark shape.
    if X.ndim != 2:
        raise ValueError("X must be 2D array")
    if y.ndim != 1:
        raise ValueError("y must be 1D array")
    
    #Stratify validation.
    unique, counts = np.unique(y, return_counts=True)
    if np.any(counts < 2):
        raise ValueError( "Each class requires at least 2 samples")

def _validate_output(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    """
    Why loop?
        To avoid repeating check and validate thru loop.
    """
    outputs = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    #Check data type.
    for name, value in outputs.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be numpy.ndarray.")
    
    #Check empty array.
    for name, value in outputs.items():
        if value.size == 0:
            raise ValueError(f"{name} cannot be empty.")