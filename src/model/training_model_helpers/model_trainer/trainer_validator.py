import numpy as np
from sklearn.base import ClassifierMixin

"""
What:
    Handle All validation for the dataset_splitter module.
"""

#NOTE: CONSTANTS
EXPECTED_LANDMARKS = 63
MINIMUM_SAMPLES = 2

def _validate_input(X_train: np.ndarray, y_train: np.ndarray, model: ClassifierMixin):
    
    #Check data type.
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
        raise TypeError("Training dataset must be Numpy array.")
    
    #Check if it has fit attribute.
    if not hasattr(model, "fit"):
        raise TypeError("Model must implement fit()")  
    
    #Check if it's empty
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training dataset must not empty.")
    
    #Check dimensions.
    if X_train.ndim != 2:
        raise ValueError("X(landmarks) training dataset must be 2D array.")
    if y_train.ndim != 1:
        raise ValueError("y(labels) training dataset must be 1D array.")
    
    #Check number of samples.
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples.")
    if X_train.shape[0] < MINIMUM_SAMPLES:
        raise ValueError("Training requires multiple samples.")
    if X_train.shape[1] != EXPECTED_LANDMARKS:
        raise ValueError("Expected normalized hand landmarks with 63 features.")
    
    #Check label validation.
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training requires at least two classes.")
