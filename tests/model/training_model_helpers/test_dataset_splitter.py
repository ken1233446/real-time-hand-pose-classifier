import numpy as np
import pytest

from src.model.training_model_helpers.dataset_splitter.dataset_splitter import split_dataset
from src.model.training_model_helpers.dataset_splitter.splitter_validator import _validate_input, _validate_output

"""
What:
    Test dataset splitter module.

Responsibilities:
    Test input validation.
    Test output validation.
    Test dataset splitting behavior.

Input:
    Synthetic numpy dataset.

Process:
    Validate invalid inputs.
    Run split_dataset().
    Validate returned outputs.

Output:
    All tests should pass if module behaves correctly.

Invariants:
    Input validation must reject invalid dataset.
    Output must always be numpy.ndarray.
    Dataset split must not modify original dataset.
"""


#NOTE: TEST INPUT VALIDATION ↓

def test_validate_input_accept_numpy_array():
    
    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])

    y = np.array([0,0,1,1])
    #Validate input.
    _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_non_numpy_X():

    #Create fake dataset.
    X = [
        [1,2,3],
        [4,5,6]
    ]
    y = np.array([0,1])

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_non_numpy_y():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6]
    ])
    y = [0,1]

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_empty_X():

    #Create empty dataset.
    X = np.array([])
    y = np.array([0,1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_empty_y():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6]
    ])
    y = np.array([])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_different_sample_size():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])
    y = np.array([0,1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_invalid_X_dimension():

    #Create fake dataset.
    X = np.array([1,2,3,4])
    y = np.array([0,0,1,1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_invalid_y_dimension():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])

    y = np.array([
        [0],
        [0],
        [1],
        [1]
    ])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X, y, 0.25, 42)

def test_validate_input_reject_negative_random_state():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])
    y = np.array([0,0,1,1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X, y, 0.25, -1)

#NOTE: TEST OUTPUT VALIDATION ↓

def test_validate_output_accept_numpy_array():

    #Create fake output.
    X_train = np.array([[1,2,3]])
    X_test = np.array([[4,5,6]])
    y_train = np.array([0])
    y_test = np.array([1])

    #Validate output.
    _validate_output(X_train, X_test, y_train, y_test)

def test_validate_output_reject_empty_output():

    #Create empty output.
    X_train = np.array([])
    X_test = np.array([[4,5,6]])
    y_train = np.array([0])
    y_test = np.array([1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_output(X_train, X_test, y_train, y_test)

#NOTE: TEST MAIN FUNCTION ↓

def test_split_dataset_output_type():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])

    y = np.array([0,0,1,1])

    #Split dataset.
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        0.5,
        42
    )

    #Check output type.
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

def test_split_dataset_preserve_sample_count():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])

    y = np.array([0,0,1,1])
    #Split dataset.
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        0.5,
        42
    )

    #Check sample count.
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

def test_split_dataset_not_modify_original_dataset():

    #Create fake dataset.
    X = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])
    y = np.array([0,0,1,1])
    #Copy original dataset.
    original_X = X.copy()
    original_y = y.copy()

    #Split dataset.
    split_dataset(
        X,
        y,
        0.5,
        42
    )

    #Check original dataset.
    assert np.array_equal(X, original_X)
    assert np.array_equal(y, original_y)