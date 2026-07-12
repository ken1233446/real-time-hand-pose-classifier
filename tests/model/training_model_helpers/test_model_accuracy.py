import numpy as np
import pytest

from src.model.training_model_helpers.model_accuracy.model_accuracy import get_accuracy_score
from src.model.training_model_helpers.model_accuracy.accuracy_validator import (_validate_input, _validate_output)

"""
What:
    Test model accuracy module.

Responsibilities:
    Test input validation.
    Test output validation.
    Test accuracy calculation.

Input:
    Synthetic prediction and test labels.

Process:
    Validate invalid inputs.
    Calculate prediction accuracy.
    Validate returned accuracy score.

Output:
    All tests should pass.

Invariants:
    Accuracy score must be between 0.0 and 1.0.
"""


#NOTE: TEST INPUT VALIDATION ↓

def test_validate_input_accept_valid_dataset():

    #Create fake labels.
    predictions = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1, 0, 1])

    #Validate input.
    _validate_input(predictions, y_test)

def test_validate_input_reject_non_numpy_predictions():

    #Create fake labels.
    predictions = [0, 1, 0]
    y_test = np.array([0, 1, 0])

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(predictions, y_test)

def test_validate_input_reject_non_numpy_y_test():

    #Create fake labels.
    predictions = np.array([0, 1, 0])
    y_test = [0, 1, 0]

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(predictions, y_test)

def test_validate_input_reject_empty_predictions():

    #Create fake labels.
    predictions = np.array([])
    y_test = np.array([0, 1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(predictions, y_test)

def test_validate_input_reject_empty_y_test():

    #Create fake labels.
    predictions = np.array([0, 1])
    y_test = np.array([])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(predictions, y_test)

def test_validate_input_reject_invalid_prediction_dimension():

    #Create fake labels.
    predictions = np.array([[0],[1]])
    y_test = np.array([0, 1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(predictions, y_test)

def test_validate_input_reject_invalid_y_test_dimension():

    #Create fake labels.
    predictions = np.array([0, 1])
    y_test = np.array([[0],[1]])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(predictions, y_test)

def test_validate_input_reject_different_sample_count():

    #Create fake labels.
    predictions = np.array([0, 1, 0])
    y_test = np.array([0, 1])

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(predictions, y_test)

#NOTE: TEST OUTPUT VALIDATION ↓

def test_validate_output_accept_float():

    #Create fake accuracy.
    accuracy = 1.0

    #Validate output.
    _validate_output(accuracy)

def test_validate_output_reject_invalid_type():

    #Create fake accuracy.
    accuracy = "1.0"

    #Check error.
    with pytest.raises(TypeError):
        _validate_output(accuracy)

def test_validate_output_reject_invalid_range():

    #Create fake accuracy.
    accuracy = 1.5

    #Check error.
    with pytest.raises(ValueError):
        _validate_output(accuracy)

#NOTE: TEST MAIN FUNCTION ↓

def test_get_accuracy_score_return_float():

    #Create fake labels.
    predictions = np.array([
        0,
        1,
        0,
        1
    ])

    y_test = np.array([
        0,
        1,
        0,
        1
    ])

    #Calculate accuracy.
    accuracy = get_accuracy_score(predictions, y_test)

    #Check output.
    assert isinstance(accuracy,(float, np.floating))

def test_get_accuracy_score_return_correct_accuracy():

    #Create fake labels.
    predictions = np.array([
        0,
        1,
        1,
        0
    ])

    y_test = np.array([
        0,
        1,
        0,
        0
    ])

    #Calculate accuracy.
    accuracy = get_accuracy_score(predictions, y_test)

    #Check output.
    assert accuracy == 0.75

def test_get_accuracy_score_return_one_for_perfect_prediction():

    #Create fake labels.
    predictions = np.array([
        0,
        1,
        0,
        1
    ])

    y_test = np.array([
        0,
        1,
        0,
        1
    ])

    #Calculate accuracy.
    accuracy = get_accuracy_score(predictions, y_test)

    #Check output.
    assert accuracy == 1.0