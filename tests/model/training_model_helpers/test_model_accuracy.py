import numpy as np
import pytest

from src.model.training_model_helpers.model_accuracy.model_accuracy import get_accuracy_score


"""
What:
    Test the model accuracy module.

Responsibilities:
    Validate that the accuracy calculator:
        - Calculates correct prediction accuracy.
        - Returns float accuracy score.
        - Rejects invalid prediction inputs.
        - Rejects mismatched datasets.

Input:
    predictions:
        np.ndarray
        Model predicted labels.

    y_test:
        np.ndarray
        Actual labels.

Process:
    Create synthetic predictions.
    Create synthetic ground truth labels.
    Calculate accuracy score.
    Validate output and failure conditions.

Output:
    accuracy:
        float
        Value between 0.0 and 1.0.

Failure Conditions:
    predictions is not np.ndarray.
    y_test is not np.ndarray.
    Empty arrays.
    Different sample count.

Invariants:
    Returned accuracy must be float.
    Accuracy must be between 0.0 and 1.0.
"""


#NOTE: SUCCESS TEST ↓


#Test correct accuracy calculation. | Return: float
def test_get_accuracy_score_valid_input():

    #Create predicted labels.
    predictions = np.array(
        [
            "open_hand",
            "fist",
            "point",
            "fist"
        ]
    )

    #Create actual labels.
    y_test = np.array(
        [
            "open_hand",
            "fist",
            "point",
            "open_hand"
        ]
    )

    #Calculate accuracy.
    accuracy = get_accuracy_score(
        predictions,
        y_test
    )

    #Check output type.
    assert isinstance(
        accuracy,
        float
    )

    #Check accuracy value.
    #3 correct predictions out of 4.
    assert accuracy == 0.75


#Test perfect accuracy. | Return: 1.0
def test_get_accuracy_score_perfect_prediction():

    predictions = np.array(
        [
            "open_hand",
            "fist"
        ]
    )

    y_test = np.array(
        [
            "open_hand",
            "fist"
        ]
    )

    accuracy = get_accuracy_score(
        predictions,
        y_test
    )

    assert accuracy == 1.0



#NOTE: FAILURE TEST ↓


#Test invalid prediction type. | Expected: TypeError
def test_get_accuracy_score_invalid_prediction_type():

    #Create invalid prediction.
    predictions = [
        "open_hand",
        "fist"
    ]

    y_test = np.array(
        [
            "open_hand",
            "fist"
        ]
    )

    #Expect validation failure.
    with pytest.raises(TypeError):
        get_accuracy_score(
            predictions,
            y_test
        )



#Test invalid y_test type. | Expected: TypeError
def test_get_accuracy_score_invalid_y_test_type():

    predictions = np.array(
        [
            "open_hand",
            "fist"
        ]
    )

    y_test = [
        "open_hand",
        "fist"
    ]

    #Expect validation failure.
    with pytest.raises(TypeError):
        get_accuracy_score(
            predictions,
            y_test
        )



#Test empty predictions. | Expected: ValueError
def test_get_accuracy_score_empty_predictions():

    predictions = np.array([])

    y_test = np.array(
        [
            "open_hand"
        ]
    )

    #Expect validation failure.
    with pytest.raises(ValueError):
        get_accuracy_score(
            predictions,
            y_test
        )



#Test empty y_test. | Expected: ValueError
def test_get_accuracy_score_empty_y_test():

    predictions = np.array(
        [
            "open_hand"
        ]
    )

    y_test = np.array([])

    #Expect validation failure.
    with pytest.raises(ValueError):
        get_accuracy_score(
            predictions,
            y_test
        )



#Test mismatched sample count. | Expected: ValueError
def test_get_accuracy_score_mismatched_samples():

    predictions = np.array(
        [
            "open_hand",
            "fist"
        ]
    )

    y_test = np.array(
        [
            "open_hand"
        ]
    )

    #Expect validation failure.
    with pytest.raises(ValueError):
        get_accuracy_score(
            predictions,
            y_test
        )