from sklearn.metrics import accuracy_score

from src.model.training_model_helpers.model_accuracy.accuracy_validator import (_validate_input, _validate_output)

import numpy as np

"""
What:
    Calculate the prediction accuracy of the trained
    machine learning model.

Responsibilities:
    - Validate prediction results.
    - Compute the accuracy score.
    - Return the accuracy score.

Input:
    predictions: np.ndarray
        Labels predicted by the trained model.

    y_test: np.ndarray
        Ground truth labels used for evaluation.

Process:
    1. Validate predictions and y_test.
    2. Calculate the accuracy score.
    3. Validate output.
    4. Return the accuracy score.

Output:
    accuracy:
        float
        Prediction accuracy between 0.0 and 1.0.

Failure Conditions:
    - predictions is not numpy array.
    - y_test is not numpy array.
    - Arrays are empty.
    - Number of samples do not match.

Invariants:
    - Returned accuracy must be float.
    - Returned accuracy must be between 0.0 and 1.0.
"""


#Calculate prediction accuracy. | Return: accuracy
def get_accuracy_score(predictions: np.ndarray,y_test: np.ndarray) -> float:

    _validate_input(predictions, y_test)

    #Calculate prediction accuracy.
    accuracy = accuracy_score(y_test, predictions)

    _validate_output(accuracy)

    return accuracy