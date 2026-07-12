from sklearn.base import ClassifierMixin
from src.model.training_model_helpers.model_trainer.trainer_validator import _validate_input
import numpy as np

"""
What:
    Train the selected machine learning model using
    prepared landmark datasets.

Responsibilities:
    - Validate training inputs.
    - Fit the model using training data.
    - Return the trained model.

Input:
    X_train: np.ndarray
        Training features containing normalized landmarks.

    y_train: np.ndarray
        Training labels corresponding to each sample.

Process:
    1. Validate X_train and y_train.
    2. Train the selected model.
    3. Return the fitted model.

Output:
    model:
        A trained machine learning classifier.

Failure Conditions:
    - X_train or y_train is not a numpy array.
    - Training data is empty.
    - Feature and label sizes do not match.
    - Model training fails.

Invariants:
    - Returned model must be fitted.
"""

#Training the model. | Return: model
def model_trainer(X_train: np.ndarray, y_train: np.ndarray, model: ClassifierMixin) -> ClassifierMixin:
    
    _validate_input(X_train, y_train, model)

    #Train the model and try to catch some error while training.
    try:
        model.fit(X_train,y_train)
    except Exception as error:
        raise RuntimeError("Model training failed.") from error

    return model