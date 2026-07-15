# src/model/prediction_model.py

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import ClassifierMixin


class PredictionModel:
    """
    Predicts hand poses using a trained classifier.

    Input:
        Normalized hand landmarks.
        Shape: (63,)

    Process:
        Converts the landmarks into the expected
        scikit-learn input shape and predicts the pose.

    Output:
        Predicted pose label.

    Failure Conditions:
        Raises FileNotFoundError
            If the trained model does not exist.

        Raises ValueError
            If the feature vector is invalid.
    """

    ROOT_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = ROOT_DIR / "trained_models" / "pose_classifier.joblib"

    def __init__(self) -> None:
        """
        Why:
            Loads the trained classifier once during
            object construction.
        """

        self._model: ClassifierMixin = joblib.load(self.MODEL_PATH)

    def predict(self, features: np.ndarray) -> str:
        """
        Predict a hand gesture.

        Input:
            features
                Shape: (63,)

        Return:
            Predicted pose label.
        """

        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy ndarray.")

        if features.ndim != 1:
            raise ValueError("features must have shape (63,).")

        prediction = self._model.predict(features.reshape(1, -1))

        return str(prediction[0])