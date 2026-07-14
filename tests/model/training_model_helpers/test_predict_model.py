import numpy as np
import pytest

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from src.model.training_model_helpers.predict_model.prediction_model import model_predictor


"""
What:
    Test the prediction model module.

Responsibilities:
    Validate that the prediction model:
        - Predicts labels using trained models.
        - Returns numpy array predictions.
        - Rejects invalid prediction inputs.
        - Handles prediction failures.

Input:
    X:
        np.ndarray
        Normalized hand landmark features.

        Shape:
            (samples, 63)

    trained_model:
        ClassifierMixin
        Previously trained machine learning model.

Process:
    Create synthetic normalized landmarks.
    Train classifier.
    Predict pose labels.
    Validate prediction output.

Output:
    predictions:
        np.ndarray
        Contains predicted pose labels.

Failure Conditions:
    - X is not numpy array.
    - X has incorrect feature size.
    - Model is not fitted.
    - Prediction fails.

Invariants:
    - Returned predictions must be numpy.ndarray.
"""


#NOTE: TEST HELPER ↓


#Create trained model. | Return: fitted classifier
def create_trained_model():

    #Create normalized landmarks.
    #21 landmarks × x,y,z = 63 features.
    X_train = np.array(
        [
            np.random.rand(63),
            np.random.rand(63),
            np.random.rand(63),
            np.random.rand(63)
        ]
    )

    #Create pose labels.
    y_train = np.array(
        [
            "open_hand",
            "open_hand",
            "fist",
            "fist"
        ]
    )

    #Create classifier.
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    #Train classifier.
    model.fit(
        X_train,
        y_train
    )

    return model



#NOTE: SUCCESS TEST ↓


#Test valid prediction. | Return: np.ndarray
def test_model_predictor_valid_input():

    #Create trained model.
    model = create_trained_model()

    #Create prediction input.
    X = np.array(
        [
            np.random.rand(63),
            np.random.rand(63)
        ]
    )

    #Predict labels.
    predictions = model_predictor(
        X,
        model
    )

    #Check output type.
    assert isinstance(
        predictions,
        np.ndarray
    )

    #Check prediction count.
    assert predictions.shape[0] == 2



#NOTE: FAILURE TEST ↓


#Test invalid X type. | Expected: TypeError
def test_model_predictor_invalid_X_type():

    #Create trained model.
    model = create_trained_model()

    #Create invalid X.
    X = [
        [
            0.1,
            0.2,
            0.3
        ]
    ]

    #Expect validation failure.
    with pytest.raises(TypeError):
        model_predictor(
            X,
            model
        )



#Test invalid feature size. | Expected: ValueError
def test_model_predictor_invalid_feature_size():

    #Create trained model.
    model = create_trained_model()

    #Create invalid landmark size.
    #Expected:
    #21 landmarks × x,y,z = 63 features
    X = np.array(
        [
            np.random.rand(3)
        ]
    )

    #Expect validation failure.
    with pytest.raises(ValueError):
        model_predictor(
            X,
            model
        )



#Test unfitted model. | Expected: NotFittedError
def test_model_predictor_unfitted_model():

    #Create unfitted classifier.
    model = RandomForestClassifier()

    X = np.array(
        [
            np.random.rand(63)
        ]
    )

    #Expect validation failure.
    with pytest.raises(Exception):
        model_predictor(
            X,
            model
        )



#Test prediction failure. | Expected: RuntimeError
def test_model_predictor_prediction_failure():

    from sklearn.base import ClassifierMixin, BaseEstimator


    class BrokenModel(ClassifierMixin, BaseEstimator):

        #Create fitted state.
        def fit(self, X, y):

            #Sklearn detects attributes ending with "_".
            self.fitted_ = True

            return self


        #Force prediction failure.
        def predict(self, X):
            raise Exception("Prediction failed")


    model = BrokenModel()

    #Fake training.
    model.fit(
        np.array(
            [
                np.random.rand(63)
            ]
        ),
        np.array(
            [
                "open_hand"
            ]
        )
    )


    X = np.array(
        [
            np.random.rand(63)
        ]
    )


    #Expect prediction failure.
    with pytest.raises(RuntimeError):
        model_predictor(
            X,
            model
        )