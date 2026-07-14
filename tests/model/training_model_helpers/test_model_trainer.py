import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.model.training_model_helpers.model_trainer.model_trainer import model_trainer
from src.model.schemas.dataset_schema import DatasetSplit


"""
What:
    Test the model trainer module.

Responsibilities:
    Validate that the model trainer:
        - Trains valid machine learning models.
        - Returns fitted classifier models.
        - Rejects invalid training inputs.
        - Handles model training failures.

Input:
    DatasetSplit:
        - X_train
            Normalized hand landmarks.
            Shape: (samples, 63)

        - y_train
            Pose labels.
            Shape: (samples,)

    model:
        Machine learning classifier.

Process:
    Create synthetic normalized landmark dataset.
    Create classifier model.
    Train model using model_trainer().
    Validate returned model.

Output:
    model:
        Fitted machine learning classifier.

Failure Conditions:
    - X_train is not numpy array.
    - y_train is not numpy array.
    - Dataset is empty.
    - Dataset feature size is incorrect.
    - Model training fails.

Invariants:
    - Returned model must be fitted.
    - X_train must contain 63 landmark features.
"""


#NOTE: TEST HELPER ↓


#Create synthetic dataset. | Return: DatasetSplit
def create_dataset():

    #Create normalized hand landmarks.
    #21 landmarks × x,y,z = 63 features
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

    return DatasetSplit(
        X_train=X_train,
        X_test=X_train,
        y_train=y_train,
        y_test=y_train
    )


#NOTE: SUCCESS TEST ↓


#Test valid model training. | Return: fitted model
def test_model_trainer_valid_input():

    #Create dataset.
    dataset = create_dataset()

    #Create classifier.
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    #Train model.
    trained_model = model_trainer(
        dataset,
        model
    )

    #Check returned model.
    assert trained_model is model

    #Check model is fitted.
    check_is_fitted(
        trained_model
    )


#NOTE: FAILURE TEST ↓


#Test invalid X_train type. | Expected: TypeError
def test_model_trainer_invalid_X_train_type():

    dataset = create_dataset()

    #Replace X_train with invalid type.
    dataset.X_train = [
        [0.1, 0.2, 0.3]
    ]

    model = RandomForestClassifier()

    #Expect validation failure.
    with pytest.raises(TypeError):
        model_trainer(
            dataset,
            model
        )



#Test invalid y_train type. | Expected: TypeError
def test_model_trainer_invalid_y_train_type():

    dataset = create_dataset()

    #Replace y_train with invalid type.
    dataset.y_train = [
        "open_hand",
        "fist"
    ]

    model = RandomForestClassifier()

    #Expect validation failure.
    with pytest.raises(TypeError):
        model_trainer(
            dataset,
            model
        )



#Test empty training dataset. | Expected: ValueError
def test_model_trainer_empty_dataset():

    dataset = DatasetSplit(
        X_train=np.array([]),
        X_test=np.array([]),
        y_train=np.array([]),
        y_test=np.array([])
    )

    model = RandomForestClassifier()

    #Expect validation failure.
    with pytest.raises(ValueError):
        model_trainer(
            dataset,
            model
        )



#Test invalid landmark feature size. | Expected: ValueError
def test_model_trainer_invalid_feature_size():

    #Create invalid feature dataset.
    #Expected: 63 features.
    X_train = np.array(
        [
            np.random.rand(3),
            np.random.rand(3),
            np.random.rand(3),
            np.random.rand(3)
        ]
    )

    y_train = np.array(
        [
            "open_hand",
            "open_hand",
            "fist",
            "fist"
        ]
    )

    dataset = DatasetSplit(
        X_train=X_train,
        X_test=X_train,
        y_train=y_train,
        y_test=y_train
    )

    model = RandomForestClassifier()

    #Expect validation failure.
    with pytest.raises(ValueError):
        model_trainer(
            dataset,
            model
        )



#Test invalid model object. | Expected: TypeError
def test_model_trainer_invalid_model():

    dataset = create_dataset()

    #Object without fit() method.
    model = "invalid_model"

    #Expect validation failure.
    with pytest.raises(TypeError):
        model_trainer(
            dataset,
            model
        )



#Test model training failure. | Expected: RuntimeError
def test_model_trainer_training_failure():

    class BrokenModel:

        #Force training failure.
        def fit(self, X, y):
            raise Exception("Training failed")


    dataset = create_dataset()

    model = BrokenModel()

    #Expect training failure.
    with pytest.raises(RuntimeError):
        model_trainer(
            dataset,
            model
        )