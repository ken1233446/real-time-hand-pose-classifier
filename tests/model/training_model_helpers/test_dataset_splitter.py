import numpy as np
import pytest

from src.model.training_model_helpers.dataset_splitter.dataset_splitter import split_dataset
from src.model.schemas.dataset_schema import Dataset, DatasetSplit


"""
What:
    Test the dataset splitter module.

Responsibilities:
    Validate that the dataset splitter:
        - Splits valid datasets correctly.
        - Returns DatasetSplit dataclass.
        - Keeps X and y relationship.
        - Rejects invalid inputs.

Input:
    Dataset dataclass:
        - X_landmarks
        - y_labels

    test_size:
        Size of test dataset.

    random_state:
        Reproducibility value.

Process:
    Create synthetic dataset.
    Pass dataset into split_dataset().
    Validate output structure.
    Validate failure conditions.

Output:
    DatasetSplit:
        - X_train
        - X_test
        - y_train
        - y_test

Failure Conditions:
    Invalid dataset type.
    Invalid test size.
    Invalid random state.

Invariants:
    Dataset must not be modified.
    Output must always be DatasetSplit.
    X and y sample count must remain equal.
"""


#NOTE: SUCCESS TEST ↓


#Test valid dataset splitting. | Return: DatasetSplit
def test_split_dataset_valid_input():

    #Create synthetic landmarks.
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8]
        ]
    )

    #Create synthetic labels.
    #Need multiple samples per class because stratify=y.
    y = np.array(
        [
            "open_hand",
            "open_hand",
            "fist",
            "fist",
            "point",
            "point"
        ]
    )

    #Create dataset dataclass.
    dataset = Dataset(
        X_landmarks=X,
        y_labels=y
    )

    #Split dataset.
    dataset_split = split_dataset(
        dataset,
        test_size=0.5,
        random_state=42
    )

    #Check returned type.
    assert isinstance(
        dataset_split,
        DatasetSplit
    )

    #Check output type.
    assert isinstance(
        dataset_split.X_train,
        np.ndarray
    )

    assert isinstance(
        dataset_split.X_test,
        np.ndarray
    )

    assert isinstance(
        dataset_split.y_train,
        np.ndarray
    )

    assert isinstance(
        dataset_split.y_test,
        np.ndarray
    )

    #Check total samples are preserved.
    assert len(dataset_split.X_train) + len(dataset_split.X_test) == len(X)

    assert len(dataset_split.y_train) + len(dataset_split.y_test) == len(y)


#NOTE: FAILURE TEST ↓


#Test invalid test size. | Expected: ValueError
def test_split_dataset_invalid_test_size():

    #Create synthetic dataset.
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2]
        ]
    )

    y = np.array(
        [
            "open_hand",
            "open_hand",
            "fist",
            "fist"
        ]
    )

    dataset = Dataset(
        X_landmarks=X,
        y_labels=y
    )

    #Expect validation failure.
    with pytest.raises(ValueError):
        split_dataset(
            dataset,
            test_size=2,
            random_state=42
        )



#Test invalid random state. | Expected: TypeError
def test_split_dataset_invalid_random_state():

    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2]
        ]
    )

    y = np.array(
        [
            "open_hand",
            "open_hand",
            "fist",
            "fist"
        ]
    )

    dataset = Dataset(
        X_landmarks=X,
        y_labels=y
    )

    #Expect validation failure.
    with pytest.raises(TypeError):
        split_dataset(
            dataset,
            test_size=0.5,
            random_state="invalid"
        )



#Test empty dataset. | Expected: ValueError
def test_split_dataset_empty_dataset():

    #Create empty dataset.
    X = np.array([])
    y = np.array([])

    dataset = Dataset(
        X_landmarks=X,
        y_labels=y
    )

    #Expect validation failure.
    with pytest.raises(ValueError):
        split_dataset(
            dataset,
            test_size=0.2,
            random_state=42
        )