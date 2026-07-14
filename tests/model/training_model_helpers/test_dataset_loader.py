from pathlib import Path
import numpy as np
import pytest

from src.model.training_model_helpers.dataset_loader.dataset_loader import load_dataset
from src.model.schemas.dataset_schema import Dataset


"""
What:
    Test the dataset loader module.

Responsibilities:
    Validate that the dataset loader:
        - Loads valid .npz dataset files.
        - Returns correct Dataset dataclass.
        - Rejects invalid dataset files.
        - Rejects missing required keys.

Input:
    Temporary .npz dataset files.

Process:
    Create synthetic dataset.
    Save dataset as .npz file.
    Load dataset using load_dataset().
    Validate output and failure conditions.

Output:
    Dataset:
        - Contains X landmarks.
        - Contains y labels.
        - Type must be Dataset dataclass.

Failure Conditions:
    Missing X key.
    Missing y key.
    Invalid dataset path.

Invariants:
    Loader must always return Dataset.
"""


#NOTE: SUCCESS TEST ↓


#Test loading valid dataset file. | Return: Dataset
def test_load_dataset_valid_file(tmp_path: Path):

    #Create synthetic landmarks dataset.
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    )

    #Create synthetic labels.
    y = np.array(
        [
            "open_hand",
            "fist"
        ]
    )

    #Create temporary dataset file.
    dataset_path = tmp_path / "hand_pose_dataset.npz"

    np.savez(
        dataset_path,
        X=X,
        y=y
    )

    #Load dataset.
    dataset = load_dataset(dataset_path)

    #Check returned type.
    assert isinstance(dataset, Dataset)

    #Check X output.
    assert np.array_equal(
        dataset.X_landmarks,
        X
    )

    #Check y output.
    assert np.array_equal(
        dataset.y_labels,
        y
    )


#NOTE: FAILURE TEST ↓


#Test missing dataset file. | Expected: FileNotFoundError
def test_load_dataset_file_not_exist():

    #Create invalid dataset path.
    dataset_path = Path(
        "invalid_dataset.npz"
    )

    #Expect file validation failure.
    with pytest.raises(FileNotFoundError):
        load_dataset(dataset_path)



#Test dataset missing X key. | Expected: ValueError
def test_load_dataset_missing_X_key(tmp_path: Path):

    #Create dataset with only y.
    dataset_path = tmp_path / "missing_X.npz"

    y = np.array(
        [
            "open_hand"
        ]
    )

    np.savez(
        dataset_path,
        y=y
    )

    #Expect missing key failure.
    with pytest.raises(ValueError):
        load_dataset(dataset_path)



#Test dataset missing y key. | Expected: ValueError
def test_load_dataset_missing_y_key(tmp_path: Path):

    #Create dataset with only X.
    dataset_path = tmp_path / "missing_y.npz"

    X = np.array(
        [
            [0.1, 0.2, 0.3]
        ]
    )

    np.savez(
        dataset_path,
        X=X
    )

    #Expect missing key failure.
    with pytest.raises(ValueError):
        load_dataset(dataset_path)



#Test mismatched X and y samples. | Expected: ValueError
def test_load_dataset_invalid_X_y_shape(tmp_path: Path):

    #Create invalid dataset.
    #X has 2 samples but y has 1 label.
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    )

    y = np.array(
        [
            "open_hand"
        ]
    )

    dataset_path = tmp_path / "invalid_shape.npz"

    np.savez(
        dataset_path,
        X=X,
        y=y
    )

    #Expect validation failure.
    with pytest.raises(ValueError):
        load_dataset(dataset_path)