import pytest
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from src.model.training_model_helpers.save_model.save_trained_model import save_trained_model


"""
What:
    Test the save model module.

Responsibilities:
    Validate that the save model module:
        - Saves fitted machine learning models.
        - Creates pose_classifier.joblib file.
        - Rejects invalid models.
        - Rejects invalid paths.
        - Handles saving failures.

Input:
    trained_model:
        ClassifierMixin
        Previously trained machine learning model.

    trained_models_path:
        Path object pointing to trained_models folder.

Process:
    Create synthetic landmark dataset.
    Train classifier.
    Save model using save_trained_model().
    Validate saved model file.

Output:
    pose_classifier.joblib:
        Saved trained model.

Failure Conditions:
    - Model is not fitted.
    - Path does not exist.
    - Saving fails.

Invariants:
    - Saved file must exist.
    - Model must be fitted before saving.
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

    #Create labels.
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

    #Train model.
    model.fit(
        X_train,
        y_train
    )

    return model



#NOTE: SUCCESS TEST ↓


#Test saving valid trained model. | Return: joblib file
def test_save_trained_model_valid_input(tmp_path: Path):

    #Create trained model.
    model = create_trained_model()

    #Create trained models folder.
    trained_models_path = tmp_path / "trained_models"

    trained_models_path.mkdir()


    #Save model.
    result = save_trained_model(
        model,
        trained_models_path
    )


    #Check return.
    assert result is None


    #Check saved file exists.
    model_file = (
        trained_models_path /
        "pose_classifier.joblib"
    )

    assert model_file.exists()



#NOTE: FAILURE TEST ↓


#Test invalid model type. | Expected: TypeError
def test_save_trained_model_invalid_model():

    #Create invalid model.
    model = "invalid_model"

    trained_models_path = Path(
        "trained_models"
    )

    #Expect validation failure.
    with pytest.raises(TypeError):
        save_trained_model(
            model,
            trained_models_path
        )



#Test unfitted model. | Expected: ValueError
def test_save_trained_model_unfitted_model(tmp_path: Path):

    #Create unfitted model.
    model = RandomForestClassifier()

    #Create folder.
    trained_models_path = tmp_path / "trained_models"

    trained_models_path.mkdir()


    #Expect validation failure.
    with pytest.raises(ValueError):
        save_trained_model(
            model,
            trained_models_path
        )



#Test invalid path. | Expected: FileNotFoundError
def test_save_trained_model_invalid_path():

    #Create trained model.
    model = create_trained_model()


    #Invalid directory.
    trained_models_path = Path(
        "invalid_folder"
    )


    #Expect validation failure.
    with pytest.raises(FileNotFoundError):
        save_trained_model(
            model,
            trained_models_path
        )



#Test save failure. | Expected: RuntimeError
def test_save_trained_model_save_failure(tmp_path: Path):

    #Create trained model.
    model = create_trained_model()


    #Create file instead of folder.
    invalid_path = tmp_path / "trained_models"

    invalid_path.write_text(
        "This is a file, not a directory."
    )


    #Expect save failure.
    with pytest.raises(Exception):
        save_trained_model(
            model,
            invalid_path
        )