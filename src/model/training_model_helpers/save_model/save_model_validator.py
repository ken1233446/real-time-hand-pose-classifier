from pathlib import Path

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


"""
What:
    Handle all validation for the save model module.
"""


# Validate input.
def _validate_input(
        model: ClassifierMixin,
        trained_models_path: Path
) -> None:


    # Check model type.
    if not isinstance(model, ClassifierMixin):
        raise TypeError(
            "Model must inherit from ClassifierMixin."
        )


    # Check if model is fitted.
    try:
        check_is_fitted(model)

    except NotFittedError as error:
        raise ValueError(
            "Model must be fitted before saving."
        ) from error


    # Check path type.
    if not isinstance(trained_models_path, Path):
        raise TypeError(
            "trained_models_path must be pathlib.Path."
        )


    # Check directory exists.
    if not trained_models_path.exists():
        raise FileNotFoundError(
            "trained_models directory does not exist."
        )


    # Check directory.
    if not trained_models_path.is_dir():
        raise ValueError(
            "trained_models_path must be a directory."
        )



# Validate output.
def _validate_output(
        model_file: Path
) -> None:


    # Check file exists.
    if not model_file.exists():
        raise ValueError(
            "pose_classifier.joblib was not created."
        )


    # Check file type.
    if not model_file.is_file():
        raise ValueError(
            "pose_classifier.joblib must be a file."
        )