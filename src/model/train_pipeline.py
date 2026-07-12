from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Optional

import joblib
import numpy as np
import json

class TrainModel:

    """
    What:
        Trains and save ML model for pose classification.

    Input:
        dataset NPZ (features + labels)

    Process:
        1. Load features and labels.
        2. Split into training and testing sets.
        3. Initialize the ML algorithm.
        4. Train the model using fit().
        5. Evaluate performance on the test set.
        6. Save the trained model.

    Output:
        pose_classifier.joblib
    """

    """
    Variables:
        self.dataset_file_path
         self.rdnmodel
    """
    def __init__(self) -> None:

        #Initiate the root folder
        self.ROOT_FOLDER = Path(__file__).resolve().parents[2]

        #Initiate the CONFIG folder.
        self.CONFIG_FOLDER = self.ROOT_FOLDER / "config.json"

        #Load the config file.
        with open(self.CONFIG_FOLDER, 'r') as file:
            self.config = json.load(file)

        #Load the dataset folder using config file.
        self.dataset_file_path = self.ROOT_FOLDER / self.config['dataset']['path']

        #load the RandomForestClassifier model.
        self.rdnmodel = RandomForestClassifier(n_estimators = 200, random_state = 42)
    
    #Training the model. | Return: predictions
    def train_and_predict_model(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray) -> Optional[np.ndarray]:
       
        #Train the model.
        self.rdnmodel.fit(X_train, y_train)

        #Prediction from dataset.
        predict = self.rdnmodel.predict(X_test)

        return predict

    #Get the accuracy score of the model. | Return: accurary score
    def get_accuracy_score(self, predictions: np.ndarray, y_test: np.ndarray) -> np.float64:

        #Get the accuracy score.
        accuracy = accuracy_score(y_test, predictions)

        return accuracy

    #Save the train model. | Return: None
    def save_trained_model(self, model: RandomForestClassifier) -> None:
        
        #Load the trained models path using config file.
        self.trained_models = self.ROOT_FOLDER / self.config['trained_models']['path'] / "gesture_classifier.joblib"

        #Save the model.
        joblib.dump(model, self.trained_models)

    
    """
    MAIN PROGRAM ↓
    """
    def train_and_save_model(self) -> None:

        """
        PIPELINE:
            load_dataset → Load landmarks and labels from NPZ file
                ↓
            split_dataset → Split the dataset for train 80% and test 20%
                ↓
            train_and_predict_model → Train and predict models
                ↓
            save_trained_model → Save trained model.
        """
        
        #Load the landmarks and labels from NPZ.
        X_landmarks, y_labels = self.load_dataset(self.dataset_file_path)

        #Split dataset into train and test.
        X_train, X_test, y_train, y_test = self.split_dataset(X_landmarks, y_labels)

        #Train models and give predictions.
        predictions = self.train_and_predict_model(X_train, X_test, y_train)

        #Save trained model.
        self.save_trained_model(self.rdnmodel)


