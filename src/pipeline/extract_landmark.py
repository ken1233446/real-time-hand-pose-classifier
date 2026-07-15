#Hand detector Module
from src.detection.hand_detector import HandDetector
#Normalizer Module
from src.feature.landmark_normalizer.normalizer import Normalizer

import numpy as np
from typing import List

class ExtractLandmarkPipeline:
    
    """
    ExtractLandmarkPipeline Contract

    What:
        Coordinate the hand landmark extraction and normalization process.

    Why:
        Convert raw RGB camera frames into normalized landmark features
        that are ready for machine learning models.

    Input:
        detector:
            HandDetector module.

        normalizer:
            Normalizer module.

    Process:
        Receive an RGB frame.
        Detect raw hand landmarks.
        Normalize the detected landmarks.
        Return the normalized landmark vector.

    Output:
        List[float]
            Normalized landmark feature vector.

        None
            Returned when no hand is detected.
    """
    
    #Initiates Modules.
    def __init__(self, detector: HandDetector, normalizer: Normalizer) -> None:
        
        #Objects Initializations. 
        self.detector = detector
        self.normalizer = normalizer

    #Extract normalized landmarks from an RGB frame.| Return: 21 Normalized Landmarks.
    def extract(self, rgb_frames: np.ndarray) -> List[float] | None:
        
        """
        Pipeline:
            rgb_frames (from camera → grab grames → convert BGR to RGB)
                ↓
            hand_detector.py (Extract the 21 landmarks of hands.)
                ↓
            normalizer.py (LandmarkProcesser → Make the landmarks relative to wrist, and proper ratio.)
                ↓
            Output (Extracted normalized landmarks)
        """

        """
        Input:
            rgb_frames:
                RGB Numpy Matrix.

        Process:
            Detect hand landmarks.f
            Check if a hand is detected.
            Normalize the detected landmarks.

        Output:
            List[float]
                Normalized landmark feature vector.

            None
                Returned when no hand is detected.
        """
        #Process RGB frames. Return: UnNormalized 21 Hand landmarks.
        unnormalized_landmarks = self.detector.detect_hands(rgb_frames)

        #No hand detected.
        if unnormalized_landmarks is None:
            return None

        #Normalize the detected landmarks.
        normalized_landmarks = self.normalizer.transform(unnormalized_landmarks)

        return normalized_landmarks