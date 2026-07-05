#Hand detector Module
from detection.hand_detector import HandDetector
#Normalizer Module
from feature.normalizer import Normalizer

import numpy as np
from typing import List

class LandmarkPipeline:

    #Initiates Modules.
    def __init__(self, detector: HandDetector, normalizer: Normalizer) -> None:
        
        #Objects Initializations. 
        self.detector = detector
        self.normalizer = normalizer

    #Get the 21 Normalized Landmarks from live frame. Return: 21 Normalized Landmarks.
    def extract(self, rgb_frames: np.ndarray) -> List[float]:
        
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

        #Process RGB frames. Return: UnNormalized 21 Hand landmarks.
        unnormalized_landmarks = self.detector.detect_hands(rgb_frames)

        #Convert Normalized the landmarks. Return: Normalized landmarks.
        normalized_landmarks = self.normalizer.transform(unnormalized_landmarks)

        return normalized_landmarks