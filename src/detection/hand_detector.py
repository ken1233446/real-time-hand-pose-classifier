import numpy as np
import json
import mediapipe as mp

from pathlib import Path

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark



#Main Function.
class HandDetector:

    """
    HandDetector Contract

    What:
        Detect one or more hands from an RGB frame using MediaPipe Hand Landmarker.

    Why:
        Convert raw camera frames into structured hand landmarks that can be
        processed by downstream modules such as the landmark normalizer and
        classifier.

    Input:
        RGB Numpy Matrix (H x W x 3)

    Process:
        Validate the RGB frame.
        Convert the frame into MediaPipe Image format.
        Detect hands using MediaPipe Hand Landmarker.
        Validate the detected landmarks.

    Output:
        List[List[NormalizedLandmark]]
            Detected hand landmarks.

        None
            Returned when no hand is detected in the current frame.
    """

    #Variables: self.hand_detector
    def __init__(self, detector = None) -> None:
        
        self.config = {} #Variable for config file.
        
        #This ensures the project can find its configuration file safely
        config_path = Path(__file__).parent.parent.parent / "config.json"

        #This ensures the project can find the hand landmarker task.
        model_path = Path(__file__).parent / "hand_landmarker.task"

        #Error handling for the config file.
        try:

            #This load the config.json into self.config variable.
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Unable to load configuration file: {config_path}") from e

        #Store and initiate what task will be used.
        base_options = BaseOptions(model_asset_path = str(model_path))

        #Initialize the MediaPipe Hands inference configurations.
        options = vision.HandLandmarkerOptions( 
            
            #Initialize the configurations of hand detector.
            base_options = base_options,
            num_hands = int(self.config.get("max_num_hands", 2)),
            #min_hand_detection_confidence = float(self.config.get("min_detection_confidence", 0.5)),
            #min_hand_presence_confidence = float(self.config.get("min_tracking_confidence", 0.5)),
            #min_tracking_confidence = float(self.config.get("min_tracking_confidence", 0.5))
        )

        #Initiate the Hand detector.
        self.hand_detector = detector or vision.HandLandmarker.create_from_options(options)

    #Validation of inputs
    def validate_input(self, frame: np.ndarray) -> np.ndarray:
        
        """
        Validate the RGB input frame.

        Input:
            frame: np.ndarray

        Process:
            Check if frame exists.
            Check frame data type.
            Check frame dimensions.
            Check color channels.
            Check image data type.

        Output:
            Validated RGB frame.

        Failure Conditions:
            Raise TypeError if the frame type is invalid.

            Raise ValueError if the frame is empty or has
            an invalid image shape.
        """

        if frame is None:
            raise ValueError("No frame detected!")
        
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be Numpy array!")
        
        if frame.size == 0:
            raise ValueError("Frame is empty!")
        
        if frame.ndim != 3:
            raise ValueError("Frame must have 3 dimensions!")
        
        if frame.shape[2] != 3:
            raise ValueError("Frame must have 3 Color Channels!")
        
        if frame.dtype != np.uint8:
            raise TypeError("Invalid Image dtype. Must be uint8!")
        
        return frame

    #Validation of landmarks.
    def validate_hand_landmark(self, landmarks: list[list[NormalizedLandmark]] | None) -> list[list[NormalizedLandmark]] | None:
        
        """
        Validate MediaPipe hand landmarks.

        Input:
            landmarks:
                List[List[NormalizedLandmark]]
                OR
                None

        Process:
            Check if no hand is detected.
            Validate the landmark container.
            Validate each detected hand.
            Validate each landmark coordinate.

        Output:
            Validated hand landmarks.

            None
                Returned when no hand is detected.

        Failure Conditions:
            Raise TypeError if the landmark structure is invalid.

            Raise ValueError if a detected hand does not
            contain exactly 21 landmarks.
        """

        #No hand detected.
        if landmarks is None or len(landmarks) == 0:
            return None
        if not isinstance(landmarks, list):
            raise TypeError("Invalid landmarks data type!")
        
        for hand in landmarks: #Check each landmark.
            
            if not isinstance(hand, list):
                raise TypeError("Each hand must be a list of landmarks!")
            if len(hand) != 21:
                raise ValueError("Each hand must contain exactly 21 landmarks.")

            for landmark in hand:

                if not hasattr(landmark, "x") or not hasattr(landmark, "y") or not hasattr(landmark, "z"):
                    raise TypeError("Invalid landmark object structure.")
                if not isinstance(landmark.x, (int, float)):
                    raise TypeError("landmark.x must be numeric")
                   
        return landmarks

    #Main Hand Detection Module. | Return: UnNormalized 21 Hand landmarks
    def detect_hands(self, rgb_frames: np.ndarray) -> list[list[NormalizedLandmark]] | None:

        """
        Detect hand landmarks from an RGB frame.

        Input:
            rgb_frames:
                RGB Numpy Matrix.

        Process:
            Validate the RGB frame.
            Convert the frame into MediaPipe Image format.
            Run MediaPipe Hand Landmarker.
            Validate detected landmarks.

        Output:
            List[List[NormalizedLandmark]]
                Validated hand landmarks.

            None
                Returned when no hand is detected.

        Failure Conditions:
            Raise RuntimeError when MediaPipe processing fails.
        """

        #Validate rgb_frames first
        validated_frames = self.validate_input(rgb_frames)

        try:
            formatted_frames = mp.Image(image_format=mp.ImageFormat.SRGB,data=validated_frames)
            
            result = self.hand_detector.detect(formatted_frames)

            landmarks = result.hand_landmarks

            #Validate landmarks.
            validated_landmarks = self.validate_hand_landmark(landmarks)

        except Exception as e:
            raise RuntimeError("MediaPipe processing failed!") from e
        
        return validated_landmarks

    #Close all resources mediapipe uses
    def close_mediapipe(self):

        """
        Release MediaPipe resources.

        Process:
            Close the MediaPipe Hand Landmarker instance.

        Output:
            None
        """
        
        """Release MediaPipe resources."""
        self.hand_detector.close()