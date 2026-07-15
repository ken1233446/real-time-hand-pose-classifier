from typing import List
import math as m

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

class Normalizer:
    
    """
    Normalizer Contract

    What:
        Convert raw MediaPipe hand landmarks into normalized landmark features
        suitable for machine learning models.

    Why:
        Remove translation and scale variations so that the classifier learns
        the hand pose rather than the hand position or distance from the camera.

    Input:
        List[List[NormalizedLandmark]]
            Raw MediaPipe hand landmarks.

        None
            Returned when no hand is detected.

    Process:
        Convert MediaPipe landmark objects into float coordinates.
        Validate the converted landmarks.
        Translate landmarks relative to the wrist.
        Normalize hand size using the middle finger base.
        Produce a normalized feature vector.

    Output:
        List[float]
            Normalized landmark feature vector containing 63 values.

        None
            Returned when no hand is detected.
    """
    
    WRIST_INDEX = 0 #Index of the wrist.
    MIDDLE_FINGER_BASE_START = 27
    MIDDLE_FINGER_BASE_END = 30

    #Convert landmarkto normal list coord. | Return: converted landmarks | None
    def landmark_processor(self, raw_landmarks: List[List[NormalizedLandmark]] | None) -> List[List[float]] | None:

        """
        Convert MediaPipe landmarks into numerical coordinates.

        Input:
            raw_landmarks:
                List[List[NormalizedLandmark]]

                OR

                None

        Process:
            Check if landmarks exist.
            Validate each landmark object.
            Convert each landmark into
            [x, y, z] float coordinates.

        Output:
            List[List[float]]
                Converted landmark coordinates.

            None
                Returned when no hand is detected.

        Failure Conditions:
            Raise ValueError if a landmark object
            does not contain x, y and z attributes.
        """

        #No hand detected.
        if raw_landmarks is None:
            return None

        #Validate if landmark empty
        if not raw_landmarks or not raw_landmarks[0]:
            return None

        #Validate each landmark coordinates.
        for lm in raw_landmarks[0]:
            if not all(hasattr(lm, attr) for attr in ("x", "y", "z")):
                raise ValueError("Each landmark must have x, y, and z attributes")

        converted_landmarks = [[lm.x, lm.y, lm.z]for lm in raw_landmarks[0]]

        return converted_landmarks
    
    #Validation landmark if True, have 21 list, 3 coordinate each List and if it's a list. | Return: validated_landmark
    def validate(self, landmark: List[List[float]]) -> List[List[float]]:
        
        """
        Validate converted landmark coordinates.

        Input:
            landmark:
                List[List[float]]

        Process:
            Validate landmark container.
            Validate total number of landmarks.
            Validate coordinate dimensions.
            Validate coordinate data types.

        Output:
            Validated landmark coordinates.

        Failure Conditions:
            Raise TypeError if landmark data types
            are invalid.

            Raise ValueError if landmark structure
            is invalid.
        """

        if not isinstance(landmark, list):
            raise TypeError(f"Expected a list, but got {type(landmark).__name__}.")
        
        if not landmark:
            raise ValueError("Landmarks can't be empty.")

        if not len(landmark) == 21:
            raise ValueError(f"Expected 21 landmarks, but got {len(landmark)}.")
        
        if not all(isinstance(point, list) and len(point) == 3 for point in landmark):
            raise ValueError('Expected each landmark to be a list of 3 coordinates.')

        if not all(isinstance(coord, float) for point in landmark for coord in point):
            raise TypeError("Coordinates must be float")
  
        return landmark
    
    #Translate landmarks relative to wrist. | Return: relative_landmarks
    def trans_relative_wrist(self, processed_landmark: List[List[float]]) -> List[float]:
        
        """
        Translate landmarks relative to the wrist.

        Why:
            Remove hand position by treating the wrist
            as the origin of the coordinate system.

        Input:
            processed_landmark:
                List[List[float]]

        Process:
            Get wrist coordinates.
            Subtract the wrist coordinates from
            every landmark.
            Flatten the translated coordinates
            into a single list.

        Output:
            List[float]
                Relative landmark coordinates.
        """

        x_wrist, y_wrist, z_wrist = processed_landmark[self.WRIST_INDEX] #Get the coordinates of WRIST.
        relative_coordinates = [] #Container of Coordinate relative to wrist.

        for point in processed_landmark: #landmark: [[2.0,5.0,6.9],[8.4,7.5,6.6], ......] -> point: [2.0, 5.0, 6.9]

            x_ref, y_ref, z_ref = point #Get the 3 coordinate in each landmark (0-21).
            #Get the relative of each coordinate relative to wrist.
            #Store the relative coordinates.
            relative_coordinates.extend((x_ref - x_wrist, y_ref - y_wrist, z_ref - z_wrist)) 

        return relative_coordinates
    
    #Normalize hand size for scale invariance. | Return: scaled_coordinates
    def trans_scale_invariance(self, relative_landmark: List[float]) -> List[float]:

        """
        Normalize landmark scale.

        Why:
            Remove differences caused by the hand
            being closer or farther from the camera.

        Formula:
            D = √(x² + y² + z²)

        Input:
            relative_landmark:
                List[float]

        Process:
            Compute the scaling factor using the
            wrist and middle finger base.
            Divide every coordinate by the
            scaling factor.

        Output:
            List[float]
                Scale-normalized landmark coordinates.

        Failure Conditions:
            Raise ValueError if the scaling factor
            is equal to zero.
        """

        scaled_coordinates = []
        #Get the Landmark 9 (Base of middle finger) for scale factor
        x_mfinger, y_mfinger, z_mfinger = relative_landmark[self.MIDDLE_FINGER_BASE_START:self.MIDDLE_FINGER_BASE_END]

        #Get scaling factor from WRIST to BASE MIDDLE FINGER
        scaling_factor = m.sqrt((x_mfinger **2) + (y_mfinger **2) + (z_mfinger **2))

        #Edge Case if scaling_factor is ZERO (0)
        if scaling_factor == 0:
            raise ValueError('Scaling factor MUST NOT be zero.')
        
        #Iterate to landmark List[float] & Fix scale invariance of all coordinates.
        for coord in relative_landmark:
            new_coord = coord / scaling_factor
            scaled_coordinates.append(new_coord) # Store scaled factors.

        return scaled_coordinates

     #Translate the Raw landmarks to relative and scaled landmarks. | Return: final_landmarks  

    #Main Function. | Return: final_landmarks
    def transform(self, raw_landmark: List[List[NormalizedLandmark]] | None) -> List[float] | None:

        """
        Transform raw landmarks into ML-ready features.

        Input:
            raw_landmark:
                List[List[NormalizedLandmark]]

                OR

                None

        Process:
            Convert landmark objects into float coordinates.
            Validate the converted landmarks.
            Translate landmarks relative to the wrist.
            Normalize landmark scale.

        Output:
            List[float]
                Normalized landmark feature vector.

            None
                Returned when no hand is detected.
        """

        converted = self.landmark_processor(raw_landmark)


        #No hand detected.
        if converted is None:
            return None


        validated_landmarks = self.validate(converted)

        relative_landmarks = self.trans_relative_wrist(validated_landmarks)

        final_landmarks = self.trans_scale_invariance(relative_landmarks)

        return final_landmarks
    
