from src.detection.hand_detector import HandDetector
import pytest
import numpy as np
from types import SimpleNamespace

class TestHandDetectorValidation:
    
    """
    
    Unit tests for Hand Detector (Validation) module.

    Variables:
    self.test_frame ← Main object
    
    self.valid_frames, self.no_frame, self.not_ndarray
    self.empty_frame, self.not_3d_frame, self.frame_4ch
    self.invalid_dtype_frame
    
    """

    def setup_method(self):

        """
        This runs automatically before every individual test.
        It ensures a fresh instance of Normalizer is always available.
        """

        #Create a new object of Hand Detector class.
        self.test_frame = HandDetector()

        #-----------------------------
        # Helper: Mock Numpy matrix
        #-----------------------------

        #Valid numpy maxtrix
        self.valid_frames = np.array(
            [
                [[0, 0, 255], [0, 255, 0]],
                [[255, 0, 0], [255, 255, 255]]
            ], dtype=np.uint8)
        
        #No frames detected variable
        self.no_frame = None
        
        #Not Numpy array variable
        self.not_ndarray = [[1, 2], [3, 4]] 

        #Empty numpy array variable
        self.empty_frame = np.array([])

        #Not 3D numpy array variable
        self.not_3d_frame = np.array(
                [
                    [0, 255, 128],
                    [64, 128, 255]
                ], dtype=np.uint8)

        #4 Channel Numpy Array
        self.frame_4ch = np.array(
                [
                    [[255, 0, 0, 255], [0, 255, 0, 255]],
                    [[0, 0, 255, 255], [255, 255, 255, 255]]
                ],dtype=np.uint8)
        
        #Invalid dtype Numpy Array
        self.invalid_dtype_frame = np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]]
                ], dtype=np.int32)
        
    #Test Valid frame case.
    def test_valid_frame(self):

        #Store the return of validation method.
        result = self.test_frame.validate_input(self.valid_frames)

        #Check if it return the same numpy matrix after validation.
        assert result is self.valid_frames, \
            "Error: Something wrong with validation of frames."

    #Test no frame detected.
    def test_no_frame(self):
        
        #Check if it raises error when no frame detected.
        with pytest.raises(ValueError, match= "No frame detected!"):
            self.test_frame.validate_input(self.no_frame) #type: ignore
    
    #Test frame not Numpy array.
    def test_not_ndarray(self):

        #Check if it raise error when frame is not ndarray.
        with pytest.raises(TypeError, match= "Frame must be Numpy array!"):
            self.test_frame.validate_input(self.not_ndarray) #type: ignore

    #Test empty frame.
    def test_empty_frame(self):

          #Check if it raise error when frame is empty.
        with pytest.raises(ValueError, match= "Frame is empty!"):
            self.test_frame.validate_input(self.empty_frame)

    #Test not 3 dimension.
    def test_not_3dimension(self):

        #Check if it raises error when frame is not 3 dimension.
        with pytest.raises(ValueError, match= "Frame must have 3 dimensions!"):
            self.test_frame.validate_input(self.not_3d_frame)

    #Test frame not 3 color channel.
    def test_not_3colorchannel(self):

        #Check if it raises error when frame is not 3 color channel.
        with pytest.raises(ValueError, match= "Frame must have 3 Color Channels!"):
            self.test_frame.validate_input(self.frame_4ch)

    #Test frame not correct dtype.
    def test_not_proper_dtype(self):
        
        #Check if it raises error when incorrect dtype.
        with pytest.raises(TypeError, match= "Invalid Image dtype. Must be uint8!"):
            self.test_frame.validate_input(self.invalid_dtype_frame)

class TestHandDetectorLandmarkValidation:

    """

    Unit tests for Hand Detector (Landmark Validation) module.

    Variables:
    self.test_frame ← Main object

    self.valid_landmarks, self.no_landmarks, self.not_list_input
    self.empty_landmarks, self.not_hand_list
    self.invalid_landmark_count
    self.invalid_landmark_object
    self.invalid_landmark_dtype

    """

    def setup_method(self):

        """
        This runs automatically before every individual test.
        It ensures a fresh instance of HandDetector is always available.
        """

        # Create new object of Hand Detector class.
        self.test_frame = HandDetector()

        # -----------------------------
        # Helper: Mock Landmark Object
        # -----------------------------

        def make_landmark(x=0.0, y=0.0, z=0.0):
            return SimpleNamespace(x=x, y=y, z=z)

        # Valid landmarks (1 hand, 21 landmarks)
        self.valid_landmarks = [ [make_landmark(0.1, 0.2, 0.3) for _ in range(21)] ]

        # No landmarks detected variable
        self.no_landmarks = None

        # Not list input
        self.not_list_input = "invalid_input"

        # Empty landmarks list
        self.empty_landmarks = []

        # Not hand list
        self.not_hand_list = "not_a_hand_list"

        # Invalid landmark count (not 21)
        self.invalid_landmark_count = [[make_landmark() for _ in range(10)]]

        # Invalid landmark object structure (missing z)
        self.invalid_landmark_object = [[SimpleNamespace(x=1.0, y=2.0) for _ in range(21)]]

        # Invalid landmark dtype (x is string)
        self.invalid_landmark_dtype = [[SimpleNamespace(x="bad", y=1.0, z=1.0) for _ in range(21)]]

    # Test valid landmarks case.
    def test_valid_landmarks(self):

        # Store return of validation method.
        result = self.test_frame.validate_hand_landmark(self.valid_landmarks)

        # Check if it returns same landmarks after validation.
        assert result is self.valid_landmarks, \
            "Error: Something wrong with landmark validation."

    # Test no landmarks detected.
    def test_no_landmarks(self):

        # Check if it raises error when landmarks is None.
        with pytest.raises(ValueError, match="No landmarks detected!"):
            self.test_frame.validate_hand_landmark(self.no_landmarks)

    # Test not list input.
    def test_not_list_input(self):

        # Check if it raises error when input is not list.
        with pytest.raises(TypeError, match="Invalid landmarks data type!"):
            self.test_frame.validate_hand_landmark(self.not_list_input)

    # Test empty landmarks.
    def test_empty_landmarks(self):

        # Check if it raises error when landmarks is empty.
        with pytest.raises(ValueError, match="No hands detected!"):
            self.test_frame.validate_hand_landmark(self.empty_landmarks)

    # Test hand not list.
    def test_not_hand_list(self):

        # Check if it raises error when hand is not list.
        with pytest.raises(TypeError, match="Invalid landmarks data type!"):
            self.test_frame.validate_hand_landmark(self.not_hand_list)

    # Test invalid landmark count.
    def test_invalid_landmark_count(self):

        # Check if it raises error when landmark count is not 21.
        with pytest.raises(ValueError, match="Each hand must contain exactly 21 landmarks."):
            self.test_frame.validate_hand_landmark(self.invalid_landmark_count)

    # Test invalid landmark object.
    def test_invalid_landmark_object(self):

        # Check if landmark object structure is invalid.
        with pytest.raises(TypeError, match="Invalid landmark object structure."):
            self.test_frame.validate_hand_landmark(self.invalid_landmark_object)

    # Test invalid landmark dtype.
    def test_invalid_landmark_dtype(self):

        # Check if landmark.x is not numeric.
        with pytest.raises(TypeError, match="landmark.x must be numeric"):
            self.test_frame.validate_hand_landmark(self.invalid_landmark_dtype)

