#Camera Module
from camera.camera import Camera, FrameControl
#Hand detector Module
from detection.hand_detector import HandDetector
#Normalizer Module
from feature.normalizer import Normalizer

from typing import List
import numpy as np

#Get the 21 Normalized Landmarks. Return: 21 Normalized Landmarks.
def extract_normalized_landmarks(rgb_frames: np.ndarray) -> List[float]:
    
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

    #Creation of objects.
    detect = HandDetector()
    normalizer = Normalizer()

    #Process RGB frames. Return: UnNormalized 21 Hand landmarks.
    unnormalized_landmarks = detect.detect_hands(rgb_frames)
    #Close all resources mediapipe uses.
    detect.close_mediapipe()

    #Convert Normalized the landmarks. Return: Normalized landmarks.
    normalized_landmarks = normalizer.transform(unnormalized_landmarks)

    return normalized_landmarks

"""
Main Operation of the system.
"""
def main():

    #Object creations.
    cam = Camera()
    view = FrameControl()

    #Initialize the camera.
    cam.open_camera()

    while True:

        """
        Why:
            A continuous loop is used to fetch frames from 
            the video stream in real time. The loop runs 
            until an exit condition is met, 
            allowing controlled termination of the capture process.
        """
        #Grab each frame. Return: NDarray frames (BGR)
        frame = cam.get_frames() 

        #Show the window Screen
        view.show(frame)

        #Convert the BRG -> RGB frames. | Return: RGB frames.
        rgb_frames = cam.bgr_to_rgb(frame)

        #Grab the Operation key.
        key = view.operation_key()

        #Close the window.
        if key == ord('q'):
            break
        
        #Get landmarks
        if key == ord('s'):
            #Get the 21 Normalized Landmarks
            result = extract_normalized_landmarks(rgb_frames)
            print(f"\nLandmark is successfully extracted\n The landmarks:\n {result}")
            print(f"\nInfos:\nData type: {type(result)}\nNumber of Landmark:{len(result)}")

    cam.release_camera()
    view.close()

main()

