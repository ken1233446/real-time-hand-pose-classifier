#Camera Module
from camera.camera import Camera, FrameControl
#Hand detector Module
from detection.hand_detector import HandDetector
#Normalizer Module
from feature.normalizer import Normalizer
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

#Get the 21 Normalized Landmarks. Return: 21 Normalized Landmarks.
def extract_normalized_landmarks(rgb_frames: np.ndarray, detect: HandDetector, normalizer: Normalizer) -> List[float]:
    
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
    unnormalized_landmarks = detect.detect_hands(rgb_frames)

    #Convert Normalized the landmarks. Return: Normalized landmarks.
    normalized_landmarks = normalizer.transform(unnormalized_landmarks)

    return normalized_landmarks

#Draw Temporary Text box.
def text_box(frame: np.ndarray, text: str) -> None:
    
    # Draw textbox
    cv.rectangle(frame, (100, 410), (500, 450), (255,255,255), 2)
    
    #Draw current text.
    cv.putText(frame, text, (110,440),cv.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)

#Use for text box.
def key_operations(key: int, text: str) -> str:

    #Key Operations.
    if key == 8: #When Backspace is entered it subtract the pose_label string.
        text = text[:-1]

    #All keys except the conditions will update the pose label variable.
    elif 32 <= key <= 126 and key != ord('s') and key != ord('q'):
        text += chr(key)

    return text

#Folder creation. | Return: text
def folder_creation(folder_path: Path, text: str) -> str:
    
    #Create folder for dataset.
    try: 
        folder_path.mkdir(parents=True)
        print(f"New Pose folder created at {folder_path} named {text}")

    #Check if the folder already exist.
    except FileExistsError:
        print(f"Folder already exists: {folder_path}")
    
    #Check for any Error in creating folder.
    except OSError as e:
        print(f"Error creating folder: {e}")

    #Set the text to default empty after pressing enter.
    text = ""

    return text


"""
Main Operation of the system.
"""
def main():

    #Set Base directory of project folder.
    BASE_DIR = Path(__file__).resolve().parent.parent

    #Object creations.
    cam = Camera()
    view = FrameControl()
    detect = HandDetector()
    normalizer = Normalizer()
    
    #Variables:
    pose_label = ""

    #Folder creation for datasets (PNG or JPEG).
    dataset_folder_path = BASE_DIR / "data" / "raw" / pose_label

    #Initialize the camera.
    cam.open_camera()

    while True:
        #Folder creation for datasets (PNG or JPEG).
        dataset_folder_path = BASE_DIR / "data" / "raw" / pose_label
        """
        Why:
            A continuous loop is used to fetch frames from 
            the video stream in real time. The loop runs 
            until an exit condition is met, 
            allowing controlled termination of the capture process.
        """
        #Grab each frame. Return: NDarray frames (BGR)
        frame = cam.get_frames()

        #Draw Text box.
        text_box(frame, pose_label)

        #Show the window Screen
        view.show(frame)

        #Convert the BRG -> RGB frames. | Return: RGB frames.
        rgb_frames = cam.bgr_to_rgb(frame)

        #Grab the Operation key.
        key = view.operation_key()

        #Close the window.
        if key == ord('q'):
            break

        #When ENTER is pressed, it create a folder for the pose dataset (PNG or JPEG).
        if key == 13:  # ENTER key
           folder_creation(dataset_folder_path, pose_label)
           
        #Get landmarks
        if key == ord('s'):
            #Get the 21 Normalized Landmarks
            result = extract_normalized_landmarks(rgb_frames, detect, normalizer)
            print(f"\nLandmark is successfully extracted\n The landmarks:\n {result}")
            print(f"\nInfos:\nData type: {type(result)}\nNumber of Landmark:{len(result)}")
        
        #Key Operations for text box.
        pose_label = key_operations(key, pose_label)

        
    detect.close_mediapipe()
    cam.release_camera()
    view.close()

if __name__ == "__main__":
    main()
