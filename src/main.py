#Pipelines.
from pipeline.landmark_pipeline import LandmarkPipeline
#Camera Module
from camera.camera import Camera, FrameControl
#Hand detector Module
from detection.hand_detector import HandDetector
#Normalizer Module
from feature.normalizer import Normalizer
#Keyboard_operations Module
from utils.keyboard_operations import KeyboardOps

from pathlib import Path

import cv2 as cv
import numpy as np

#Draw Temporary Text box.
def text_box(frame: np.ndarray, text: str) -> None:
    
    # Draw textbox
    cv.rectangle(frame, (100, 410), (500, 450), (255,255,255), 2)
    
    #Draw current text.
    cv.putText(frame, text, (110,440),cv.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)

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

    #Update the text to default empty after pressing enter.
    text = ""

    return text


"""
Main Operation of the system.
"""
def main():

    #
    CLOSE_KEY = 27 #ESC Key.
    SAVE_POSE_KEY = 9 #TAB Key.
    ENTER_KEY = 13 #ENTER Key.

    #Set Base directory of project folder.
    BASE_DIR = Path(__file__).resolve().parent.parent

    #Object creations.
    cam = Camera()
    view = FrameControl()
    detect = HandDetector()
    normalizer = Normalizer()
    keyboard_ops = KeyboardOps()

    #Pipelines.
    landmark_pipeline = LandmarkPipeline(detect, normalizer)
    
    
    #Variables:
    pose_label = ""

    #Folder path for datasets (PNG or JPEG).
    dataset_folder_path = BASE_DIR / "data" / "raw" / pose_label

    #Initialize the camera.
    cam.open_camera()

    """
    Why:
        A continuous loop is used to fetch frames from 
        the video stream in real time. The loop runs 
        until an exit condition is met, 
        allowing controlled termination of the capture process.
    """
    while True:

        #Folder creation for datasets (PNG or JPEG).
        dataset_folder_path = BASE_DIR / "data" / "raw" / pose_label
        
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
        if key == CLOSE_KEY:
            break

        #When ENTER is pressed, it create a folder for the pose dataset (PNG or JPEG).
        if key == ENTER_KEY:  # ENTER key
           folder_creation(dataset_folder_path, pose_label)
           
        #Get landmarks
        if key == SAVE_POSE_KEY:
            #Get the 21 Normalized Landmarks
            result = landmark_pipeline.extract(rgb_frames)
            print(f"\nLandmark is successfully extracted\n The landmarks:\n {result}")
            print(f"\nInfos:\nData type: {type(result)}\nNumber of Landmark:{len(result)}")
        
        #Handle text box and update the texts.
        pose_label = keyboard_ops.update_text_input(key, pose_label)

    #Release resources opened.   
    detect.close_mediapipe() 
    cam.release_camera()
    view.close()

if __name__ == "__main__":
    main()
