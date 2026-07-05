import numpy as np
import cv2 as cv

#Create and Update the text box.
class TextBox:

    #Initialized the OpenCameraVision object.
    def __init__(self) -> None:
        self.cv = cv
    
    #Draw Temporary Text box.
    def create_text_box(self, frame: np.ndarray, text: str) -> np.ndarray:
    
        # Draw textbox
        textbox = self.cv.rectangle(frame, (100, 410), (500, 450), (255,255,255), 2)
        
        #Draw current text.
        result = self.cv.putText(textbox, text, (110,440),cv.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)

        #Return the frame with updated text box.
        return result