from typing import Union
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cv2.typing import MatLike
import cv2 as cv
from utils.annotator import Annotator
from utils.config import MODEL_ASSET_PATH

class LandMarker:
    def __init__(self, annotator: Union[Annotator,None]=None) -> None:
        base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2)
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        if annotator is None:
            self.annotator = Annotator()
        else:
            self.annotator = annotator

    def detect(self, frame: MatLike):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = self.detector.detect(image)
    
        annotated_image = self.annotator.draw_landmarks_on_image(image.numpy_view(), detection_result)
        return cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB)
        return cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    


if __name__ == '__main__':
    print('landmarker.py')