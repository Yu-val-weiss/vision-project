import time
import cv2 as cv
import numpy as np
from utils.config import MODEL_ASSET_PATH
from utils.landmarker import LandMarker
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: ', result)

if __name__ == '__main__':
    cap = cv.VideoCapture(0) # 0 sets to default camera (index)
    
    # lm = LandMarker()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
    pTime = 0
    
    base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=print_result)
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
    
        while True:
            # reads next frame
            ret, frame = cap.read()
            frame_ts = int(time.time() * 1000)
            
            # frame read correctly -> ret is True
            if not ret:
                print("Can't receive frame. Exiting...")
                break
            
            # cvtColor converts an image from one colour space to another
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            frame = cv.flip(frame, flipCode=1)
            
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            landmarker.detect_async(mp_image,frame_ts)
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv.putText(frame, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # display the frame --- cv.imshow(winname, mat)
            cv.imshow('frame', frame)
            
            # wait for...
            if cv.waitKey(1) == ord('q'):
                break
            
    cap.release()
    cv.destroyAllWindows()