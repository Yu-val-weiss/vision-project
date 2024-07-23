from queue import PriorityQueue
import time
import cv2 as cv
import numpy as np
from utils.annotator import Annotator
from utils.config import MODEL_ASSET_PATH
from utils.landmarker import LandMarker
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


annotator = Annotator()

result_queue = PriorityQueue()

def process_res(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        output_ndarray = output_image.numpy_view()
        copied_ndarray = np.copy(output_ndarray)
        img = cv.flip(copied_ndarray, flipCode=1)
        img = annotator.draw_landmarks_on_image(img,result)
        result_queue.put((timestamp_ms, (result, img)))
    except Exception as e:
        print(f"Error processing results: {e}")
    

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
        result_callback=process_res)
    
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
            
            # frame = cv.flip(frame, flipCode=1)
            
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            landmarker.detect_async(mp_image, frame_ts)
            
            _,(result, res_img) = result_queue.get(block=True)
            
            print(result)
            
            # res_img = cv.flip(res_img, flipCode=1)
            
            cv.imshow('frame', res_img)
            
            # wait for...
            if cv.waitKey(1) == ord('q'):
                break
            
            
    cap.release()
    cv.destroyAllWindows()