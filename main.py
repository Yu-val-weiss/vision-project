from queue import PriorityQueue
import time
import cv2 as cv
import numpy as np
from utils.annotator import Annotator
from utils.config import MODEL_ASSET_PATH, COLLECTED_DATA_CSV
from utils.data import check_class_proportions
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv


class_dict = {
    ')': 10,
    '!': 11,
    '@': 12,
    '£': 13,
    '$': 14,
    '%': 15,
    '^': 16,
    '&': 17,
    '*': 18,
    '(': 19
}


annotator = Annotator()

result_queue = PriorityQueue()

def process_res(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        output_ndarray = output_image.numpy_view()
        copied_ndarray = np.copy(output_ndarray)
        img = cv.flip(copied_ndarray, flipCode=1)
        img, embed_list = annotator.draw_landmarks_on_image(img,result)
        result_queue.put((timestamp_ms, (embed_list, img)))
    except Exception as e:
        print(f"Error processing results: {e}")
    
def record_datapoint(class_index, embedding: np.ndarray):
    with open(COLLECTED_DATA_CSV, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([class_index] + embedding.reshape(-1).tolist())

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
    
    record_mode = False
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # reads next frame
            ret, frame = cap.read()
            frame_ts = int(time.time() * 1000)
            
            # frame read correctly -> ret is True
            if not ret:
                print("Can't receive frame. Exiting...")
                break
            
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            landmarker.detect_async(mp_image, frame_ts)
            
            _, (embed_list, res_img) = result_queue.get(block=True)
            
            
            
            # print(embed_list)
            if record_mode:
                res_img = cv.putText(res_img,"Record mode", (30,30), cv.FONT_HERSHEY_DUPLEX,1,(255,0,0))
            
            cv.imshow('frame', res_img)
            
            # wait for...
            key = cv.waitKey(1)
            if key == ord('q'):
                if record_mode:
                    print('Quit in record mode, printing class proportions...')
                    check_class_proportions()
                break
            
            elif key == ord('r'):
                record_mode = not record_mode
                print(f"\n\n{'!!  Entered' if record_mode else 'Exited'} record mode  !!\n")
                print('Class proportions...')
                check_class_proportions()
                    
            elif key != -1:
                if record_mode:
                    if len(embed_list) == 0:
                        print('No hands found to record')
                        continue
                    if len(embed_list) > 1:
                        print('More than one hand found, skipped recording')
                        continue
                    c = chr(key)
                    if c.isdigit():
                        for emb in embed_list:
                            record_datapoint(int(c), emb)
                        print(f'Recorded {int(c)}')
                    elif c in class_dict:
                        for emb in embed_list:
                            record_datapoint(class_dict[c], emb)
                        print(f'Recorded {class_dict[c]}')
                    else:
                        print(f"Invalid key {c}")
            
            
            
    cap.release()
    cv.destroyAllWindows()