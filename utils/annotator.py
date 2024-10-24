import os
from typing import Union
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from utils.config import GESTURE_MODEL_PATH
from utils.data import read_class_labels

from model.model import GestureModel
from utils.embedder import embed_from_palm_base, embed_to_box

class Annotator:
    def __init__(self, margin=10,
                 font_size=1,
                 font_thickness=1,
                 handedness_text_colour=(88,205,54), # vibrant green
                 load_model:Union[str,None]=None,
                 ) -> None:
        self.margin = margin  # pixels
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.handedness_txt_colour = handedness_text_colour
        self.gesture_model = None
        self.metal = torch.backends.mps.is_available()
        if load_model is not None:
            self.gesture_model = GestureModel()
            self.gesture_model.load_state_dict(torch.load(os.path.join(GESTURE_MODEL_PATH, load_model)))
            self.gesture_model.eval()
            self.class_labels = read_class_labels()
            if self.metal:
                self.gesture_model.to('mps:0')

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        
        embed_list = []

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # type: ignore
            hand_landmarks_proto.landmark.extend([
            # 1.0 - for webcam flip
            landmark_pb2.NormalizedLandmark(x=1.0-landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks # type: ignore
            ])
            solutions.drawing_utils.draw_landmarks( # type: ignore
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS, # type: ignore
            solutions.drawing_styles.get_default_hand_landmarks_style(), # type: ignore
            solutions.drawing_styles.get_default_hand_connections_style()) # type: ignore

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [1.0 - landmark.x for landmark in hand_landmarks] # 1.0 - for the webcam flip
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            
            
            emb = embed_from_palm_base(x_coordinates,y_coordinates)
            
            # predict gesture
            
            prediction = ''
            
            if self.gesture_model is not None:
                emb_tensor = torch.tensor(emb, dtype=torch.float32)
                if self.metal:
                    emb_tensor = emb_tensor.to('mps:0')
                
                output_logits = self.gesture_model(emb_tensor.view(-1))
                
                
                output_probs = F.softmax(output_logits, dim=0)
    
                # Get the predicted index and its confidence score
                score, predicted_idx = torch.max(output_probs, dim=0)
                
                # print(score)
                
                # Check if the confidence score is above the threshold
                if score > 0.85:
                    # Convert the index to a class label
                    predicted_label = self.class_labels[predicted_idx.item()]
                    prediction = str(predicted_label)
                
                # print(predic)
            
            embed_list.append(emb)

            # calculate bounding box
            x_min = int(min(x_coordinates) * width)
            x_max = int(max(x_coordinates) * width)
            y_min = int(min(y_coordinates) * height)
            y_max = int(max(y_coordinates) * height)

            box_width = x_max - x_min
            box_height = y_max - y_min
            
            BOX_SCALING = 0.2 
            x_min = max(0, x_min - int(BOX_SCALING * box_width))
            x_max = min(width, x_max + int(BOX_SCALING * box_width))
            y_min = max(0, y_min - int(BOX_SCALING * box_height))
            y_max = min(height, y_max + int(BOX_SCALING * box_height))

            # Draw the bounding box
            cv.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)


            # Calculate text position
            text_x = x_min
            text_y = y_min - self.margin

            # Draw handedness (left or right hand) on the image.
            cv.putText(annotated_image, f"{handedness[0].category_name}" + ('' if prediction == '' else f' - {prediction}'),
                        (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                        self.font_size, self.handedness_txt_colour, self.font_thickness, cv.LINE_AA)


        return annotated_image, embed_list