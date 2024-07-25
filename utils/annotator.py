from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2 as cv
import numpy as np

from utils.embedder import embed_from_palm_base, embed_to_box

class Annotator:
    def __init__(self, margin=10,
                 font_size=1,
                 font_thickness=1,
                 handedness_text_colour=(88,205,54) # vibrant green
                 ) -> None:
        self.margin = margin  # pixels
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.handedness_txt_colour = handedness_text_colour

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
            
            # em = embed_to_box(x_coordinates,y_coordinates)
            # print(f"Embedding   shape: {em.shape}\nEmbedding:")
            # print(em)
            emb = embed_from_palm_base(x_coordinates,y_coordinates)
            # print(f"Embedding 2 shape: {emb.shape}\nEmbedding:")
            # print(emb)
            
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
            cv.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                        self.font_size, self.handedness_txt_colour, self.font_thickness, cv.LINE_AA)

        return annotated_image, embed_list