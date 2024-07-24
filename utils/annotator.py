from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2 as cv
import numpy as np

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
        
        print("# results:", detection_result)
        
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            # 1.0 - for webcam flip
            landmark_pb2.NormalizedLandmark(x=1.0-landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            # 1.0 - for the webcam flip
            text_x = int((1.0 - max(x_coordinates)) * width)
            text_y = int(min(y_coordinates) * height) - self.margin

            # Draw handedness (left or right hand) on the image.
            cv.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                        self.font_size, self.handedness_txt_colour, self.font_thickness, cv.LINE_AA)

        return annotated_image