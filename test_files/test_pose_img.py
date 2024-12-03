import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose import Pose, PoseLandmark, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
import numpy as np


if __name__ == '__main__':
    with Pose(
        static_image_mode=True,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5) as pose:
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image = cv2.imread("test3.jpg")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        for index, landmark in enumerate(results.pose_landmarks.landmark): # type: ignore
            print(index) 
            print(f"{landmark.x * image.shape[0]}")
            print(f"{landmark.y * image.shape[1]}")
        print(f"length of {len(results.pose_landmarks.landmark)}") # type: ignore
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw_landmarks(
            image,
            results.pose_landmarks, # type: ignore
            list(POSE_CONNECTIONS),
            landmark_drawing_spec=get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        while True:
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27: # press esc
                break