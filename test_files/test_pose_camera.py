import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose import Pose, PoseLandmark, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
import numpy as np

# For webcam input:
print("obtaining camera object...")
cap = cv2.VideoCapture(0)
print("done", cap, cap.isOpened())
# capture video with 30 fps
cap.set(cv2.CAP_PROP_FPS, 30)
  
with Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_landmarks(
        image,
        results.pose_landmarks, # type: ignore
        list(POSE_CONNECTIONS),
        landmark_drawing_spec=get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()