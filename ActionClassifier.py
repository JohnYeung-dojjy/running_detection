from __future__ import annotations
from .PoseDetector import PoseLandmark, PoseDetectionResult
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
import cv2
from .models import load_pose_action_classifier, load_label_encoder
try:
  from sklearnex import patch_sklearn # speed up sklearn if cpu is intel
  patch_sklearn()
except ImportError as sklearnex_not_installed:
  print("sklearnex not installed, use default sklearn instead")
  print("if you want to use sklearn, please refer to https://pypi.org/project/scikit-learn-intelex/")

from sklearn import svm, preprocessing
from mediapipe.python.solutions.drawing_utils import draw_landmarks, draw_detection

# The unneeded features to be filtered
MEDIAPIPE_MASK: list[bool] = [False, False, # nose
                              False, False, # left_eye_inner
                              False, False, # left_eye
                              False, False, # left_eye_outer
                              False, False, # right_eye_inner
                              False, False, # right_eye
                              False, False, # right_eye_outer
                              False, False, # left_ear
                              False, False, # right_ear
                              False, False, # mouth_left
                              False, False, # mouth_right
                              True , True , # left_shoulder
                              True , True , # right_shoulder
                              True , True , # left_elbow
                              True , True , # right_elbow
                              True , True , # left_wrist
                              True , True , # right_wrist
                              False, False, # left_pinky
                              False, False, # right_pinky
                              False, False, # left_index
                              False, False, # right_index
                              False, False, # left_thumb
                              False, False, # right_thumb
                              True , True , # left_hip
                              True , True , # right_hip
                              True , True , # left_knee
                              True , True , # right_knee
                              True , True , # left_ankle
                              True , True , # right_ankle
                              False, False, # left_heel
                              False, False, # right_heel
                              False, False, # left_foot_index
                              False, False, # right_foot_index
                              ]

class PoseActionClassifier:
  def __init__(self, jogging: str=""):
    """Set up the pose action classifier
    if set to 'running', load the classifier which is trained with 'jogging' treated as 'running' in the dataset.
    if set to 'walking', load the classifier which is trained with 'jogging' treated as 'walking' in the dataset.
    if set to ''       , load the classifier which is trained with 'jogging' treated as ''        in the dataset.

    Args:
        pose_action_classifier_path (Path | str): the path to the pre-trained pose action classifier model
    """
    self.model: svm.SVC = load_pose_action_classifier(jogging=jogging)
    self.label_encoder: preprocessing.LabelEncoder = load_label_encoder()

  def classify(self, pose: PoseDetectionResult)->str:
    data = pose.normalize().np_landmarks.flatten()[MEDIAPIPE_MASK].reshape(1, -1) # reshape as it contains only 1 sample
    y_pred = self.model.predict(data)
    label = self.label_encoder.inverse_transform(y_pred)
    # print(y_pred, label)
    return label[0]
  
  def display_image(self, image, label_pred: str, detected_pose: PoseDetectionResult|None, display_image:bool=False, display_landmarks:bool=False):
    """Display the detected pose skeleton and the predicted pose action"""
    # make image larger if it was small
    image_height, image_width, _ = image.shape
    target_width = min(680, image_width*2)
    target_height = target_width * image.shape[0] // image.shape[1]
    image = cv2.resize(image, (target_width, target_height))
    
    
    image_height, image_width, _ = image.shape
    if display_image and detected_pose is not None:
      if display_landmarks:
        for landmark in detected_pose.landmarks.landmark: # type: ignore
          # landmark_x = min(int(landmark.x * image_width), image_width - 1)
          # landmark_y = min(int(landmark.y * image_height), image_height - 1)
          landmark_x = int(landmark.x * image_width)
          landmark_y = int(landmark.y * image_height)
          cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
        draw_landmarks(
          image,
          detected_pose.landmarks, # type: ignore
          list(POSE_CONNECTIONS),
          landmark_drawing_spec=get_default_pose_landmarks_style()
        )
      head_coordinate = detected_pose.landmarks.landmark[0]
      ori_x = min(int(head_coordinate.x * image_width), image_width) + 30
      ori_y = min(int(head_coordinate.y * image_height), image_height)
      # print(ori_x, ori_y)
      cv2.putText(image, label_pred, (ori_x, ori_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('detect pose result', image)