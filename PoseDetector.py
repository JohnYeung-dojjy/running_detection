from __future__ import annotations # for >3.10 builtin type hints
import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose import Pose, PoseLandmark, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
import numpy as np
# import torch
from typing import NamedTuple
# import tensorflow as tf
    
class PoseDetectionResult:
  def __init__(self, landmarks):
    """The detected pose result

    Args:
        landmarks (_type_): Pose.process().pose_landmarks
    """
    # landmark.x, landmark.y are in a [0,1] scale, except when those are out-of-bound
    assert landmarks is not None, "landmarks should not be None"
    #TODO: make it np array by default
    self.landmarks = landmarks
    self.np_landmarks: np.ndarray = np.array([[landmark.x, landmark.y] for landmark in landmarks.landmark])
  
  def normalize(self):
    """Inplace normalization to the concise_landmarks"""
    max_landmarks_xy = np.max(self.np_landmarks, axis=0)
    min_landmarks_xy = np.min(self.np_landmarks, axis=0)
    self.np_landmarks = (self.np_landmarks - min_landmarks_xy) / (max_landmarks_xy - min_landmarks_xy)
    return self
        
  # def to_torch_tensor(self):
  #   return torch.tensor(self.landmarks)
  
  def to_flattened_list(self):
    return list(self.np_landmarks.flatten())

class SinglePersonPoseDetector():
  def __init__(self,
              static_image_mode=False,
              min_detection_confidence=0.5,
              min_tracking_confidence=0.5):
    """init the single-person pose detector

    Args: check https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
        static_image_mode (bool, optional): _description_. Defaults to False.
        min_detection_confidence (float, optional): _description_. Defaults to 0.5.
        min_tracking_confidence (float, optional): _description_. Defaults to 0.5.
    """
    self.pose: Pose = Pose(static_image_mode=static_image_mode,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence)
      
  def detect(self, image)->PoseDetectionResult|None:
    """Detect pose from the given image, return None if no pose is found.

    Args:
        image (cv2.Mat): the image obtained from cv2.CaptureVideo()

    Returns:
        PoseDetectionResult|None: the detected pose object
    """
    results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # print(results.pose_landmarks)
    # if display_image:
    #   self.display_image(image, results, display_landmarks)
    if results.pose_landmarks: # type: ignore
      return PoseDetectionResult(results.pose_landmarks) # type: ignore
    else:
      return None
  # def display_image(self, image, results, display_landmarks:bool):
  #   image_height, image_width, _ = image.shape
  #   if display_landmarks and results.pose_landmarks:
  #     for landmark in results.pose_landmarks.landmark: # type: ignore
  #       # landmark_x = min(int(landmark.x * image_width), image_width - 1)
  #       # landmark_y = min(int(landmark.y * image_height), image_height - 1)
  #       landmark_x = int(landmark.x * image_width)
  #       landmark_y = int(landmark.y * image_height)
  #       cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
  #     draw_landmarks(
  #       image,
  #       results.pose_landmarks, # type: ignore
  #       list(POSE_CONNECTIONS),
  #       landmark_drawing_spec=get_default_pose_landmarks_style()
  #     )
  #   cv2.imshow('image', image)
    # if results.pose_landmarks: # type: ignore
    #     for landmark in results.pose_landmarks.landmark: # type: ignore
    #         landmark_x = min(int(landmark.x * image_width), image_width - 1)
    #         landmark_y = min(int(landmark.y * image_height), image_height - 1)
    #         cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
    # draw_landmarks(
    # image,
    # results.pose_landmarks, # type: ignore
    # list(POSE_CONNECTIONS),
    # landmark_drawing_spec=get_default_pose_landmarks_style())
    # return image
  
def test():
  from VideoStreamManager import VideoStreamManager
  # video_stream = VideoStreamManager(camera_id=0)
  video_stream = VideoStreamManager(video_file="Dataset/KTH/walking/person01_walking_d1_uncomp.avi")
  pose_detector = SinglePersonPoseDetector()
  for frame in video_stream.read_frames():
    result: PoseDetectionResult | None = pose_detector.detect(frame)

if __name__ == '__main__':
  test()