from __future__ import annotations # for >3.10 builtin type hints
from pathlib import Path
from typing import Literal
from abc import ABC, abstractmethod
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch

class PoseDetectionResult(ABC):
  landmarks: np.ndarray

  @property
  @abstractmethod
  def normalized_landmarks(self) -> np.ndarray:
    return NotImplemented

class UltralyticsPoseDetectionResult(PoseDetectionResult):
  def __init__(self, result: Results):
    self.result = result
    if result.keypoints is None:
      raise ValueError("result.keypoints is None, this should not happen if we are using a pose estimation model")
    # shape is of (detected person, keypoints, 2) always pick the first person
    # first 5 landmarks are head landmarks, we do not use them in running detection
    landmarks = result.keypoints.xy[0][6:]
    if isinstance(landmarks, torch.Tensor):
      self.landmarks = landmarks.cpu().numpy()

  @property
  def normalized_landmarks(self) -> np.ndarray:
    # already checked that result.keypoints is not None in class init, always pick the first person
    # first 5 landmarks are head landmarks, we do not use them in running detection
    normalized_landmarks = self.result.keypoints.xyn[0][6:]
    if isinstance(normalized_landmarks, torch.Tensor): # shape is of (detected person, keypoints, 2) already pick the first person
      normalized_landmarks = normalized_landmarks.cpu().numpy()
    return normalized_landmarks



class UltralyticsPoseDetector:
  POSE_OUTPUT_LABELS = [
    "nose_x", "nose_y",
    "left_eye_x", "left_eye_y",
    "right_eye_x", "right_eye_y",
    "left_ear_x", "left_ear_y",
    "right_ear_x", "right_ear_y",
    "left_shoulder_x", "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "left_elbow_x", "left_elbow_y",
    "right_elbow_x", "right_elbow_y",
    "left_wrist_x", "left_wrist_y",
    "right_wrist_x", "right_wrist_y",
    "left_hip_x", "left_hip_y",
    "right_hip_x", "right_hip_y",
    "left_knee_x", "left_knee_y",
    "right_knee_x", "right_knee_y",
    "left_ankle_x", "left_ankle_y",
    "right_ankle_x", "right_ankle_y",
  ]
  def __init__(self, model_size: Literal['n', 's', 'm', 'l', 'x']):
    model_path = Path("models") / "pose" / f"yolo11{model_size}-pose.pt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = YOLO(model_path).to(device)

  def detect(self, frame):
    result: Results = self.model.predict(frame, verbose=False)[0]
    if result.keypoints is None:
      raise ValueError("result.keypoints is None, this should not happen if we are using a pose estimation model")
    if result.keypoints.xy.shape[1] == 0: # shape is of (detected person, keypoints, 2)
      return None
    pose_detect_res = UltralyticsPoseDetectionResult(result)
    return pose_detect_res



def test():
  from vidio_stream_manager import VideoStreamManager
  # video_stream = VideoStreamManager(camera_id=0)
  video_stream = VideoStreamManager(video_file="Dataset/KTH/running/person01_running_d1_uncomp.avi")
  pose_detector = UltralyticsPoseDetector("x")
  for frame in video_stream.read_frames():
    result: PoseDetectionResult | None = pose_detector.detect(frame)

if __name__ == '__main__':
  test()