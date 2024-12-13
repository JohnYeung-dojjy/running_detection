from __future__ import annotations
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from model_classes.RNN.v1 import RNNModel

from pose_detector import PoseDetectionResult

class RNNPoseActionClassifier:
  OUTPUT_LABELS = ["no_detection", "jogging", "walking", "running"]
  def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, output_dim: int, window_length: int, epochs: int, lr: str):
    model_path = Path("models") / "RNN" / f"{input_dim}-{hidden_dim}-{layer_dim}-{output_dim}_WIN-{window_length}_EPOCH-{epochs}_LR-{lr}.pt"
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model: nn.Module = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(self.device).eval()
    self.model.load_state_dict(torch.load(model_path))
    self.window_length = window_length
    self.window = deque(maxlen=self.window_length)

  @torch.no_grad()
  def classify(self, pose: PoseDetectionResult)->str:
    self.window.append(pose.normalized_landmarks.flatten())
    if len(self.window) < self.window_length:
      return "no_detection"
    pose_landmarks = torch.tensor(self.window).unsqueeze(0).to(self.device)
    y_pred: torch.Tensor = self.model(pose_landmarks)
    label = self.OUTPUT_LABELS[torch.argmax(y_pred)]
    if label == "jogging":
      label = "running"
    return label

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
        # print(np.int8(detected_pose.landmarks))
        for landmark_x, landmark_y in np.int8(detected_pose.landmarks): # type: ignore
          # landmark_x = min(int(landmark.x * image_width), image_width - 1)
          # landmark_y = min(int(landmark.y * image_height), image_height - 1)
          cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
      head_coordinate = detected_pose.landmarks[0]
      ori_x = min(int(head_coordinate[0]), image_width) + 30
      ori_y = min(int(head_coordinate[1]), image_height)

      cv2.putText(image, label_pred, (ori_x, ori_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('detect pose result', image)
