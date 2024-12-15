from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from model_classes.RNN.v1 import RNNModel
from model_classes.LSTM.v1 import LSTMModel

from pose_detector import PoseDetectionResult

# used in model training and inference
OUTPUT_LABELS = ["walking", "running"]

class PoseActionClassifier(ABC):
  @abstractmethod
  def classify(self, pose: PoseDetectionResult)->str:
    return NotImplemented

  def display_image(self, image, label_pred: str, detected_pose: PoseDetectionResult|None, display_image:bool=False, display_landmarks:bool=False):
    """Display the detected pose skeleton and the predicted pose action"""
    # make image larger if it was small
    image_height, image_width, _ = image.shape
    target_width = min(680, image_width*2)
    target_height = target_width * image.shape[0] // image.shape[1]
    image = cv2.resize(image, (target_width, target_height))

    resize_ratio = (target_width / image_width, target_height / image_height)

    image_height, image_width, _ = image.shape
    if display_image and detected_pose is not None:
      if display_landmarks:
        for landmark_x, landmark_y in np.int8(detected_pose.landmarks): # type: ignore
          landmark_x = min(int(landmark_x * resize_ratio[0]), target_width - 1)
          landmark_y = min(int(landmark_y * resize_ratio[1]), target_width - 1)
          cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

      # put text on head position
      head_coordinate = detected_pose.landmarks[0]
      head_coordinate[0] *= resize_ratio[0]
      head_coordinate[1] *= resize_ratio[1]
      ori_x = min(int(head_coordinate[0]), image_width) + 30
      ori_y = min(int(head_coordinate[1]), image_height)

      cv2.putText(image, label_pred, (ori_x, ori_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('detect pose result', image)

class MultiClassRNNPoseActionClassifier(PoseActionClassifier):
  def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, output_dim: int, window_length: int, epochs: int, lr: str):
    model_path = Path("models") / "RNN" / "MultiClass" / f"{input_dim}-{hidden_dim}-{layer_dim}-{output_dim}_WIN-{window_length}_EPOCH-{epochs}_LR-{lr}.pt"
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model: nn.Module = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(self.device).eval()
    self.model.load_state_dict(torch.load(model_path))
    self.window_length = window_length
    self.window = deque(maxlen=self.window_length)

  @torch.no_grad()
  def classify(self, pose: PoseDetectionResult)->str:
    flattened = pose.normalized_landmarks.flatten()
    if (flattened != 0).any():
      self.window.append(pose.normalized_landmarks.flatten())
    if len(self.window) < self.window_length:
      return "no_detection"
    pose_landmarks = torch.tensor(np.array(self.window)).unsqueeze(0).to(self.device)
    y_pred: torch.Tensor = self.model(pose_landmarks)
    label = OUTPUT_LABELS[torch.argmax(y_pred)]
    return label

class MultiClassLSTMPoseActionClassifier(PoseActionClassifier):
  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, window_length: int, epochs: int, lr: str):
    model_path = Path("models") / "LSTM" / "MultiClass" / f"{input_dim}-{hidden_dim}-{output_dim}_WIN-{window_length}_EPOCH-{epochs}_LR-{lr}.pt"
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model: nn.Module = LSTMModel(input_dim, hidden_dim, output_dim).to(self.device).eval()
    self.model.load_state_dict(torch.load(model_path))
    self.window_length = window_length
    self.window = deque(maxlen=self.window_length)

  @torch.no_grad()
  def classify(self, pose: PoseDetectionResult)->str:
    flattened = pose.normalized_landmarks.flatten()
    if (flattened != 0).any():
      self.window.append(pose.normalized_landmarks.flatten())
    if len(self.window) < self.window_length:
      return "no_detection"
    pose_landmarks = torch.tensor(np.array(self.window)).unsqueeze(0).to(self.device)
    y_pred: torch.Tensor = self.model(pose_landmarks)
    label = OUTPUT_LABELS[torch.argmax(y_pred)]
    return label
