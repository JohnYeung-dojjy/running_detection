# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from VideoStreamManager import VideoStreamManager



def main():
  # referenced from: https://github.com/nicknochnack/MultiPoseMovenetLightning/blob/main/MultiPose%20MoveNet%20Tutorial.ipynb
  video_stream = VideoStreamManager(video_file="../Dataset/KTH/walking/person01_walking_d1_uncomp.avi")
  # video_stream = VideoStreamManager(camera_id=0, fps=60)
  # gpus = tf.config.list_physical_devices('GPU')
  # print(gpus)
  with tf.device("/GPU:0"): # type: ignore
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    # model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    movenet = model.signatures['serving_default']
    for frame in video_stream.read_frames():
      print(frame.shape)
      img = frame.copy()
      img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640) # pad a 128x128 image with 0
      input_img = tf.cast(img, dtype=tf.int32)
      
      # Detection section
      results = movenet(input_img)
      # print(results)
      # 1. only keey the first 51 data (x, y, confidence) * (17 keypoints), other are bounding box coordinates
      # 2. reshape it
      keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
      
      # Render keypoints 
      # frame = cv2.resize(frame, (512, 512))
      loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)
      
      cv2.imshow('Movenet Multipose', frame)


EDGES = {
    (0, 1) : 'm', (0, 2)  : 'c', (1, 3)  : 'm', (2, 4)  : 'c', (0, 5)  : 'm', (0, 6)  : 'c',
    (5, 7) : 'm', (7, 9)  : 'm', (6, 8)  : 'c', (8, 10) : 'c', (5, 6)  : 'y', (5, 11) : 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}
     

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
        
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for keypoint in shaped:
        keypoint_y, keypoint_x, keypoint_confidence = keypoint
        cv2.circle(frame, (int(keypoint_x), int(keypoint_y)), 6, (0,255,0), -1)
        if keypoint_confidence > confidence_threshold:
            cv2.circle(frame, (int(keypoint_x), int(keypoint_y)), 6, (0,255,0), -1)
       
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)
            
if __name__ == "__main__":
  main()