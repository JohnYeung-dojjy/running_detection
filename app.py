"""Main program for the play audio when see person running"""
from __future__ import annotations
from .PoseDetector import SinglePersonPoseDetector, PoseDetectionResult
from .VideoStreamManager import VideoStreamManager
from .AudioPlayer import AudioPlayer
from .ActionClassifier import PoseActionClassifier
from os import PathLike
import time

def main(camera_id: int|str|None=None, 
         video_file: PathLike|str|None=None,
         fps: int=30,
         audio_path: PathLike|str|None=None,
         jogging: str="",
         display_image: bool=False,
         display_landmarks: bool=False):
  """play audio if detected people running in camera/video
  1 detect pose in frame (from camera/video)
  2 if pose is not detected (detected_pose is None) then
      - pause audio stream
    else 
      - detect action from pose information (walking, jogging, running)
      - if action is running then
          - resume (play) audio stream
        else
          - pause audio stream

  Args:
      camera_id (int | str | None, optional): The camera id to take input from. Defaults to None.
      video_file (PathLike | str | None, optional): The video path to take input from. Defaults to None.
      fps (int, optional): frame per second. Defaults to 30.
      audio_path (PathLike | str | None, optional): path of the played audio. Defaults to None (loads default umamusume theme song).
      jogging (str, optional): Choose SVM model according to how it was trained. Defaults to "" (jogging data were ignored).
      display_image (bool, optional): Whether to display the frame. Defaults to False.
      display_landmarks (bool, optional): Whether to display the detect landmarks. Defaults to False.
  """
  audio_player = AudioPlayer(audio_path=audio_path)
  video_stream = VideoStreamManager(camera_id=camera_id,
                                    video_file=video_file,
                                    fps=fps)
  pose_detector = SinglePersonPoseDetector()
  action_classifier = PoseActionClassifier(jogging=jogging)
  
  last_resume_time = time.time()
  
  for frame in video_stream.read_frames():
    img=frame.copy()
    detected_pose: PoseDetectionResult|None = pose_detector.detect(img)
                                                                  #  display_image=display_image,
                                                                  #  display_landmarks=display_landmarks)
    if detected_pose is None: # continue to next frame if no pose detected
      audio_player.pause()
      action = ""
      # continue  
    else:
      action = action_classifier.classify(detected_pose)
      if action == "running":
        audio_player.resume()
        last_resume_time = time.time()
      else:
        current_time = time.time()
        if current_time - last_resume_time < 1: continue
        #  last_resume_time = current_time
        audio_player.pause()
        # audio_player.resume()

    action_classifier.display_image(img, action, detected_pose, display_image, display_landmarks)

"""Relative import error will occur when trying to run this file as main, try importing app.main() from a script in parent directory"""
# if __name__ == "__main__":
#   main(camera_id=0,
#        # video_file="Dataset/KTH/running/person02_running_d1_uncomp.avi",
#        # fps=60,
#        display_image=True,
#        display_landmarks=True)