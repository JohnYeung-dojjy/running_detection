"""Main program for the play audio when see person running"""
from __future__ import annotations
from pose_detector import UltralyticsPoseDetector, PoseDetectionResult
from vidio_stream_manager import VideoStreamManager
from audio_player import AudioPlayer
from action_classifier import RNNPoseActionClassifier
import os
from dotenv import load_dotenv
from pathlib import Path
import time

def main(
  display_image: bool,
  display_landmarks: bool,
  fps: int,
  camera_id: int|str|None=None,
  video_file: Path|str|None=None,
  ):
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
  audio_path = Path(__file__).parent / "audio" / os.environ["AUDIO_FILE"]
  audio_player = AudioPlayer(audio_path=audio_path)
  video_stream = VideoStreamManager(camera_id=camera_id,
                                    video_file=video_file,
                                    fps=fps)
  pose_detector = UltralyticsPoseDetector('x')
  action_classifier = RNNPoseActionClassifier(22, 64, 2, 4, 4, 100, "0.01")

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

if __name__ == "__main__":
  load_dotenv()
  main(
    # camera_id=0,
    video_file="Dataset/KTH/running/person02_running_d1_uncomp.avi",
    fps=25,
    display_image=True,
    display_landmarks=True
  )
