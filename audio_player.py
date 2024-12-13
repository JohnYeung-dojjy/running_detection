from __future__ import annotations
from contextlib import redirect_stdout
from pathlib import Path
import os
from io import StringIO
from os import PathLike
import time
import logging

with redirect_stdout(StringIO()) as pygame_console_output: # do not print pygame init message to console
  # use built-in pygame audio module to play/pause/resume/stop audio on request
  # TODO: try to see if pypat (https://github.com/tnewman/pat#Prerequisites) is a better alternative
  from pygame import mixer
  mixer.init()

# https://refactoring.guru/design-patterns/singleton/python/example
class SingletonMeta(type):
  """
  The Singleton class can be implemented in different ways in Python. Some
  possible methods include: base class, decorator, metaclass. We will use the
  metaclass because it is best suited for this purpose.
  """

  _instances = {}

  def __call__(cls, *args, **kwargs):
    """
    Possible changes to the value of the `__init__` argument do not affect
    the returned instance.
    """
    if cls not in cls._instances:
      instance = super().__call__(*args, **kwargs)
      cls._instances[cls] = instance
    return cls._instances[cls]

class AudioPlayer(metaclass=SingletonMeta):
  audio_player = mixer.music
  def __init__(self, audio_path: PathLike|str):
    self.audio_player.load(audio_path)
    self._volume: float = 0.3
    self.started: bool = False
    self.pause_time: float|None = None
    self._restart_time_threshold: float = 5 # seconds passed to restart the music instead of resume
    self.is_playing: bool = False


  @property
  def volume(self):
    return self._volume

  @volume.setter
  def volume(self, volume):
    self._volume = volume
    self.audio_player.set_volume(volume)

  def pause(self):
    if self.is_playing:
      self.is_playing = False
      self.audio_player.pause()
      self.pause_time = time.time()

  def _resume(self):
    # self.pause_time is already set at this point
    stopped_for_too_long = time.time() - self.pause_time < self._restart_time_threshold # type: ignore
    if stopped_for_too_long:
      self.audio_player.unpause()
    else:
      self.audio_player.rewind()
      self.audio_player.unpause()


  def resume(self):
    if not self.started:
      # logging.debug("starting music")
      # print("starting music")
      self.audio_player.play()
      self.started = True
      self.is_playing = True
    elif not self.is_playing:
      self._resume()
      self.is_playing = True

  def stop(self):
    self.audio_player.stop()

def test():
  audio_path = Path(__file__).parent / "audio" / os.environ["AUDIO_FILE"]
  audio_player = AudioPlayer(audio_path)
  audio_player2 = AudioPlayer(audio_path)
  assert id(audio_player) == id(audio_player2), "singleton class failed"
  while True:
    print("Press 'p' to pause")
    print("Press 'r' to resume")
    print("Press 'v' set volume")
    print("Press 'e' to exit")
    ch = input("['p','r','v','e']>>>")
    if ch == "p":
      audio_player.pause()
    elif ch == "r":
      audio_player.resume()
    elif ch == "v":
      v = float(input("Enter volume(0 to 1): "))
      audio_player.volume = v
    elif ch == "e":
      audio_player.stop()
      break

if __name__ == "__main__":
  from dotenv import load_dotenv
  load_dotenv()
  test()