import pandas as pd
from pathlib import Path
from typing import Literal

def load_KTH(pose_estimation_model: Literal["mediapipe", "ultralytics"])->pd.DataFrame:
  current_dir = Path(__file__).parent.absolute()
  return pd.read_csv(current_dir/"KTH_dataset.csv", engine="pyarrow")
