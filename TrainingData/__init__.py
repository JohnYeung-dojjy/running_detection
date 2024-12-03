import pandas as pd
from pathlib import Path
def load_KTH()->pd.DataFrame:
  current_dir = Path(__file__).parent.absolute()
  return pd.read_csv(current_dir/"KTH_dataset.csv", engine="pyarrow")
  