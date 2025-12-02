import pandas as pd
from pathlib import Path
from pipeline.config import DATA_DIR

def load_traffic_data(file_name="df_brut.csv"):
    file_path = Path(DATA_DIR) / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} n'existe pas !")
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df
