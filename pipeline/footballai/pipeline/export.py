import json
import os
import pandas as pd
from footballai import schema

def write_artifacts(df: pd.DataFrame, meta: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = df[schema.FINAL_COLUMNS]
    df.to_parquet(os.path.join(out_dir, "tracks.parquet"), index=False)
    df.to_json(os.path.join(out_dir, "tracks.json"), orient="records")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def read_tracks(out_dir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(out_dir, "tracks.parquet"))
