import pandas as pd

CLASSES = ("player", "goalkeeper", "referee", "ball")
PROVENANCE = ("observed", "predicted", "interpolated", "possessed")

DETECTION_COLUMNS = ["frame", "class", "conf", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
TRACK_COLUMNS = DETECTION_COLUMNS + ["track_id"]
TEAM_COLUMNS = TRACK_COLUMNS + ["team"]
PROJECTED_COLUMNS = TEAM_COLUMNS + ["img_x", "img_y", "pitch_x", "pitch_y"]
FINAL_COLUMNS = PROJECTED_COLUMNS + ["timestamp", "provenance"]

def validate(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
