import numpy as np
import pandas as pd
from footballai import schema

def foot_points(df: pd.DataFrame) -> np.ndarray:
    x = df["bbox_x"].to_numpy(float) + df["bbox_w"].to_numpy(float) / 2.0
    y = df["bbox_y"].to_numpy(float) + df["bbox_h"].to_numpy(float)
    return np.column_stack([x, y])

def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ homog.T).T
    return proj[:, :2] / proj[:, 2:3]

def run_projection(teams: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    out = teams.copy()
    fp = foot_points(out)
    out["img_x"], out["img_y"] = fp[:, 0], fp[:, 1]
    pitch = project_points(H, fp)
    out["pitch_x"], out["pitch_y"] = pitch[:, 0], pitch[:, 1]
    out = out[schema.PROJECTED_COLUMNS]
    schema.validate(out, schema.PROJECTED_COLUMNS)
    return out
