import numpy as np
import pandas as pd
from footballai.config import Config
from footballai import schema

_INTERP_COLS = ["pitch_x", "pitch_y", "img_x", "img_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def interpolate_track(track: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    track = track.sort_values(["frame", "conf"]).copy()
    track = track.drop_duplicates(subset="frame", keep="last")
    track = track.sort_values("frame").reset_index(drop=True)
    track["provenance"] = "observed"
    frames = track["frame"].to_numpy()
    rows = [track]
    for a, b in zip(frames[:-1], frames[1:]):
        gap = int(b - a) - 1
        if gap <= 0 or gap > max_gap:
            continue
        row_a = track[track["frame"] == a].iloc[0]
        row_b = track[track["frame"] == b].iloc[0]
        for k in range(1, gap + 1):
            t = k / (gap + 1)
            new = row_a.copy()
            new["frame"] = a + k
            for c in _INTERP_COLS:
                new[c] = (1 - t) * row_a[c] + t * row_b[c]
            new["conf"] = 0.0
            new["provenance"] = "interpolated"
            rows.append(pd.DataFrame([new]))
    out = pd.concat(rows, ignore_index=True).sort_values("frame").reset_index(drop=True)
    return out


def ball_state(df: pd.DataFrame, possession_radius: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    ball = df["class"] == "ball"
    players = df[df["class"] != "ball"]
    for idx in df[ball & (df["provenance"] == "interpolated")].index:
        row = df.loc[idx]
        same_frame = players[players["frame"] == row["frame"]]
        if len(same_frame):
            d = np.hypot(same_frame["pitch_x"] - row["pitch_x"],
                         same_frame["pitch_y"] - row["pitch_y"])
            if d.min() <= possession_radius:
                df.at[idx, "provenance"] = "possessed"
    return df


def run_refine(projected: pd.DataFrame, fps: float, cfg: Config) -> pd.DataFrame:
    max_gap = cfg.refine["max_gap"]
    possession_radius = cfg.refine.get("possession_radius", 2.0)
    parts = []
    # Players and goalkeepers with real track IDs: interpolate per track
    tracked = projected[projected["track_id"] > 0]
    for tid, grp in tracked.groupby("track_id"):
        parts.append(interpolate_track(grp, max_gap))
    # Ball: treat all ball rows as a single track so gaps get filled
    ball = projected[projected["class"] == "ball"]
    if len(ball):
        parts.append(interpolate_track(ball, max_gap))
    # Other untracked non-ball rows: pass through as observed
    other_untracked = projected[(projected["track_id"] <= 0) & (projected["class"] != "ball")].copy()
    other_untracked["provenance"] = "observed"
    if len(other_untracked):
        parts.append(other_untracked)
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = out["frame"] / fps
    out = ball_state(out, possession_radius)
    out = out[schema.FINAL_COLUMNS].sort_values(["frame", "track_id"]).reset_index(drop=True)
    schema.validate(out, schema.FINAL_COLUMNS)
    return out
