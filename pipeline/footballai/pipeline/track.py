import numpy as np
import pandas as pd
import supervision as sv
from footballai.config import Config
from footballai import schema

_TRACKED = {"player", "goalkeeper"}

def _rows_to_detections(rows: pd.DataFrame) -> sv.Detections:
    xyxy = np.column_stack([
        rows["bbox_x"], rows["bbox_y"],
        rows["bbox_x"] + rows["bbox_w"], rows["bbox_y"] + rows["bbox_h"],
    ]).astype(float)
    return sv.Detections(xyxy=xyxy, confidence=rows["conf"].to_numpy(float),
                         class_id=np.zeros(len(rows), dtype=int))

def run_tracking(detections: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    tracker = sv.ByteTrack(
        track_activation_threshold=cfg.track["activation_threshold"],
        lost_track_buffer=cfg.track["lost_track_buffer"],
    )
    out_frames: list[pd.DataFrame] = []
    for frame_idx in sorted(detections["frame"].unique()):
        fr = detections[detections["frame"] == frame_idx]
        tracked = fr[fr["class"].isin(_TRACKED)].reset_index(drop=True)
        untracked = fr[~fr["class"].isin(_TRACKED)].copy()
        untracked["track_id"] = -1
        if len(tracked):
            dets = _rows_to_detections(tracked)
            updated = tracker.update_with_detections(dets)
            tracked = tracked.copy()
            ids = np.full(len(tracked), -1, dtype=int)
            # supervision returns detections in the same order it received them
            ids[: len(updated.tracker_id)] = updated.tracker_id
            tracked["track_id"] = ids
        out_frames.append(pd.concat([tracked, untracked], ignore_index=True))
    out = pd.concat(out_frames, ignore_index=True)
    out = out[schema.TRACK_COLUMNS]
    schema.validate(out, schema.TRACK_COLUMNS)
    return out
