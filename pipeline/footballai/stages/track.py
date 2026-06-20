import numpy as np
import pandas as pd
import supervision as sv
from footballai.config import Config
from footballai import schema

_TRACKED = {"player", "goalkeeper"}


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def match_ids(
    tracked_xyxy: np.ndarray,
    updated_xyxy: np.ndarray,
    updated_ids: np.ndarray,
) -> np.ndarray:
    """Assign tracker IDs to input rows via greedy IoU matching.

    For each returned detection (updated_xyxy[i], updated_ids[i]), finds the
    input row in tracked_xyxy with the maximum IoU and assigns updated_ids[i]
    to that slot.  Each input slot is matched at most once (greedy by
    descending IoU).  Unmatched input rows receive -1.

    Parameters
    ----------
    tracked_xyxy:  (N, 4) input boxes [x1, y1, x2, y2]
    updated_xyxy:  (M, 4) boxes returned by the tracker
    updated_ids:   (M,)   tracker IDs for the returned boxes

    Returns
    -------
    (N,) array of int — tracker ID per input row, -1 if unmatched
    """
    n = len(tracked_xyxy)
    result = np.full(n, -1, dtype=int)
    if n == 0 or len(updated_xyxy) == 0:
        return result

    # Build (M, N) IoU matrix
    m = len(updated_xyxy)
    iou_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            iou_matrix[i, j] = _iou(updated_xyxy[i], tracked_xyxy[j])

    # Greedy one-to-one matching by descending IoU
    assigned_inputs = set()
    # Flatten, sort by IoU descending
    pairs = sorted(
        ((iou_matrix[i, j], i, j) for i in range(m) for j in range(n)),
        reverse=True,
    )
    assigned_updated = set()
    for iou_val, ui, ti in pairs:
        if iou_val <= 0.0:
            break
        if ui in assigned_updated or ti in assigned_inputs:
            continue
        result[ti] = updated_ids[ui]
        assigned_updated.add(ui)
        assigned_inputs.add(ti)

    return result


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
            tracked_xyxy = np.column_stack([
                tracked["bbox_x"].to_numpy(float),
                tracked["bbox_y"].to_numpy(float),
                (tracked["bbox_x"] + tracked["bbox_w"]).to_numpy(float),
                (tracked["bbox_y"] + tracked["bbox_h"]).to_numpy(float),
            ])
            tracked["track_id"] = match_ids(
                tracked_xyxy, updated.xyxy, updated.tracker_id
            )
        out_frames.append(pd.concat([tracked, untracked], ignore_index=True))
    out = pd.concat(out_frames, ignore_index=True)
    out = out[schema.TRACK_COLUMNS]
    schema.validate(out, schema.TRACK_COLUMNS)
    return out
