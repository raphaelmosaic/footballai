import numpy as np
import pandas as pd
from footballai.config import load_config
from footballai.stages.track import run_tracking, match_ids

def _moving_player(n_frames=8):
    # one player drifting right by 2px/frame, high confidence
    rows = []
    for f in range(n_frames):
        rows.append({"frame": f, "class": "player", "conf": 0.9,
                     "bbox_x": 100 + 2 * f, "bbox_y": 100, "bbox_w": 10, "bbox_h": 30})
    return pd.DataFrame(rows)

def test_single_player_keeps_one_id():
    cfg = load_config()
    out = run_tracking(_moving_player(), cfg)
    player_ids = set(out[out["class"] == "player"]["track_id"])
    assert len(player_ids) == 1
    assert all(i > 0 for i in player_ids)

def test_ball_rows_are_passthrough_untracked():
    cfg = load_config()
    df = pd.DataFrame([{"frame": 0, "class": "ball", "conf": 0.8,
                        "bbox_x": 5, "bbox_y": 5, "bbox_w": 3, "bbox_h": 3}])
    out = run_tracking(df, cfg)
    assert out.iloc[0]["track_id"] == -1

def test_match_ids_robust_to_reordering_and_dropping():
    # Three input boxes A, B, C at clearly separated coordinates
    # A: top-left area, B: middle, C: right
    box_A = np.array([[ 10,  10,  50,  50]])  # slot 0
    box_B = np.array([[200, 200, 260, 260]])  # slot 1
    box_C = np.array([[400, 100, 480, 180]])  # slot 2

    tracked_xyxy = np.vstack([box_A, box_B, box_C])  # shape (3, 4)

    # ByteTrack drops A and returns C then B (reversed order), with ids [33, 22]
    updated_xyxy = np.vstack([box_C, box_B])  # C first, then B
    updated_ids  = np.array([33, 22])          # id 33->C, id 22->B

    result = match_ids(tracked_xyxy, updated_xyxy, updated_ids)

    assert result.shape == (3,)
    assert result[0] == -1   # A was dropped -> -1
    assert result[1] == 22   # B matched by IoU -> id 22
    assert result[2] == 33   # C matched by IoU -> id 33
