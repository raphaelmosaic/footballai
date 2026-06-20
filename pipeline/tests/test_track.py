import pandas as pd
from footballai.config import load_config
from footballai.pipeline.track import run_tracking

def _moving_player(n_frames=8):
    # one player drifting right by 2px/frame, high confidence
    rows = []
    for f in range(n_frames):
        rows.append({"frame": f, "class": "player", "conf": 0.9,
                     "bbox_x": 100 + 2 * f, "bbox_y": 100, "bbox_w": 10, "bbox_h": 30})
    return pd.DataFrame(rows)

def test_single_player_keeps_one_id(tmp_path):
    cfg = load_config()
    out = run_tracking(_moving_player(), cfg)
    player_ids = set(out[out["class"] == "player"]["track_id"])
    assert player_ids == {min(player_ids)}  # exactly one id
    assert all(i > 0 for i in player_ids)

def test_ball_rows_are_passthrough_untracked():
    cfg = load_config()
    df = pd.DataFrame([{"frame": 0, "class": "ball", "conf": 0.8,
                        "bbox_x": 5, "bbox_y": 5, "bbox_w": 3, "bbox_h": 3}])
    out = run_tracking(df, cfg)
    assert out.iloc[0]["track_id"] == -1
