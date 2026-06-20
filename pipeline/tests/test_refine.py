import pandas as pd
import numpy as np
from footballai.config import load_config
from footballai.pipeline.refine import interpolate_track, run_refine

def _track_with_gap():
    # track_id 1 present at frames 0 and 2, missing at 1
    base = dict(track_id=1, **{"class": "player"}, team="A", conf=0.9,
                bbox_x=0, bbox_y=0, bbox_w=2, bbox_h=2, img_x=0, img_y=0)
    return pd.DataFrame([
        {**base, "frame": 0, "pitch_x": 0.0, "pitch_y": 0.0},
        {**base, "frame": 2, "pitch_x": 10.0, "pitch_y": 20.0},
    ])

def test_interpolate_fills_midpoint():
    out = interpolate_track(_track_with_gap(), max_gap=90).sort_values("frame")
    assert list(out["frame"]) == [0, 1, 2]
    mid = out[out["frame"] == 1].iloc[0]
    assert mid["pitch_x"] == 5.0 and mid["pitch_y"] == 10.0
    assert mid["provenance"] == "interpolated"
    assert out[out["frame"] == 0].iloc[0]["provenance"] == "observed"

def test_gap_longer_than_max_is_not_filled():
    # gap = 1 missing frame; with max_gap=0 it is NOT filled
    out = interpolate_track(_track_with_gap(), max_gap=0).sort_values("frame")
    assert list(out["frame"]) == [0, 2]

def test_run_refine_adds_timestamp_and_provenance():
    cfg = load_config()
    out = run_refine(_track_with_gap(), fps=5.0, cfg=cfg)
    assert "timestamp" in out.columns and "provenance" in out.columns
    assert out[out["frame"] == 2].iloc[0]["timestamp"] == 0.4
