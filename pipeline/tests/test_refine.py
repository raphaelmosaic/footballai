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


def _ball_rows():
    """Ball at frame 0 and frame 2 (track_id=-1), missing frame 1. No players."""
    base = dict(track_id=-1, **{"class": "ball"}, team="", conf=0.9,
                bbox_x=5, bbox_y=5, bbox_w=1, bbox_h=1, img_x=100, img_y=100)
    return pd.DataFrame([
        {**base, "frame": 0, "pitch_x": 10.0, "pitch_y": 10.0},
        {**base, "frame": 2, "pitch_x": 10.0, "pitch_y": 10.0},
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


def test_ball_gap_is_interpolated():
    """Ball rows with track_id=-1 should get gap-filled and frame 1 marked interpolated."""
    cfg = load_config()
    df = _ball_rows()
    out = run_refine(df, fps=5.0, cfg=cfg)
    ball_f1 = out[(out["class"] == "ball") & (out["frame"] == 1)]
    assert len(ball_f1) == 1, "Expected exactly one ball row at frame 1 (interpolated gap)"
    assert ball_f1.iloc[0]["provenance"] == "interpolated"


def test_ball_duplicate_frame_is_deduped_then_interpolated():
    """Two ball rows at frame 0 (diff conf/pitch_x) + one at frame 2.
    After run_refine: frame 0 has exactly one ball row (highest conf=0.9 kept),
    frame 1 exists as interpolated, no exception."""
    cfg = load_config()
    base = dict(track_id=-1, **{"class": "ball"}, team="",
                bbox_x=5, bbox_y=5, bbox_w=1, bbox_h=1, img_x=100, img_y=100)
    df = pd.DataFrame([
        {**base, "frame": 0, "conf": 0.5, "pitch_x": 1.0, "pitch_y": 0.0},
        {**base, "frame": 0, "conf": 0.9, "pitch_x": 2.0, "pitch_y": 0.0},
        {**base, "frame": 2, "conf": 0.9, "pitch_x": 10.0, "pitch_y": 0.0},
    ])
    out = run_refine(df, fps=5.0, cfg=cfg)
    ball_f0 = out[(out["class"] == "ball") & (out["frame"] == 0)]
    assert len(ball_f0) == 1, f"Expected 1 ball row at frame 0, got {len(ball_f0)}"
    assert ball_f0.iloc[0]["pitch_x"] == 2.0, "Expected high-conf row (pitch_x=2.0) to be kept"
    ball_f1 = out[(out["class"] == "ball") & (out["frame"] == 1)]
    assert len(ball_f1) == 1, "Expected interpolated ball row at frame 1"
    assert ball_f1.iloc[0]["provenance"] == "interpolated"


def test_ball_near_player_is_possessed():
    """When a player is near the interpolated ball position, provenance should be 'possessed'."""
    cfg = load_config()
    ball_df = _ball_rows()
    player_base = dict(track_id=1, **{"class": "player"}, team="A", conf=0.9,
                       bbox_x=10, bbox_y=10, bbox_w=2, bbox_h=2, img_x=110, img_y=100)
    player_row = pd.DataFrame([{**player_base, "frame": 1, "pitch_x": 11.0, "pitch_y": 10.0}])
    df = pd.concat([ball_df, player_row], ignore_index=True)
    out = run_refine(df, fps=5.0, cfg=cfg)
    ball_f1 = out[(out["class"] == "ball") & (out["frame"] == 1)]
    assert len(ball_f1) == 1, "Expected exactly one ball row at frame 1"
    assert ball_f1.iloc[0]["provenance"] == "possessed", (
        f"Expected 'possessed' but got {ball_f1.iloc[0]['provenance']!r}"
    )
