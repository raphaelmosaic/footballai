import os
import numpy as np
import pandas as pd
import pytest
from footballai.run import resolve_stage_range, STAGES, run_pipeline
from footballai.config import load_config
from footballai import schema
from tests.fixtures.make_clip import make_clip

def test_full_range_default():
    assert resolve_stage_range(None, None) == STAGES

def test_subrange_inclusive():
    assert resolve_stage_range("track", "project") == ["track", "teams", "project"]

def test_invalid_stage_raises():
    with pytest.raises(ValueError):
        resolve_stage_range("nope", None)

def test_partial_run_refine_to_export_needs_no_calib(tmp_path):
    # Build a minimal project.parquet with PROJECTED_COLUMNS
    rows = [
        {"frame": 0, "class": "player", "conf": 0.9,
         "bbox_x": 10.0, "bbox_y": 10.0, "bbox_w": 20.0, "bbox_h": 40.0,
         "track_id": 1, "team": 0,
         "img_x": 100.0, "img_y": 200.0, "pitch_x": 5.0, "pitch_y": 10.0},
        {"frame": 2, "class": "player", "conf": 0.9,
         "bbox_x": 12.0, "bbox_y": 10.0, "bbox_w": 20.0, "bbox_h": 40.0,
         "track_id": 1, "team": 0,
         "img_x": 102.0, "img_y": 200.0, "pitch_x": 5.2, "pitch_y": 10.0},
    ]
    df = pd.DataFrame(rows, columns=schema.PROJECTED_COLUMNS)
    work = str(tmp_path / "work")
    os.makedirs(work, exist_ok=True)
    df.to_parquet(os.path.join(work, "project.parquet"), index=False)

    # Create a tiny video so read_meta works
    clip = str(tmp_path / "clip.mp4")
    make_clip(clip, n=10, w=64, h=48, fps=5)

    # Run refine→export with a nonexistent calib file — must not raise
    run_pipeline(
        video_path=clip,
        calib_path="/nonexistent/does_not_exist.json",
        cfg=load_config(),
        work_dir=work,
        start="refine",
        end="export",
    )

    # Both output artifacts must exist
    assert os.path.exists(os.path.join(work, "refine.parquet"))
    assert os.path.exists(os.path.join(work, "tracks.parquet"))
