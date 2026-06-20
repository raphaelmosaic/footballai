# tests/test_e2e_smoke.py
import numpy as np
import pandas as pd
from footballai.config import load_config
from footballai import schema
from footballai.stages.project import run_projection
from footballai.stages.refine import run_refine
from footballai.stages.export import write_artifacts, read_tracks


def test_project_refine_export_smoke(tmp_path):
    cfg = load_config()
    teams_df = pd.DataFrame([
        {"frame": 0, "class": "player", "conf": 0.9, "track_id": 1, "team": "A",
         "bbox_x": 10, "bbox_y": 10, "bbox_w": 4, "bbox_h": 8},
        {"frame": 2, "class": "player", "conf": 0.9, "track_id": 1, "team": "A",
         "bbox_x": 30, "bbox_y": 10, "bbox_w": 4, "bbox_h": 8},
        {"frame": 0, "class": "ball", "conf": 0.7, "track_id": -1, "team": None,
         "bbox_x": 12, "bbox_y": 18, "bbox_w": 2, "bbox_h": 2},
    ])
    H = np.eye(3)
    projected = run_projection(teams_df, H)
    final = run_refine(projected, fps=5.0, cfg=cfg)
    write_artifacts(final, {"fps": 5.0}, str(tmp_path))
    back = read_tracks(str(tmp_path))
    assert list(back.columns) == schema.FINAL_COLUMNS
    assert len(back) > 0
    # interpolation produced the missing frame-1 row for track 1
    assert ((back["track_id"] == 1) & (back["frame"] == 1)).any()
