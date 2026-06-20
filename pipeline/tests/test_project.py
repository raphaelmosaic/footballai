import numpy as np
import pandas as pd
from footballai.stages.project import foot_points, project_points, run_projection

def test_foot_point_is_bottom_center():
    df = pd.DataFrame([{"bbox_x": 100, "bbox_y": 200, "bbox_w": 20, "bbox_h": 40}])
    assert np.allclose(foot_points(df), [[110, 240]])

def test_project_identity_homography_returns_input():
    H = np.eye(3)
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert np.allclose(project_points(H, pts), pts)

def test_run_projection_adds_pitch_columns():
    df = pd.DataFrame([{
        "frame": 0, "class": "player", "conf": 0.9, "track_id": 1, "team": "A",
        "bbox_x": 0, "bbox_y": 0, "bbox_w": 2, "bbox_h": 2,
    }])
    out = run_projection(df, np.eye(3))
    assert {"img_x", "img_y", "pitch_x", "pitch_y"}.issubset(out.columns)
    assert out.iloc[0]["img_x"] == 1 and out.iloc[0]["img_y"] == 2
    assert out.iloc[0]["pitch_x"] == 1 and out.iloc[0]["pitch_y"] == 2
