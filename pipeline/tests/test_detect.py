import numpy as np
import pytest
import supervision as sv
from footballai.stages.detect import detections_to_rows
from footballai import schema


def test_detections_to_rows_maps_xywh_and_class():
    dets = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 60.0]]),  # x1,y1,x2,y2
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    rows = detections_to_rows(frame_idx=7, sv_detections=dets, class_names={0: "player"})
    assert len(rows) == 1
    r = rows[0]
    assert r["frame"] == 7
    assert r["class"] == "player"
    assert r["conf"] == pytest.approx(0.9)
    assert (r["bbox_x"], r["bbox_y"]) == (10.0, 20.0)
    assert (r["bbox_w"], r["bbox_h"]) == (20.0, 40.0)
    assert set(r.keys()) == set(schema.DETECTION_COLUMNS)
