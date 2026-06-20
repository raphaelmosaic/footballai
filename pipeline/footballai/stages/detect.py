import pandas as pd
import supervision as sv
from footballai.config import Config
from footballai.stages.extract import iter_frames
from footballai import schema


def detections_to_rows(frame_idx, sv_detections, class_names) -> list[dict]:
    rows = []
    xyxy = sv_detections.xyxy
    conf = sv_detections.confidence
    cls = sv_detections.class_id
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        rows.append({
            "frame": int(frame_idx),
            "class": class_names[int(cls[i])],
            "conf": float(conf[i]) if conf is not None else 1.0,
            "bbox_x": x1,
            "bbox_y": y1,
            "bbox_w": x2 - x1,
            "bbox_h": y2 - y1,
        })
    return rows


def run_detection(video_path: str, cfg: Config) -> pd.DataFrame:
    from ultralytics import YOLO
    model = YOLO(cfg.detect["weights"])
    class_names = model.names
    all_rows: list[dict] = []
    for frame_idx, frame in iter_frames(video_path):
        result = model(frame, conf=cfg.detect["conf"], imgsz=cfg.detect["imgsz"], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(result)
        all_rows.extend(detections_to_rows(frame_idx, dets, class_names))
    df = pd.DataFrame(all_rows, columns=schema.DETECTION_COLUMNS)
    schema.validate(df, schema.DETECTION_COLUMNS)
    return df
