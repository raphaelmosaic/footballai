# Soccer Video Analysis — Core Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline Python pipeline that turns a fixed-camera soccer video into a long/tidy table of per-frame player and ball positions in pitch meters, with persistent IDs, team labels, and occlusion provenance.

**Architecture:** A sequence of stages (`extract → detect → track → teams → calibrate → project → refine → export`) that each read and write an on-disk artifact (Parquet), orchestrated by `run.py` with `--from/--to` partial runs. Deterministic cores (pitch geometry, homography, projection, interpolation, tracking logic) are unit-tested; model-inference stages (YOLO, SigLIP) get fixture smoke-tests.

**Tech Stack:** Python 3.11, Ultralytics YOLOv8, `supervision` (ByteTrack), SigLIP via `transformers`, `umap-learn` + scikit-learn (team clustering), OpenCV (homography/IO), `filterpy` (Kalman), pandas + pyarrow (artifacts), PyYAML (config), pytest.

## Global Constraints

- Python 3.11+; all code type-hinted.
- Inter-stage artifacts are Parquet files written with pyarrow; the canonical in-memory form is a pandas `DataFrame`.
- Pitch dimensions: **105 m × 68 m**; origin `(0,0)` at one corner, x along the 105 m touchline, y along the 68 m goal line.
- Class vocabulary is exactly: `player`, `goalkeeper`, `referee`, `ball`.
- Provenance vocabulary is exactly: `observed`, `predicted`, `interpolated`, `possessed`.
- Every threshold and weight path lives in `config.yaml`; no magic numbers in code.
- All randomized steps (KMeans, UMAP) seed from `config.seed` for reproducibility.

---

### Task 1: Project skeleton, dependencies, config

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `footballai/__init__.py`
- Create: `footballai/config.py`
- Create: `tests/test_config.py`
- Create: `tests/fixtures/.gitkeep`

**Interfaces:**
- Produces: `load_config(path: str = "config.yaml") -> Config` where `Config` is a dataclass with attributes `seed: int`, `pitch_length: float`, `pitch_width: float`, `paths: dict`, `detect: dict`, `track: dict`, `teams: dict`, `refine: dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from footballai.config import load_config

def test_load_config_reads_pitch_and_seed(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "seed: 42\n"
        "pitch_length: 105.0\n"
        "pitch_width: 68.0\n"
        "paths: {work_dir: work}\n"
        "detect: {weights: w.pt, conf: 0.25}\n"
        "track: {}\n"
        "teams: {}\n"
        "refine: {}\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.seed == 42
    assert cfg.pitch_length == 105.0
    assert cfg.detect["conf"] == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.config'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "footballai"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "ultralytics>=8.2",
  "supervision>=0.22",
  "opencv-python>=4.9",
  "pandas>=2.2",
  "pyarrow>=16",
  "numpy>=1.26",
  "pyyaml>=6",
  "filterpy>=1.4.5",
  "scikit-learn>=1.5",
  "umap-learn>=0.5.6",
  "transformers>=4.44",
  "torch>=2.2",
  "pillow>=10",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# footballai/__init__.py
```

```python
# footballai/config.py
from dataclasses import dataclass
import yaml

@dataclass
class Config:
    seed: int
    pitch_length: float
    pitch_width: float
    paths: dict
    detect: dict
    track: dict
    teams: dict
    refine: dict

def load_config(path: str = "config.yaml") -> "Config":
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=raw.get("seed", 0),
        pitch_length=raw["pitch_length"],
        pitch_width=raw["pitch_width"],
        paths=raw.get("paths", {}),
        detect=raw.get("detect", {}),
        track=raw.get("track", {}),
        teams=raw.get("teams", {}),
        refine=raw.get("refine", {}),
    )
```

```yaml
# config.yaml
seed: 42
pitch_length: 105.0
pitch_width: 68.0
paths:
  work_dir: work        # per-video intermediate artifacts go here
detect:
  weights: weights/football-players.pt
  conf: 0.25
  imgsz: 1280
  ball_class: ball
track:
  activation_threshold: 0.25
  lost_track_buffer: 60   # frames a track may coast before being dropped
teams:
  siglip_model: google/siglip-base-patch16-224
  umap_components: 3
refine:
  max_gap: 90             # frames; gaps longer than this are not interpolated
  process_var: 1.0        # Kalman process noise
  measurement_var: 4.0    # Kalman measurement noise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.yaml footballai/ tests/
git commit -m "feat: project skeleton, dependencies, and config loader"
```

---

### Task 2: Pitch geometry (single source of truth)

**Files:**
- Create: `footballai/pitch.py`
- Create: `tests/test_pitch.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PITCH_LENGTH: float = 105.0`, `PITCH_WIDTH: float = 68.0`
  - `LANDMARKS: dict[str, tuple[float, float]]` — named pitch points in meters (corners, penalty-box corners, center spot, center-circle cardinal points).
  - `landmark_array(names: list[str]) -> np.ndarray` returning an `(N,2)` float array of the named landmarks in order.
  - `in_bounds(x: float, y: float, margin: float = 0.0) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pitch.py
import numpy as np
from footballai import pitch

def test_corners_are_pitch_extents():
    assert pitch.LANDMARKS["corner_bl"] == (0.0, 0.0)
    assert pitch.LANDMARKS["corner_tr"] == (105.0, 68.0)

def test_landmark_array_order_and_shape():
    arr = pitch.landmark_array(["corner_bl", "corner_tr"])
    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[0, 0], [105, 68]])

def test_in_bounds_respects_margin():
    assert pitch.in_bounds(50, 34)
    assert not pitch.in_bounds(-1, 34)
    assert pitch.in_bounds(-1, 34, margin=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pitch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pitch'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pitch.py
import numpy as np

PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0

# Meters. Origin (0,0) = bottom-left corner; x along 105 m side, y along 68 m side.
LANDMARKS: dict[str, tuple[float, float]] = {
    "corner_bl": (0.0, 0.0),
    "corner_tl": (0.0, 68.0),
    "corner_br": (105.0, 0.0),
    "corner_tr": (105.0, 68.0),
    "pen_bl": (16.5, 13.84),
    "pen_tl": (16.5, 54.16),
    "pen_br": (88.5, 13.84),
    "pen_tr": (88.5, 54.16),
    "center": (52.5, 34.0),
    "center_top": (52.5, 43.15),
    "center_bottom": (52.5, 24.85),
}

def landmark_array(names: list[str]) -> np.ndarray:
    return np.array([LANDMARKS[n] for n in names], dtype=float)

def in_bounds(x: float, y: float, margin: float = 0.0) -> bool:
    return (-margin <= x <= PITCH_LENGTH + margin) and (
        -margin <= y <= PITCH_WIDTH + margin
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pitch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/pitch.py tests/test_pitch.py
git commit -m "feat: pitch geometry as single source of truth"
```

---

### Task 3: Artifact schema & validation

**Files:**
- Create: `footballai/schema.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DETECTION_COLUMNS`, `TRACK_COLUMNS`, `TEAM_COLUMNS`, `PROJECTED_COLUMNS`, `FINAL_COLUMNS` — lists of column names (each a superset of the prior).
  - `CLASSES = ("player", "goalkeeper", "referee", "ball")`
  - `PROVENANCE = ("observed", "predicted", "interpolated", "possessed")`
  - `validate(df: pd.DataFrame, columns: list[str]) -> None` — raises `ValueError` if required columns are missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
import pandas as pd
import pytest
from footballai import schema

def test_final_columns_superset_of_detection():
    assert set(schema.DETECTION_COLUMNS).issubset(set(schema.FINAL_COLUMNS))

def test_validate_raises_on_missing_columns():
    df = pd.DataFrame({"frame": [0]})
    with pytest.raises(ValueError, match="missing columns"):
        schema.validate(df, schema.DETECTION_COLUMNS)

def test_validate_passes_when_columns_present():
    df = pd.DataFrame({c: [] for c in schema.DETECTION_COLUMNS})
    schema.validate(df, schema.DETECTION_COLUMNS)  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/schema.py
import pandas as pd

CLASSES = ("player", "goalkeeper", "referee", "ball")
PROVENANCE = ("observed", "predicted", "interpolated", "possessed")

DETECTION_COLUMNS = ["frame", "class", "conf", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
TRACK_COLUMNS = DETECTION_COLUMNS + ["track_id"]
TEAM_COLUMNS = TRACK_COLUMNS + ["team"]
PROJECTED_COLUMNS = TEAM_COLUMNS + ["img_x", "img_y", "pitch_x", "pitch_y"]
FINAL_COLUMNS = PROJECTED_COLUMNS + ["timestamp", "provenance"]

def validate(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/schema.py tests/test_schema.py
git commit -m "feat: artifact column schema and validation"
```

---

### Task 4: Frame extraction

**Files:**
- Create: `footballai/pipeline/__init__.py`
- Create: `footballai/pipeline/extract.py`
- Create: `tests/test_extract.py`
- Create: `tests/fixtures/make_clip.py` (helper to synthesize a tiny video)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `VideoMeta` dataclass: `fps: float`, `width: int`, `height: int`, `n_frames: int`.
  - `read_meta(video_path: str) -> VideoMeta`.
  - `iter_frames(video_path: str) -> Iterator[tuple[int, np.ndarray]]` yielding `(frame_index, bgr_image)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/fixtures/make_clip.py
import cv2, numpy as np

def make_clip(path: str, n: int = 10, w: int = 64, h: int = 48, fps: int = 5) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
```

```python
# tests/test_extract.py
from tests.fixtures.make_clip import make_clip
from footballai.pipeline.extract import read_meta, iter_frames

def test_read_meta(tmp_path):
    clip = str(tmp_path / "c.mp4")
    make_clip(clip, n=10, w=64, h=48, fps=5)
    meta = read_meta(clip)
    assert meta.width == 64 and meta.height == 48
    assert meta.fps == 5.0
    assert meta.n_frames == 10

def test_iter_frames_yields_indexed_frames(tmp_path):
    clip = str(tmp_path / "c.mp4")
    make_clip(clip, n=6)
    frames = list(iter_frames(clip))
    assert [i for i, _ in frames] == [0, 1, 2, 3, 4, 5]
    assert frames[0][1].shape[2] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pipeline.extract'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/__init__.py
```

```python
# footballai/pipeline/extract.py
from dataclasses import dataclass
from typing import Iterator
import cv2
import numpy as np

@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int

def read_meta(video_path: str) -> VideoMeta:
    cap = cv2.VideoCapture(video_path)
    try:
        return VideoMeta(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()

def iter_frames(video_path: str) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extract.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/ tests/test_extract.py tests/fixtures/make_clip.py
git commit -m "feat: video frame extraction and metadata"
```

---

### Task 5: Detection (YOLO wrapper)

**Files:**
- Create: `footballai/pipeline/detect.py`
- Create: `tests/test_detect.py`

**Interfaces:**
- Consumes: `Config`, `iter_frames`, `DETECTION_COLUMNS`.
- Produces:
  - `detections_to_rows(frame_idx, sv_detections, class_names) -> list[dict]` — pure function mapping a `supervision.Detections` object to schema rows. (Unit-tested.)
  - `run_detection(video_path: str, cfg: Config) -> pd.DataFrame` — full inference, returns `DETECTION_COLUMNS` DataFrame. (Smoke-tested only when weights exist.)

- [ ] **Step 1: Write the failing test** (pure mapping logic — no model needed)

```python
# tests/test_detect.py
import numpy as np
import pytest
import supervision as sv
from footballai.pipeline.detect import detections_to_rows
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detect.py -v`
Expected: FAIL with `ImportError: cannot import name 'detections_to_rows'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/detect.py
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from footballai.config import Config
from footballai.pipeline.extract import iter_frames
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_detect.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/detect.py tests/test_detect.py
git commit -m "feat: YOLO detection with schema mapping"
```

---

### Task 6: Tracking with persistent IDs

**Files:**
- Create: `footballai/pipeline/track.py`
- Create: `tests/test_track.py`

**Interfaces:**
- Consumes: `Config`, `DETECTION_COLUMNS`, `TRACK_COLUMNS`.
- Produces:
  - `run_tracking(detections: pd.DataFrame, cfg: Config) -> pd.DataFrame` — feeds per-frame player/goalkeeper detections through `supervision.ByteTrack`, returns `TRACK_COLUMNS`. The **ball** and **referee** rows pass through with `track_id = -1` (not tracked). Player/goalkeeper rows get stable positive `track_id`s.

- [ ] **Step 1: Write the failing test** (synthetic detections → ID persistence; deterministic, no model)

```python
# tests/test_track.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_track.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pipeline.track'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/track.py
import numpy as np
import pandas as pd
import supervision as sv
from footballai.config import Config
from footballai import schema

_TRACKED = {"player", "goalkeeper"}

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
            ids = np.full(len(tracked), -1, dtype=int)
            # supervision returns detections in the same order it received them
            ids[: len(updated.tracker_id)] = updated.tracker_id
            tracked["track_id"] = ids
        out_frames.append(pd.concat([tracked, untracked], ignore_index=True))
    out = pd.concat(out_frames, ignore_index=True)
    out = out[schema.TRACK_COLUMNS]
    schema.validate(out, schema.TRACK_COLUMNS)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_track.py -v`
Expected: PASS

> Note: if `supervision`'s `update_with_detections` drops low-confidence rows so order/length diverge, match returned boxes back to input rows by nearest-IoU instead of positional assignment. Keep confidences ≥ `activation_threshold` in tests to avoid this.

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/track.py tests/test_track.py
git commit -m "feat: ByteTrack tracking with persistent player ids"
```

---

### Task 7: Team assignment (jersey clustering)

**Files:**
- Create: `footballai/pipeline/teams.py`
- Create: `tests/test_teams.py`

**Interfaces:**
- Consumes: `Config`, `TRACK_COLUMNS`, `TEAM_COLUMNS`, video frames (for crops).
- Produces:
  - `cluster_embeddings(embeddings: np.ndarray, cfg: Config) -> np.ndarray` — pure: UMAP→KMeans(2) returning a label array in `{0,1}`. (Unit-tested with synthetic separable embeddings.)
  - `assign_teams(tracks: pd.DataFrame, video_path: str, cfg: Config) -> pd.DataFrame` — crops each tracked player, embeds with SigLIP, clusters, maps cluster→`"A"/"B"`; goalkeepers/referees labeled by class, ball `team=None`. Returns `TEAM_COLUMNS`. (Smoke-tested when weights/model available.)

- [ ] **Step 1: Write the failing test** (pure clustering on separable synthetic data)

```python
# tests/test_teams.py
import numpy as np
from footballai.config import load_config
from footballai.pipeline.teams import cluster_embeddings

def test_cluster_separates_two_blobs():
    cfg = load_config()
    rng = np.random.default_rng(0)
    a = rng.normal(-5, 0.2, size=(20, 8))
    b = rng.normal(5, 0.2, size=(20, 8))
    emb = np.vstack([a, b])
    labels = cluster_embeddings(emb, cfg)
    assert set(labels) == {0, 1}
    # all of blob A share one label, all of blob B the other
    assert len(set(labels[:20])) == 1
    assert len(set(labels[20:])) == 1
    assert labels[0] != labels[20]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_teams.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pipeline.teams'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/teams.py
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from footballai.config import Config
from footballai import schema

def cluster_embeddings(embeddings: np.ndarray, cfg: Config) -> np.ndarray:
    n_comp = min(cfg.teams.get("umap_components", 3), embeddings.shape[1], len(embeddings) - 1)
    reducer = umap.UMAP(n_components=n_comp, random_state=cfg.seed)
    reduced = reducer.fit_transform(embeddings)
    km = KMeans(n_clusters=2, random_state=cfg.seed, n_init=10)
    return km.fit_predict(reduced)

def _embed_crops(crops: list[np.ndarray], cfg: Config) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    name = cfg.teams["siglip_model"]
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    imgs = [Image.fromarray(c[:, :, ::-1]) for c in crops]  # BGR->RGB
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt")
        feats = model.get_image_features(**inputs)
    return feats.cpu().numpy()

def assign_teams(tracks: pd.DataFrame, video_path: str, cfg: Config) -> pd.DataFrame:
    from footballai.pipeline.extract import iter_frames
    out = tracks.copy()
    out["team"] = None
    out.loc[out["class"] == "goalkeeper", "team"] = "GK"
    out.loc[out["class"] == "referee", "team"] = "referee"

    players = out[out["class"] == "player"]
    frames_needed = set(players["frame"])
    crops, index = [], []
    for fidx, frame in iter_frames(video_path):
        if fidx not in frames_needed:
            continue
        for ridx, r in players[players["frame"] == fidx].iterrows():
            x, y, w, h = int(r.bbox_x), int(r.bbox_y), int(r.bbox_w), int(r.bbox_h)
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            crops.append(crop)
            index.append(ridx)
    if crops:
        emb = _embed_crops(crops, cfg)
        labels = cluster_embeddings(emb, cfg)
        mapping = {0: "A", 1: "B"}
        for ridx, lab in zip(index, labels):
            out.at[ridx, "team"] = mapping[int(lab)]
    out = out[schema.TEAM_COLUMNS]
    schema.validate(out, schema.TEAM_COLUMNS)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_teams.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/teams.py tests/test_teams.py
git commit -m "feat: team assignment via SigLIP embeddings and clustering"
```

---

### Task 8: Calibration (homography)

**Files:**
- Create: `footballai/pipeline/calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `pitch.landmark_array`.
- Produces:
  - `compute_homography(image_pts: np.ndarray, pitch_pts: np.ndarray) -> np.ndarray` — `(3,3)` matrix mapping image pixels → pitch meters via `cv2.findHomography`.
  - `save_homography(H: np.ndarray, path: str) -> None` and `load_homography(path: str) -> np.ndarray` (JSON).
  - `click_landmarks(frame_path: str, landmark_names: list[str]) -> np.ndarray` — interactive OpenCV window collecting clicked pixel points (not unit-tested; manual tool).

- [ ] **Step 1: Write the failing test** (round-trip: a synthetic H, recover it, map points back)

```python
# tests/test_calibrate.py
import numpy as np
from footballai.pipeline.calibrate import compute_homography, save_homography, load_homography

def test_homography_round_trip(tmp_path):
    # known pitch points (meters) and a synthetic camera mapping
    pitch_pts = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=float)
    true_H = np.array([[2.0, 0.1, 50.0],
                       [0.05, 1.8, 30.0],
                       [0.0001, 0.0002, 1.0]])
    # project pitch->image with inverse so compute_homography(image->pitch) recovers mapping
    ones = np.ones((4, 1))
    pitch_h = np.hstack([pitch_pts, ones])
    img_h = (np.linalg.inv(true_H) @ pitch_h.T).T
    image_pts = img_h[:, :2] / img_h[:, 2:3]

    H = compute_homography(image_pts, pitch_pts)
    # mapping image_pts through H must land on pitch_pts
    rec = (H @ np.hstack([image_pts, ones]).T).T
    rec = rec[:, :2] / rec[:, 2:3]
    assert np.allclose(rec, pitch_pts, atol=1e-6)

def test_save_load_homography(tmp_path):
    H = np.eye(3)
    p = str(tmp_path / "h.json")
    save_homography(H, p)
    assert np.allclose(load_homography(p), H)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pipeline.calibrate'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/calibrate.py
import json
import cv2
import numpy as np

def compute_homography(image_pts: np.ndarray, pitch_pts: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(np.asarray(image_pts, float), np.asarray(pitch_pts, float))
    if H is None:
        raise ValueError("homography could not be computed from given points")
    return H

def save_homography(H: np.ndarray, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"H": np.asarray(H).tolist()}, f)

def load_homography(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f)["H"], dtype=float)

def click_landmarks(frame_path: str, landmark_names: list[str]) -> np.ndarray:
    img = cv2.imread(frame_path)
    pts: list[tuple[int, int]] = []
    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(landmark_names):
            pts.append((x, y))
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while len(pts) < len(landmark_names):
        disp = img.copy()
        cv2.putText(disp, f"click: {landmark_names[len(pts)]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(disp, p, 4, (0, 0, 255), -1)
        cv2.imshow("calibrate", disp)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    return np.array(pts, dtype=float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibrate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/calibrate.py tests/test_calibrate.py
git commit -m "feat: homography calibration with manual landmark clicking"
```

---

### Task 9: Coordinate projection (foot-point → meters)

**Files:**
- Create: `footballai/pipeline/project.py`
- Create: `tests/test_project.py`

**Interfaces:**
- Consumes: `TEAM_COLUMNS`, `PROJECTED_COLUMNS`, `load_homography`.
- Produces:
  - `foot_points(df: pd.DataFrame) -> np.ndarray` — `(N,2)` bottom-center pixel points (`bbox_x + bbox_w/2`, `bbox_y + bbox_h`).
  - `project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray` — applies homography, returns `(N,2)` meters.
  - `run_projection(teams: pd.DataFrame, H: np.ndarray) -> pd.DataFrame` — adds `img_x,img_y,pitch_x,pitch_y`, returns `PROJECTED_COLUMNS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_project.py
import numpy as np
import pandas as pd
from footballai.pipeline.project import foot_points, project_points, run_projection

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_project.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pipeline.project'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/project.py
import numpy as np
import pandas as pd
from footballai import schema

def foot_points(df: pd.DataFrame) -> np.ndarray:
    x = df["bbox_x"].to_numpy(float) + df["bbox_w"].to_numpy(float) / 2.0
    y = df["bbox_y"].to_numpy(float) + df["bbox_h"].to_numpy(float)
    return np.column_stack([x, y])

def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ homog.T).T
    return proj[:, :2] / proj[:, 2:3]

def run_projection(teams: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    out = teams.copy()
    fp = foot_points(out)
    out["img_x"], out["img_y"] = fp[:, 0], fp[:, 1]
    pitch = project_points(H, fp)
    out["pitch_x"], out["pitch_y"] = pitch[:, 0], pitch[:, 1]
    out = out[schema.PROJECTED_COLUMNS]
    schema.validate(out, schema.PROJECTED_COLUMNS)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_project.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/project.py tests/test_project.py
git commit -m "feat: foot-point projection to pitch meters"
```

---

### Task 10: Temporal refine (bidirectional gap-fill, ball state, smoothing)

**Files:**
- Create: `footballai/pipeline/refine.py`
- Create: `tests/test_refine.py`

**Interfaces:**
- Consumes: `Config`, `PROJECTED_COLUMNS`, `FINAL_COLUMNS`, `pitch.in_bounds`.
- Produces:
  - `interpolate_track(track: pd.DataFrame, max_gap: int) -> pd.DataFrame` — for a single `track_id`, fills missing frames between first and last observed frame by linear interpolation of `pitch_x/pitch_y` (and `img_x/img_y`); gaps longer than `max_gap` are left unfilled. Filled rows get `provenance="interpolated"`; original rows `provenance="observed"`.
  - `ball_state(df: pd.DataFrame) -> pd.DataFrame` — assigns ball `provenance`: `observed` where detected, `possessed` for interpolated frames whose nearest player is within possession radius, else `interpolated`.
  - `run_refine(projected: pd.DataFrame, fps: float, cfg: Config) -> pd.DataFrame` — adds `timestamp` and `provenance`, runs per-track interpolation + ball state, returns `FINAL_COLUMNS`.

- [ ] **Step 1: Write the failing test** (synthetic track with a one-frame gap → interpolated midpoint)

```python
# tests/test_refine.py
import pandas as pd
import numpy as np
from footballai.config import load_config
from footballai.pipeline.refine import interpolate_track, run_refine

def _track_with_gap():
    # track_id 1 present at frames 0 and 2, missing at 1
    base = dict(track_id=1, class="player", team="A", conf=0.9,
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
    out = interpolate_track(_track_with_gap(), max_gap=1).sort_values("frame")
    assert list(out["frame"]) == [0, 2]  # gap of 1 missing frame == within? boundary

def test_run_refine_adds_timestamp_and_provenance():
    cfg = load_config()
    out = run_refine(_track_with_gap(), fps=5.0, cfg=cfg)
    assert "timestamp" in out.columns and "provenance" in out.columns
    assert out[out["frame"] == 2].iloc[0]["timestamp"] == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_refine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pipeline.refine'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/refine.py
import numpy as np
import pandas as pd
from footballai.config import Config
from footballai import schema

_INTERP_COLS = ["pitch_x", "pitch_y", "img_x", "img_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]

def interpolate_track(track: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    track = track.sort_values("frame").copy()
    track["provenance"] = "observed"
    frames = track["frame"].to_numpy()
    rows = [track]
    for a, b in zip(frames[:-1], frames[1:]):
        gap = int(b - a) - 1
        if gap <= 0 or gap > max_gap:
            continue
        row_a = track[track["frame"] == a].iloc[0]
        row_b = track[track["frame"] == b].iloc[0]
        for k in range(1, gap + 1):
            t = k / (gap + 1)
            new = row_a.copy()
            new["frame"] = a + k
            for c in _INTERP_COLS:
                new[c] = (1 - t) * row_a[c] + t * row_b[c]
            new["conf"] = 0.0
            new["provenance"] = "interpolated"
            rows.append(pd.DataFrame([new]))
    out = pd.concat(rows, ignore_index=True).sort_values("frame").reset_index(drop=True)
    return out

def ball_state(df: pd.DataFrame, possession_radius: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    ball = df["class"] == "ball"
    players = df[df["class"] != "ball"]
    for idx in df[ball & (df["provenance"] == "interpolated")].index:
        row = df.loc[idx]
        same_frame = players[players["frame"] == row["frame"]]
        if len(same_frame):
            d = np.hypot(same_frame["pitch_x"] - row["pitch_x"],
                         same_frame["pitch_y"] - row["pitch_y"])
            if d.min() <= possession_radius:
                df.at[idx, "provenance"] = "possessed"
    return df

def run_refine(projected: pd.DataFrame, fps: float, cfg: Config) -> pd.DataFrame:
    max_gap = cfg.refine["max_gap"]
    parts = []
    # interpolate per real track; untracked rows (track_id=-1) pass through as observed
    tracked = projected[projected["track_id"] > 0]
    for tid, grp in tracked.groupby("track_id"):
        parts.append(interpolate_track(grp, max_gap))
    untracked = projected[projected["track_id"] <= 0].copy()
    untracked["provenance"] = "observed"
    parts.append(untracked)
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = out["frame"] / fps
    out = ball_state(out)
    out = out[schema.FINAL_COLUMNS].sort_values(["frame", "track_id"]).reset_index(drop=True)
    schema.validate(out, schema.FINAL_COLUMNS)
    return out
```

> Note on `test_gap_longer_than_max_is_not_filled`: a gap of one missing frame (frames 0→2) has `gap == 1`; with `max_gap=1` it IS filled. To make the test meaningful, change its assertion to expect `[0, 1, 2]`, OR set `max_gap=0`. Use `max_gap=0` in that test and assert `[0, 2]`. Fix the test accordingly before running Step 4.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_refine.py -v`
Expected: PASS (with the `max_gap=0` correction noted above)

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/refine.py tests/test_refine.py
git commit -m "feat: bidirectional gap-fill, ball possession state, provenance"
```

---

### Task 11: Export artifacts

**Files:**
- Create: `footballai/pipeline/export.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: `FINAL_COLUMNS`, `VideoMeta`.
- Produces:
  - `write_artifacts(df: pd.DataFrame, meta: dict, out_dir: str) -> None` — writes `tracks.parquet`, `tracks.json`, `meta.json` under `out_dir`.
  - `read_tracks(out_dir: str) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export.py
import pandas as pd
from footballai import schema
from footballai.pipeline.export import write_artifacts, read_tracks

def test_write_and_read_round_trip(tmp_path):
    df = pd.DataFrame({c: [0] for c in schema.FINAL_COLUMNS})
    write_artifacts(df, {"fps": 25.0}, str(tmp_path))
    assert (tmp_path / "tracks.parquet").exists()
    assert (tmp_path / "tracks.json").exists()
    assert (tmp_path / "meta.json").exists()
    back = read_tracks(str(tmp_path))
    assert list(back.columns) == schema.FINAL_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.pipeline.export'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/pipeline/export.py
import json
import os
import pandas as pd
from footballai import schema

def write_artifacts(df: pd.DataFrame, meta: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = df[schema.FINAL_COLUMNS]
    df.to_parquet(os.path.join(out_dir, "tracks.parquet"), index=False)
    df.to_json(os.path.join(out_dir, "tracks.json"), orient="records")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def read_tracks(out_dir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(out_dir, "tracks.parquet"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/pipeline/export.py tests/test_export.py
git commit -m "feat: export tracks.parquet, tracks.json, meta.json"
```

---

### Task 12: Orchestrator with partial runs

**Files:**
- Create: `footballai/run.py`
- Create: `tests/test_run.py`

**Interfaces:**
- Consumes: every stage above, `load_config`, `read_meta`.
- Produces:
  - `STAGES = ["detect", "track", "teams", "project", "refine", "export"]`
  - `run_pipeline(video_path, calib_path, cfg, work_dir, start=None, end=None) -> None` — runs the stage range, reading/writing intermediate parquet (`detect.parquet`, `track.parquet`, …) so a partial run reuses prior artifacts.
  - CLI: `python -m footballai.run --video V --calib H.json --config config.yaml [--from track --to export]`.

- [ ] **Step 1: Write the failing test** (stage-range resolution is pure logic)

```python
# tests/test_run.py
from footballai.run import resolve_stage_range, STAGES

def test_full_range_default():
    assert resolve_stage_range(None, None) == STAGES

def test_subrange_inclusive():
    assert resolve_stage_range("track", "project") == ["track", "teams", "project"]

def test_invalid_stage_raises():
    import pytest
    with pytest.raises(ValueError):
        resolve_stage_range("nope", None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footballai.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# footballai/run.py
import argparse
import os
import pandas as pd
from footballai.config import load_config, Config
from footballai.pipeline.extract import read_meta
from footballai.pipeline import detect, track, teams, project, refine, export
from footballai.pipeline.calibrate import load_homography

STAGES = ["detect", "track", "teams", "project", "refine", "export"]

def resolve_stage_range(start: str | None, end: str | None) -> list[str]:
    s = STAGES.index(start) if start else 0
    e = STAGES.index(end) if end else len(STAGES) - 1
    if start and start not in STAGES:
        raise ValueError(start)
    if end and end not in STAGES:
        raise ValueError(end)
    return STAGES[s : e + 1]

def _p(work_dir: str, name: str) -> str:
    return os.path.join(work_dir, f"{name}.parquet")

def run_pipeline(video_path, calib_path, cfg: Config, work_dir, start=None, end=None) -> None:
    os.makedirs(work_dir, exist_ok=True)
    stages = resolve_stage_range(start, end)
    meta = read_meta(video_path)
    H = load_homography(calib_path)

    if "detect" in stages:
        detect.run_detection(video_path, cfg).to_parquet(_p(work_dir, "detect"))
    if "track" in stages:
        df = pd.read_parquet(_p(work_dir, "detect"))
        track.run_tracking(df, cfg).to_parquet(_p(work_dir, "track"))
    if "teams" in stages:
        df = pd.read_parquet(_p(work_dir, "track"))
        teams.assign_teams(df, video_path, cfg).to_parquet(_p(work_dir, "teams"))
    if "project" in stages:
        df = pd.read_parquet(_p(work_dir, "teams"))
        project.run_projection(df, H).to_parquet(_p(work_dir, "project"))
    if "refine" in stages:
        df = pd.read_parquet(_p(work_dir, "project"))
        refine.run_refine(df, meta.fps, cfg).to_parquet(_p(work_dir, "refine"))
    if "export" in stages:
        df = pd.read_parquet(_p(work_dir, "refine"))
        export.write_artifacts(df, {"fps": meta.fps, "width": meta.width,
                                    "height": meta.height, "n_frames": meta.n_frames},
                               work_dir)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--work", default="work")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", dest="end", default=None)
    a = ap.parse_args()
    run_pipeline(a.video, a.calib, load_config(a.config), a.work, a.start, a.end)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add footballai/run.py tests/test_run.py
git commit -m "feat: pipeline orchestrator with partial-run support"
```

---

### Task 13: End-to-end smoke test

**Files:**
- Create: `tests/test_e2e_smoke.py`

**Interfaces:**
- Consumes: `run_pipeline` stages `project → export` (skips model stages so it needs no weights), `compute_homography`, `make_clip`.

- [ ] **Step 1: Write the failing test** (drive the deterministic tail of the pipeline end-to-end)

```python
# tests/test_e2e_smoke.py
import numpy as np
import pandas as pd
from footballai.config import load_config
from footballai import schema
from footballai.pipeline.calibrate import compute_homography, save_homography
from footballai.pipeline.project import run_projection
from footballai.pipeline.refine import run_refine
from footballai.pipeline.export import write_artifacts, read_tracks

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
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run: `pytest tests/test_e2e_smoke.py -v`
Expected: PASS immediately if Tasks 9–11 are done (this is an integration test over existing code; if it fails, the failure points at a real wiring bug to fix).

- [ ] **Step 3: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke over project/refine/export"
```

---

## Manual verification (requires GPU + weights, not in CI)

After the suite is green, verify the model stages on a real clip:

1. Place soccer YOLO weights at `weights/football-players.pt` (from roboflow/sports) and a short match clip at `samples/clip.mp4`.
2. Extract one frame, run `calibrate.click_landmarks` to build `work/homography.json`.
3. `python -m footballai.run --video samples/clip.mp4 --calib work/homography.json --work work`
4. Open `work/tracks.parquet`; confirm: player rows have stable `track_id`s, two `team` values appear, `pitch_x/pitch_y` fall within 0–105 / 0–68, ball rows exist with sensible `provenance`.
5. Sanity assert in a REPL: `df[df["class"]=="player"][["pitch_x","pitch_y"]].describe()` — means should sit roughly mid-pitch, no wild outliers (outliers ⇒ recheck calibration clicks).

---

## Self-Review

**Spec coverage:**
- Detection (players/GK/ref/ball) → Task 5 ✓
- Persistent IDs → Task 6 ✓
- Team/role labels → Task 7 ✓
- Pitch meters via homography → Tasks 8–9 ✓
- Temporal memory / bidirectional gap-fill / ball state / provenance → Task 10 ✓
- Long/tidy schema export (parquet+json+meta) → Tasks 3, 11 ✓
- Partial-run orchestration → Task 12 ✓
- Testing strategy (geometry, homography round-trip, calibration-bounds, e2e smoke) → Tasks 2, 8, 13 ✓
- **Deferred (by design, separate plan):** FastAPI + SvelteKit dashboard; BoT-SORT Re-ID upgrade; PnLCalib auto-calibration. These are upgrades/Plan-2 items, intentionally out of this plan.
- **Partial gap:** the spec's appearance-memory Re-ID (relinking the *same* ID after long occlusion) is approximated here by ByteTrack's `lost_track_buffer` + linear interpolation; true appearance Re-ID arrives with the BoT-SORT upgrade. Calibration-bounds test is folded into the manual-verification step rather than a unit test (it needs real projected data); acceptable since `pitch.in_bounds` itself is unit-tested in Task 2.

**Placeholder scan:** no TBD/TODO; every code step is complete. Two tests carry explicit correction notes (Task 6 ordering caveat, Task 10 `max_gap=0` fix) — these are deliberate guardrails, not placeholders.

**Type consistency:** column-list names (`DETECTION_COLUMNS` … `FINAL_COLUMNS`) are defined once in Task 3 and referenced unchanged throughout. `run_*` stage functions consistently take/return DataFrames; `H` is always a `(3,3)` np.ndarray; `track_id` convention (`>0` tracked, `-1` untracked) is consistent across Tasks 6, 9, 10, 12.
