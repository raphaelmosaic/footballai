# Soccer Video Analysis — Design Spec

**Date:** 2026-06-20
**Status:** Approved design, pre-implementation

## 1. Purpose

Build an offline tool that ingests a soccer (Fußball) match video from a **single fixed/wide
camera** and produces, as accurately as possible, the positions of **all players and the ball**
on the pitch — in real pitch coordinates (meters) — even when the feed is partially obstructed or
skewed.

The **primary deliverable is the coordinate data**, which feeds a downstream prediction model
trained on player and ball positions. A web dashboard is a secondary surface, used mainly to
**visualize and QA** the extracted coordinates.

### Per-frame outputs required
- Persistent **player IDs** (stable across frames and occlusions)
- **Team / role labels** (team A/B, goalkeeper, referee)
- **Pitch coordinates in meters** (via homography; the "accurate even if skewed" requirement)
- **Ball position + occlusion state**

### Constraints / environment
- Input: single fixed/wide camera (stable view — no per-frame re-calibration needed)
- Compute: NVIDIA GPU, offline batch processing
- Dashboard: SvelteKit web app served by a thin FastAPI layer (Streamlit dropped)

## 2. Approach

**Approach A — stand on the existing soccer-CV ecosystem**, orchestrating proven open components
rather than building detection/tracking from scratch. The fixed camera is the key simplifier: the
hardest part of soccer analytics — homography (pixels → pitch meters) — is calibrated **once** and
reused, instead of re-estimated every frame as on broadcast feeds.

Modules are structured so each proven component can be swapped or upgraded without touching the
rest (e.g. ByteTrack → BoT-SORT, manual calibration → PnLCalib auto-calibration).

### Concrete technologies (from research)
- **Detection:** Ultralytics YOLOv8, fine-tuned soccer weights (players / goalkeeper / referee /
  ball), e.g. from `roboflow/sports` (trained on DFL Bundesliga Data Shootout dataset).
- **Tracking:** ByteTrack via the `supervision` library (MIT). Upgradeable to **BoT-SORT** for
  appearance Re-ID (fewer ID switches under occlusion).
- **Team classification:** SigLIP crop embeddings → UMAP dim-reduction → KMeans into two teams
  (from the `roboflow/sports` soccer example; more robust than raw HSV histograms).
- **Calibration:** OpenCV `findHomography` from manually clicked landmarks (default), with
  **PnLCalib** / No-Bells-Just-Whistles pretrained keypoint models as a drop-in auto-calibration
  upgrade.
- **Offline refinement:** bidirectional smoothing / gap-fill, informed by Global Tracklet
  Association (GTA / GTATrack, SoccerTrack 2025 winner) — trajectory-level association.
- **Dashboard reference:** the `roboflow/sports` RADAR mode (2D bird's-eye view) is a working
  reference for the minimap.

### Sources
- roboflow/sports soccer example: https://github.com/roboflow/sports/blob/main/examples/soccer/README.md
- How to Track Football Players notebook: https://github.com/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-football-players.ipynb
- Camera calibration in sports: https://blog.roboflow.com/camera-calibration-sports-computer-vision/
- PnLCalib: https://github.com/mguti97/PnLCalib · No-Bells-Just-Whistles: https://github.com/mguti97/No-Bells-Just-Whistles
- Deep-EIoU: https://arxiv.org/pdf/2306.13074 · GTATrack: https://arxiv.org/abs/2602.00484

## 3. Architecture — offline batch pipeline

Stages communicate through **on-disk artifacts** (one Parquet/JSON per stage), not in-memory
objects. This allows checkpointing, inspecting intermediate results, and re-running a single stage
cheaply while tuning (the practical payoff: `run.py --from track --to teams`).

```
video.mp4
   │
   ▼
[1] Frame extraction ──────────► frames + metadata (fps, resolution); GPU-decoded if available
   │
   ▼
[2] Detection (YOLOv8) ────────► per-frame boxes: {class: player|goalkeeper|referee|ball, bbox, conf}
   │
   ▼
[3] Tracking + state memory ───► persistent track_id per player (ByteTrack/BoT-SORT + Kalman)
   │
   ▼
[4] Team assignment ───────────► team label (A/B/GK/referee) via SigLIP+UMAP+KMeans
   │
   ▼
[5] Calibration (one-time) ────► homography matrix H (pixels → pitch meters)
   │
   ▼
[6] Coordinate projection ─────► foot-point × H → pitch (x,y) meters
   │
   ▼
[7] Bidirectional refine ──────► forward+backward smoothing, gap interpolation, ball-state inference
   │
   ▼
[8] Export ────────────────────► tracks.parquet + tracks.json + meta.json  (THE DELIVERABLE)
   │
   ▼
[9] FastAPI + SvelteKit ───────► video overlay + 2D pitch minimap + table + export
```

## 4. Temporal memory (track state across frames)

A first-class concern, not buried in smoothing. The system maintains a persistent **world state**
— a live estimate of every entity (each player, the ball) — that carries known information forward
when a frame's detection fails. A missing detection makes an entity **coast** on its predicted
trajectory rather than vanish.

**Mechanisms:**
1. **Motion model per track (Kalman filter):** position + velocity for every player and the ball.
   On a missing detection, predict from last velocity and keep emitting the entity, flagged
   `predicted` with decaying confidence.
2. **Appearance memory + Re-ID:** per-track memory bank (appearance embedding + team + last pitch
   position). On a new/unmatched detection, match against the bank (appearance + plausible distance
   given elapsed time) to **restore the original ID**. Backed by team label + pitch-distance gating.
   (BoT-SORT provides the appearance Re-ID.)
3. **Ball-specific memory:** trajectory buffer + possession heuristic — when the ball is lost, its
   likely location is constrained to the nearest player who last had it. Ball state labeled
   `observed | coasting | interpolated | possessed`.

**Offline advantage — bidirectional fill:** because this is batch (whole video available), memory
works in both directions. A forward pass builds tracks; a backward pass fills gaps using future
observations. A player vanishing at frame 100 and reappearing at 120 is interpolated from *both*
endpoints — far more accurate than forward-only prediction.

Every memory-derived coordinate carries a **provenance flag** (`observed` / `predicted` /
`interpolated` / `possessed`) and a confidence value, so the prediction model can weight or ignore
inferred positions.

## 5. Modules & project structure

```
footballai/
├── config.yaml                 # paths, model weights, thresholds, pitch dimensions
├── pitch.py                    # SINGLE SOURCE OF TRUTH: pitch dims + named landmark coords (105×68m)
├── pipeline/
│   ├── extract.py              # [1] video → frames/metadata
│   ├── detect.py               # [2] YOLOv8 inference → detections.parquet
│   ├── track.py                # [3] ByteTrack/BoT-SORT + Kalman state → tracks_raw.parquet
│   ├── teams.py                # [4] SigLIP+UMAP+KMeans → adds team labels
│   ├── calibrate.py            # [5] interactive homography tool (+ PnLCalib upgrade) → homography.json
│   ├── project.py              # [6] foot-point × H → pitch (x,y) meters
│   ├── refine.py               # [7] bidirectional smoothing + gap-fill + ball state
│   └── export.py               # [8] final tracks.parquet + tracks.json + meta.json
├── run.py                      # CLI orchestrator; supports --from/--to partial runs
├── api/                        # FastAPI serving layer
│   └── main.py
├── web/                        # SvelteKit dashboard
└── tests/
    └── fixtures/               # short ~10s clip + expected artifacts
```

**Key choices:**
- `pitch.py` is the single source of truth for field geometry, used by both calibration and the
  minimap — one definition, no drift.
- `run.py` supports partial runs so tuning re-runs only the affected stage.
- `config.yaml` centralizes every threshold/weight path — tuning never means code edits.

## 6. Calibration (stage 5)

Camera is fixed → runs **once per video**.
- **Default (manual):** open one representative frame; click ~6 known pitch landmarks (corners,
  penalty-box corners, center circle). Solve via OpenCV `findHomography` → `homography.json`.
- **Upgrade (auto):** PnLCalib pretrained keypoint model detects landmarks automatically; same
  `homography.json` output, nothing downstream changes.
- `project.py` maps each track's **foot-point** (bottom-center of bbox — the ground-contact point
  where the planar homography is valid), not the box center, through `H` to pitch meters.

## 7. Output schema (the deliverable)

**Long / tidy format** — one row per entity per frame. Handles variable player counts, occlusions,
and the ball uniformly; trivially pivoted to wide later. Exported as Parquet (efficient) + JSON
(inspectable).

| field | type | notes |
|---|---|---|
| `frame` | int | frame index |
| `timestamp` | float | seconds (frame / fps) |
| `track_id` | int | persistent player ID; stable across occlusions via Re-ID |
| `class` | enum | `player` / `goalkeeper` / `referee` / `ball` |
| `team` | enum | `A` / `B` / `referee` / `null` |
| `bbox_x,y,w,h` | float | pixel-space bounding box |
| `img_x, img_y` | float | foot-point in pixels (bottom-center of bbox) |
| `pitch_x, pitch_y` | float | **pitch coordinates in meters** (0–105, 0–68) |
| `provenance` | enum | `observed` / `predicted` / `interpolated` / `possessed` |
| `confidence` | float | detection conf, decayed while coasting |

The ball is simply rows with `class=ball` (with `provenance` capturing occlusion state), so the
consumer reads one uniform table.

`meta.json` accompanies the export: fps, resolution, pitch dimensions, calibration info, model
versions — for reproducibility.

## 8. Dashboard (SvelteKit + FastAPI)

Pipeline stays pure Python producing artifacts; a thin **FastAPI** layer serves artifacts + video
to a **SvelteKit** frontend (read-only over static artifacts; no run-triggering).

**FastAPI endpoints:**
- `GET /videos` — list processed videos
- `GET /videos/{id}/meta` — meta.json
- `GET /videos/{id}/tracks?from=&to=` — tracks by **frame range** (windowed; a 90-min match is a
  lot of rows)
- `GET /videos/{id}/frame/{n}` — frame-level detail
- `GET /media/{id}.mp4` — video stream

**SvelteKit views:**
- Video player with live overlay (boxes, IDs, team colors)
- **2D pitch minimap (SVG)** driven by `pitch_x/pitch_y`, synced to the video timeline — the visual
  proof coordinates are correct (a dot leaving the pitch outline = calibration off). Doubles as a
  calibration-QA tool.
- Scrubber + play/pause; minimap and video share one clock
- Entity table (filter by team / id / provenance) + CSV/JSON export
- **Provenance rendered visually:** solid dots = `observed`, hollow = `predicted`/`interpolated` —
  exposes the temporal-memory behavior directly.

## 9. Testing strategy

- **Unit:** `pitch.py` geometry; homography round-trip (project a known landmark → assert it lands
  at its real meter coordinate within tolerance); schema validation on export.
- **Stage golden tests:** run each stage on a short (~10s) clip in `tests/fixtures/`; assert
  artifact shape/columns and that track IDs persist across a known occlusion.
- **Calibration sanity:** assert all projected player positions fall within pitch bounds (+ small
  margin) — the automatic version of the minimap's visual check.
- **End-to-end smoke:** `run.py` on the fixture clip produces a non-empty `tracks.parquet` with all
  expected columns.

## 10. Build order (high level)

1. `pitch.py` + `config.yaml` + repo skeleton
2. `extract.py` → `detect.py` (get boxes out of a clip)
3. `track.py` (persistent IDs) → `teams.py`
4. `calibrate.py` (manual) → `project.py` (pitch meters)
5. `refine.py` (temporal memory + bidirectional fill) → `export.py`
6. FastAPI `api/` over artifacts
7. SvelteKit `web/` dashboard (overlay + minimap + table)
8. Tests + fixtures throughout

Upgrades deferred until the baseline works: BoT-SORT Re-ID, PnLCalib auto-calibration.
