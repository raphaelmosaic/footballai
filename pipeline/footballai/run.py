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
