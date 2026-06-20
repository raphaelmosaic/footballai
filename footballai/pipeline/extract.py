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
