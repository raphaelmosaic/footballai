import cv2, numpy as np

def make_clip(path: str, n: int = 10, w: int = 64, h: int = 48, fps: int = 5) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
