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
    if img is None:
        raise FileNotFoundError(f"could not read frame image: {frame_path}")
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
    if len(pts) < len(landmark_names):
        raise RuntimeError(f"calibration aborted: collected {len(pts)}/{len(landmark_names)} points")
    return np.array(pts, dtype=float)
