import numpy as np
from footballai.stages.calibrate import compute_homography, save_homography, load_homography

def test_homography_round_trip():
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
    assert np.allclose(rec, pitch_pts, atol=1e-4)

def test_save_load_homography(tmp_path):
    H = np.eye(3)
    p = str(tmp_path / "h.json")
    save_homography(H, p)
    assert np.allclose(load_homography(p), H)
