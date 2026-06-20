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
