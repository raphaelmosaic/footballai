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
