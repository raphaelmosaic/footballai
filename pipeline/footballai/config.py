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
