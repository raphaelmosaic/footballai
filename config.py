"""Hyperparameter configuration for the football state representation pipeline."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """Config object for model / training / task hyperparameters.

    Fields are grouped by:
      - pitch / coordinate metadata
      - raw entity feature layout
      - spatial Transformer encoder
      - temporal recurrent backbone
      - pretraining task horizons
      - optimizer / training
    """

    # ------------------------------------------------------------------
    # Pitch and coordinate metadata (StatsBomb uses 120 x 80 metres)
    # ------------------------------------------------------------------
    pitch_length: float = 120.0           # metres (x axis)
    pitch_width: float = 80.0             # metres (y axis)
    coord_dim: int = 2                    # (x, y)

    # ------------------------------------------------------------------
    # Entity feature layout
    # Per-entity feature vector is [x, y, vx, vy, team0, team1,
    #                                position_id, is_possession, ball, on_pitch]
    # ------------------------------------------------------------------
    num_teams: int = 2                    # home / away one-hot
    num_positions: int = 26               # discrete role id vocabulary (0-25 observed in StatsBomb)
    max_entities: int = 23                # 22 players + 1 ball
    raw_feature_dim: int = 10             # (x,y,vx,vy) + 2 team + 1 pos + 3 flags

    # ------------------------------------------------------------------
    # Spatial Transformer encoder
    # ------------------------------------------------------------------
    embed_dim: int = 128
    num_transformer_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 256
    transformer_dropout: float = 0.1
    use_distance_bias: bool = True
    pool_type: str = "mean"               # mean / max / cls

    # ------------------------------------------------------------------
    # Temporal recurrent backbone
    # ------------------------------------------------------------------
    temporal_hidden_dim: int = 128
    num_temporal_layers: int = 2
    temporal_bidirectional: bool = True
    temporal_dropout: float = 0.1
    temporal_cell: str = "GRU"            # GRU or LSTM

    # ------------------------------------------------------------------
    # Pretraining task horizons
    # ------------------------------------------------------------------
    label_horizon_seconds: float = 5.0
    pass_receiver_slots: int = 22         # one per outfield player slot

    # ------------------------------------------------------------------
    # Sequence sampling
    # ------------------------------------------------------------------
    seq_len: int = 50                     # events per training sequence
    seq_stride: int = 25                  # overlap between sequences
    data_root: str = "/home/jack/workspace/open-data/data"
    max_matches: Optional[int] = None     # debug cap on number of matches

    # ------------------------------------------------------------------
    # Optimizer / training
    # ------------------------------------------------------------------
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    grad_clip_norm: float = 1.0
    seed: int = 42

    # ------------------------------------------------------------------
    # Derived read-only properties
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        """Dimensionality of the fixed-size vector returned by the spatial encoder."""
        return self.embed_dim

    @property
    def temporal_output_dim(self) -> int:
        """Dimensionality of per-timestep hidden states from the temporal model."""
        return self.temporal_hidden_dim * (2 if self.temporal_bidirectional else 1)
