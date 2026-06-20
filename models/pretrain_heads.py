"""Multi-task pretraining heads sitting on top of the temporal backbone.

Input:  [B, T, H] per-timestep temporal hidden state.
Outputs:
    - pass_receiver: dict with "end_xy" [B, T, 2] and "receiver_logits" [B, T, slots]
    - shot_xg:       dict with "shot_prob" [B, T, 1], "xg" [B, T, 1]
    - turnover:      dict with "turnover_prob" [B, T, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class PassReceiverHead(nn.Module):
    """Predicts the destination of a pass: continuous (x, y) and discrete receiver slot."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        in_dim = config.temporal_output_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(in_dim // 2, in_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.end_xy_proj = nn.Linear(in_dim // 2, 2)
        self.receiver_logits = nn.Linear(in_dim // 2, config.pass_receiver_slots)

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Args:
            hidden: [B, T, H]
        Returns:
            dict with "end_xy" [B, T, 2] and "receiver_logits" [B, T, slots].
        """
        x = self.mlp(hidden)
        end_xy = self.end_xy_proj(x)  # unbounded real-valued pitch coordinates
        receiver_logits = self.receiver_logits(x)
        return {"end_xy": end_xy, "receiver_logits": receiver_logits}


class ShotXgHead(nn.Module):
    """Predicts probability of a shot occurring and the expected xG value."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        in_dim = config.temporal_output_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.transformer_dropout),
        )

        self.shot_logit = nn.Linear(in_dim // 2, 1)
        self.xg_proj = nn.Sequential(
            nn.Linear(in_dim // 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Args:
            hidden: [B, T, H]
        Returns:
            dict with "shot_prob" [B, T, 1] and "xg" [B, T, 1].
        """
        x = self.shared(hidden)
        shot_prob = torch.sigmoid(self.shot_logit(x))
        xg = self.xg_proj(x)
        return {"shot_prob": shot_prob, "xg": xg}


class TurnoverHead(nn.Module):
    """Binary probability of an interception/recovery turnover in next k seconds."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        in_dim = config.temporal_output_dim
        self.k = config.label_horizon_seconds

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Args:
            hidden: [B, T, H]
        Returns:
            dict with "turnover_prob" [B, T, 1].
        """
        return {"turnover_prob": self.net(hidden)}


class PretrainHeads(nn.Module):
    """Combined multi-task wrapper for all pretraining objectives."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pass_receiver = PassReceiverHead(config)
        self.shot_xg = ShotXgHead(config)
        self.turnover = TurnoverHead(config)

    def forward(self, temporal_hidden: torch.Tensor) -> dict:
        """
        Args:
            temporal_hidden: [B, T, H]
        Returns:
            Nested dict keyed by task name.
        """
        return {
            "pass_receiver": self.pass_receiver(temporal_hidden),
            "shot_xg": self.shot_xg(temporal_hidden),
            "turnover": self.turnover(temporal_hidden),
        }


FootballPretrainHeads = PretrainHeads  # public alias
