"""Multi-task pretraining heads sitting on top of the temporal backbone.

Input:  [B, T, H] per-timestep temporal hidden state and optional
        [B, T, N, D] per-entity spatial features.
Outputs:
    - pass_receiver: dict with "end_xy" [B, T, 2] and "receiver_logits" [B, T, slots]
    - shot_xg:       dict with "shot_prob" [B, T, 1], "xg" [B, T, 1]
    - turnover:      dict with "turnover_prob" [B, T, 1]
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class PassReceiverHead(nn.Module):
    """Predicts the destination of a pass: continuous (x, y) and discrete receiver slot.

    The receiver slot is predicted directly from the per-entity spatial features so
    that player identity/order is preserved. Mean pooling the entities before
    classification destroys the information needed to distinguish which of the 22
    outfield players will receive the ball.
    """

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

        # Per-entity logit for each outfield player. Entity 0 is the ball, so
        # player i corresponds to entity index i+1 and logit index i.
        self.player_logit = nn.Linear(config.embed_dim, 1)


    def forward(self, hidden: torch.Tensor, entity_features: torch.Tensor) -> dict:
        """
        Args:
            hidden:          [B, T, H] temporal hidden state.
            entity_features: [B, T, N, D] per-entity spatial features (ball is entity 0).
        Returns:
            dict with "end_xy" [B, T, 2] and "receiver_logits" [B, T, slots].
        """
        x = self.mlp(hidden)
        end_xy = self.end_xy_proj(x)  # unbounded real-valued pitch coordinates

        # Player features: exclude the ball (entity 0).
        players = entity_features[:, :, 1:, :]  # [B, T, slots, D]
        receiver_logits = self.player_logit(players).squeeze(-1)  # [B, T, slots]
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
            dict with "shot_logits" [B, T, 1] and "xg" [B, T, 1].
        """
        x = self.shared(hidden)
        shot_logits = self.shot_logit(x)  # logits for BCEWithLogitsLoss
        xg = self.xg_proj(x)
        return {"shot_logits": shot_logits, "xg": xg, "shot_prob": torch.sigmoid(shot_logits)}


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
            # Output logits; probability is computed on demand in forward.
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Args:
            hidden: [B, T, H]
        Returns:
            dict with "turnover_logits" [B, T, 1] and "turnover_prob" [B, T, 1].
        """
        logits = self.net(hidden)
        return {"turnover_logits": logits, "turnover_prob": torch.sigmoid(logits)}


class PretrainHeads(nn.Module):
    """Combined multi-task wrapper for all pretraining objectives."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pass_receiver = PassReceiverHead(config)
        self.shot_xg = ShotXgHead(config)
        self.turnover = TurnoverHead(config)

    def forward(
        self,
        temporal_hidden: torch.Tensor,
        entity_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            temporal_hidden: [B, T, H]
            entity_features: [B, T, N, D] per-entity spatial features (optional,
                             required by the pass receiver head).
        Returns:
            Nested dict keyed by task name.
        """
        return {
            "pass_receiver": self.pass_receiver(temporal_hidden, entity_features),
            "shot_xg": self.shot_xg(temporal_hidden),
            "turnover": self.turnover(temporal_hidden),
        }


FootballPretrainHeads = PretrainHeads  # public alias
