"""Football state representation model package.

Public API:
    - FootballStateEncoder   : Transformer-based spatial encoder.
    - FootballStateTemporalModel : GRU / LSTM temporal backbone.
    - FootballPretrainHeads  : Multi-task pretraining objectives.
    - FootballStateModel     : End-to-end wrapper.
"""

import torch
import torch.nn as nn

from config import ModelConfig
from models.spatial_encoder import FootballStateEncoder
from models.temporal_model import FootballStateTemporalModel
from models.pretrain_heads import FootballPretrainHeads


class FootballStateModel(nn.Module):
    """End-to-end model: spatial encoder -> temporal model -> multi-task heads.

    Input:
        entity_features: [B, T, N, F] where N is variable number of entities
                         (players + ball) and F = config.raw_feature_dim.
        seq_len: optional [B] actual sequence lengths for packing.

    Output:
        Nested dict of task predictions plus intermediate "state" and "temporal_hidden".
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.spatial_encoder = FootballStateEncoder(config)
        self.temporal_model = FootballStateTemporalModel(config)
        self.pretrain_heads = FootballPretrainHeads(config)

    def forward(
        self,
        entity_features: torch.Tensor,
        seq_len: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            entity_features: [B, T, N, F]
            seq_len:         [B]
        Returns:
            {
                "state":           [B, T, D]  spatial state vectors,
                "temporal_hidden": [B, T, H]  temporal hidden states,
                "pass_receiver":   {...},
                "shot_xg":         {...},
                "turnover":        {...},
            }
        """
        B, T, N, F = entity_features.shape
        flat = entity_features.view(B * T, N, F)
        state = self.spatial_encoder(flat)         # [B*T, D]
        state = state.view(B, T, -1)               # [B, T, D]

        temporal_hidden = self.temporal_model(state, seq_len=seq_len)  # [B, T, H]
        predictions = self.pretrain_heads(temporal_hidden)

        return {
            "state": state,
            "temporal_hidden": temporal_hidden,
            **predictions,
        }


__all__ = [
    "FootballStateEncoder",
    "FootballStateTemporalModel",
    "FootballPretrainHeads",
    "FootballStateModel",
]
