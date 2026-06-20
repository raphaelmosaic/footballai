"""Spatial encoder: Transformer over a variable number of players + ball.

Input layout per entity (config.raw_feature_dim == 10):
    [0] x            continuous coordinate, metres, normalized to [-1,1]
    [1] y            continuous coordinate, metres, normalized to [-1,1]
    [2] vx           continuous velocity, m/s, normalized similarly
    [3] vy           continuous velocity, m/s, normalized similarly
    [4] team0        one-hot team id (home)
    [5] team1        one-hot team id (away)
    [6] position_id  integer role id -> nn.Embedding (0 for ball)
    [7] possession   binary flag (1 = team in possession; 0 for ball)
    [8] ball         binary flag (1 = ball entity)
    [9] on_pitch     binary flag / padding mask (1 = valid)

Output:
    [B, D] fixed-size state vector, where D = config.embed_dim.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# Indices into the per-entity raw feature vector.
IDX_X, IDX_Y, IDX_VX, IDX_VY = 0, 1, 2, 3
IDX_TEAM0, IDX_TEAM1 = 4, 5
IDX_POSITION = 6
IDX_POSS = 7
IDX_BALL = 8
IDX_ON_PITCH = 9

# Slices used by the embedding layer.
CONTINUOUS_IDXS = [IDX_X, IDX_Y, IDX_VX, IDX_VY, IDX_POSS, IDX_BALL, IDX_ON_PITCH]


class EntityEmbedding(nn.Module):
    """Combines continuous projection + learned categorical embeddings for one entity."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Continuous fields: coordinates, velocity, possession/ball/on_pitch flags.
        self.continuous_proj = nn.Linear(len(CONTINUOUS_IDXS), config.embed_dim)

        # Learned categorical embeddings.
        self.position_embed = nn.Embedding(
            config.num_positions, config.embed_dim, padding_idx=0
        )
        self.entity_type_embed = nn.Embedding(2, config.embed_dim)  # player / ball
        self.team_proj = nn.Linear(config.num_teams, config.embed_dim)

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.transformer_dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, F] raw per-entity features.
        Returns:
            [B, N, D] entity representation.
        """
        x = self.continuous_proj(features[..., CONTINUOUS_IDXS])
        x = x + self.position_embed(features[..., IDX_POSITION].long().clamp(min=0))
        x = x + self.entity_type_embed(features[..., IDX_BALL].long().clamp(min=0, max=1))
        x = x + self.team_proj(features[..., IDX_TEAM0 : IDX_TEAM1 + 1])
        return self.dropout(self.norm(x))


class DistanceAwareTransformerBlock(nn.Module):
    """Single Transformer encoder block with optional pairwise distance bias."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.ff_dim, config.embed_dim),
        )

        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.dropout1 = nn.Dropout(config.transformer_dropout)
        self.dropout2 = nn.Dropout(config.transformer_dropout)
        self.dropout3 = nn.Dropout(config.transformer_dropout)

        self.use_distance_bias = config.use_distance_bias
        if self.use_distance_bias:
            self.distance_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, config.num_heads),
            )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:      [B, N, D]
            coords: [B, N, 2]  (x, y) used for distance bias
            mask:   [B, N] bool, True = valid entity
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        residual = x

        # Self-attention.
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, E]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        if self.use_distance_bias:
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
            dist = diff.norm(dim=-1, keepdim=True)             # [B, N, N, 1]
            bias = self.distance_mlp(dist).permute(0, 3, 1, 2)  # [B, H, N, N]
            scores = scores + bias

        if mask is not None:
            # mask shape [B, N] -> [B, 1, 1, N]
            padding_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~padding_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # deterministic guard for all-masked rows
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v)  # [B, H, N, E]
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        x = self.norm1(residual + self.dropout2(out))

        # Feed-forward.
        x = self.norm2(x + self.dropout3(self.ffn(x)))
        return x


class SpatialEncoder(nn.Module):
    """Transformer encoder that pools variable-length entities to a fixed state vector."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.entity_embed = EntityEmbedding(config)
        self.blocks = nn.ModuleList(
            [DistanceAwareTransformerBlock(config) for _ in range(config.num_transformer_layers)]
        )

        if config.pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        self.pool_type = config.pool_type
        self.output_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, F] raw entity features.
        Returns:
            [B, D] state vector.
        """
        B, N, _ = features.shape
        on_pitch = features[..., IDX_ON_PITCH].bool()  # [B, N]
        coords = features[..., :2]          # [B, N, 2]

        x = self.entity_embed(features)     # [B, N, D]

        if self.pool_type == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat([cls, x], dim=1)          # [B, N+1, D]
            coords = F.pad(coords, (0, 0, 1, 0), value=0.0)
            on_pitch = F.pad(on_pitch, (1, 0), value=True)

        for block in self.blocks:
            x = block(x, coords, on_pitch)

        x = self.output_norm(x)

        if self.pool_type == "cls":
            state = x[:, 0]  # [B, D]
        else:
            # Mean pool over valid entities; masked fill for safety.
            x_masked = x.masked_fill(~on_pitch.unsqueeze(-1), 0.0)
            counts = on_pitch.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            state = x_masked.sum(dim=1) / counts                     # [B, D]
        return state


FootballStateEncoder = SpatialEncoder  # public alias
