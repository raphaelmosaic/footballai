"""Temporal backbone over a sequence of spatial state vectors.

Input:  [B, T, D] where D = config.state_dim (spatial encoder output).
Output: [B, T, H] where H = config.temporal_output_dim.
        If bidirectional, H = 2 * temporal_hidden_dim.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import ModelConfig


class TemporalModel(nn.Module):
    """Forward-only or bidirectional GRU/LSTM over state-vector sequences."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cell_type = config.temporal_cell.upper()
        self.bidirectional = config.temporal_bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        rnn_class = nn.GRU if self.cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_class(
            input_size=config.state_dim,
            hidden_size=config.temporal_hidden_dim,
            num_layers=config.num_temporal_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0.0,
        )

    def forward(
        self,
        state_sequence: torch.Tensor,
        seq_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state_sequence: [B, T, D]
            seq_len: optional [B] tensor of actual sequence lengths for packing.
        Returns:
            [B, T, H] per-timestep hidden states.
        """
        if seq_len is not None:
            # Pack padded sequences for efficient / correct bidirectional training.
            packed = nn.utils.rnn.pack_padded_sequence(
                state_sequence,
                seq_len.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            outputs, _ = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=state_sequence.size(1)
            )
            return outputs

        outputs, _ = self.rnn(state_sequence)
        return outputs  # [B, T, H]


FootballStateTemporalModel = TemporalModel  # public alias
