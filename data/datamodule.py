"""PyTorch Lightning DataModule for StatsBomb sequence data."""

from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from config import ModelConfig
from data.statsbomb_dataset import SequenceDataset


def padded_collate(batch: list) -> dict:
    """Collate variable-length sequences with padding and mask."""
    import torch

    lengths = torch.tensor([item["lengths"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)
    n_entities = batch[0]["frames"].shape[1]
    feat_dim = batch[0]["frames"].shape[2]

    frames = torch.zeros(batch_size, max_len, n_entities, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, batch[0]["mask"].shape[1], dtype=torch.float32)
    pass_xy = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    pass_slot = torch.full((batch_size, max_len), -1, dtype=torch.long)
    shot_xg = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    turnover = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        seq_len = min(int(item["lengths"]), max_len)
        frames[i, :seq_len] = item["frames"][:seq_len]
        mask[i, :seq_len] = item["mask"][:seq_len]
        pass_xy[i, :seq_len] = item["pass_receiver_xy"][:seq_len]
        pass_slot[i, :seq_len] = item["pass_receiver_slot"][:seq_len]
        shot_xg[i, :seq_len] = item["shot_xg"][:seq_len]
        turnover[i, :seq_len] = item["turnover"][:seq_len]

    return {
        "frames": frames,
        "mask": mask,
        "lengths": lengths,
        "pass_receiver_xy": pass_xy,
        "pass_receiver_slot": pass_slot,
        "shot_xg": shot_xg,
        "turnover": turnover,
    }


class StatsBombDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping SequenceDataset.

    Splits sequences by match_id to prevent leakage.
    """

    def __init__(
        self,
        config: ModelConfig,
        val_ratio: float = 0.15,
        num_workers: int = 4,
    ):
        super().__init__()
        self.config = config
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self._full_dataset: Optional[SequenceDataset] = None
        self._train_indices: Optional[list] = None
        self._val_indices: Optional[list] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._full_dataset = SequenceDataset(
            data_root=self.config.data_root,
            seq_len=self.config.seq_len,
            stride=self.config.seq_stride,
            horizon_seconds=self.config.label_horizon_seconds,
            max_matches=self.config.max_matches,
        )
        unique_matches = sorted({match_id for match_id, _, _ in self._full_dataset._index})
        rng = np.random.default_rng(self.config.seed)
        rng.shuffle(unique_matches)
        split_idx = int(len(unique_matches) * (1 - self.val_ratio))
        train_matches = set(unique_matches[:split_idx])
        val_matches = set(unique_matches[split_idx:])

        self._train_indices = [
            i for i, (mid, _, _) in enumerate(self._full_dataset._index) if mid in train_matches
        ]
        self._val_indices = [
            i for i, (mid, _, _) in enumerate(self._full_dataset._index) if mid in val_matches
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self._full_dataset, self._train_indices),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=padded_collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self._full_dataset, self._val_indices),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=padded_collate,
        )
