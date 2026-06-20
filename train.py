"""Pre-training loop for the football state representation model."""

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import ModelConfig
from data import SequenceDataset
from models import FootballStateModel
from utils.metrics import masked_bce_loss, masked_ce_loss, masked_mse_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FootballStateModel")
    parser.add_argument("--data_dir", type=str, default="/home/jack/workspace/open-data/data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--seq_stride", type=int, default=25)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--max_matches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--receiver_weight", type=float, default=1.0)
    parser.add_argument("--shot_weight", type=float, default=1.0)
    parser.add_argument("--turnover_weight", type=float, default=1.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ModelConfig:
    cfg = ModelConfig(
        data_root=args.data_dir,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        label_horizon_seconds=args.horizon,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def match_train_val_split(dataset: SequenceDataset, val_ratio: float, seed: int) -> tuple:
    """Split dataset by match_id to prevent leakage across sequences."""
    unique_matches = sorted({match_id for match_id, _, _ in dataset._index})
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_matches)
    split_idx = int(len(unique_matches) * (1 - val_ratio))
    train_matches = set(unique_matches[:split_idx])
    val_matches = set(unique_matches[split_idx:])

    train_indices = [
        i for i, (mid, _, _) in enumerate(dataset._index) if mid in train_matches
    ]
    val_indices = [
        i for i, (mid, _, _) in enumerate(dataset._index) if mid in val_matches
    ]
    return train_indices, val_indices


def padded_collate(batch: List[dict]) -> dict:
    """Collate variable-length sequences with padding and mask."""
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


def build_dataloaders(cfg: ModelConfig, num_workers: int, val_ratio: float, seed: int):
    full_dataset = SequenceDataset(
        data_root=cfg.data_root,
        seq_len=cfg.seq_len,
        stride=cfg.seq_stride,
        horizon_seconds=cfg.label_horizon_seconds,
        max_matches=cfg.max_matches,
    )
    train_indices, val_indices = match_train_val_split(full_dataset, val_ratio, seed)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=padded_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=padded_collate,
    )
    return train_loader, val_loader


class MultiTaskLoss(nn.Module):
    """Combine MSE/BCE/CE for the three pretraining objectives."""

    def __init__(
        self,
        receiver_weight: float,
        shot_weight: float,
        turnover_weight: float,
    ):
        super().__init__()
        self.receiver_weight = receiver_weight
        self.shot_weight = shot_weight
        self.turnover_weight = turnover_weight

    def forward(self, outputs: dict, batch: dict) -> Dict[str, torch.Tensor]:
        lengths = batch["lengths"]
        losses = {}

        # Pass receiver: end coordinates (MSE) and receiver slot (CE).
        pass_out = outputs["pass_receiver"]
        pass_mask = batch["pass_receiver_slot"] != -1  # [B, T]
        if pass_mask.any():
            losses["pass_xy"] = masked_mse_loss(
                pass_out["end_xy"], batch["pass_receiver_xy"], lengths, extra_mask=pass_mask
            )
            losses["pass_slot"] = masked_ce_loss(
                pass_out["receiver_logits"], batch["pass_receiver_slot"], lengths, extra_mask=pass_mask
            )

        # Shot / xG: shot probability (BCE) and xG value (MSE on shots only).
        shot_out = outputs["shot_xg"]
        shot_flag = batch["shot_xg"][..., 0]
        shot_value = batch["shot_xg"][..., 1]
        losses["shot_prob"] = masked_bce_loss(
            shot_out["shot_prob"].squeeze(-1), shot_flag, lengths
        )
        shot_value_mask = shot_flag > 0
        if shot_value_mask.any():
            losses["shot_xg"] = masked_mse_loss(
                shot_out["xg"].squeeze(-1), shot_value, lengths, extra_mask=shot_value_mask
            )

        # Turnover.
        losses["turnover"] = masked_bce_loss(
            outputs["turnover"]["turnover_prob"].squeeze(-1),
            batch["turnover"],
            lengths,
        )

        total = (
            self.receiver_weight * (losses.get("pass_xy", 0) + losses.get("pass_slot", 0))
            + self.shot_weight * (losses.get("shot_prob", 0) + losses.get("shot_xg", 0))
            + self.turnover_weight * losses["turnover"]
        )
        losses["total"] = total
        return losses


def evaluate(model: nn.Module, loader: DataLoader, criterion: MultiTaskLoss, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, List[float]] = {k: [] for k in ["total", "pass_xy", "pass_slot", "shot_prob", "shot_xg", "turnover"]}
    pass_acc, shot_mae, turnover_acc = [], [], []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ("frames",)}

            outputs = model(frames, seq_len=lengths)
            batch_on_device = {"lengths": lengths, **labels}
            losses = criterion(outputs, batch_on_device)
            for k in metrics:
                if k in losses:
                    metrics[k].append(losses[k].item())

            # Per-task quick metrics.
            pass_out = outputs["pass_receiver"]
            pass_mask = labels["pass_receiver_slot"] != -1
            if pass_mask.any():
                preds = pass_out["receiver_logits"].argmax(dim=-1)
                correct = ((preds == labels["pass_receiver_slot"]) & pass_mask).float().sum()
                pass_acc.append((correct / (pass_mask.sum() + 1e-8)).item())

            shot_flag = labels["shot_xg"][..., 0]
            shot_value = labels["shot_xg"][..., 1]
            shot_pred = outputs["shot_xg"]["shot_prob"].squeeze(-1)
            if shot_flag.any():
                shot_mae.append(
                    ((outputs["shot_xg"]["xg"].squeeze(-1) - shot_value).abs() * shot_flag).sum()
                    / (shot_flag.sum() + 1e-8)
                )
            turnover_pred = outputs["turnover"]["turnover_prob"].squeeze(-1)
            turnover_mask = _length_to_mask(lengths, turnover_pred.size(1))
            t_correct = (((turnover_pred > 0.5).long() == labels["turnover"].long()) & turnover_mask).float().sum()
            turnover_acc.append((t_correct / (turnover_mask.sum() + 1e-8)).item())

    result = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    if pass_acc:
        result["pass_acc"] = float(np.mean(pass_acc))
    if shot_mae:
        result["shot_mae"] = float(torch.stack(shot_mae).mean().item())
    if turnover_acc:
        result["turnover_acc"] = float(np.mean(turnover_acc))
    return result


def _length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def train(cfg: ModelConfig, args: argparse.Namespace) -> None:
    seed_everything(cfg.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = FootballStateModel(cfg).to(device)
    criterion = MultiTaskLoss(
        receiver_weight=args.receiver_weight,
        shot_weight=args.shot_weight,
        turnover_weight=args.turnover_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_loader, val_loader = build_dataloaders(cfg, args.num_workers, args.val_ratio, cfg.seed)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)
        print(f"Resumed from epoch {start_epoch}")

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses: Dict[str, List[float]] = {
            k: [] for k in ["total", "pass_xy", "pass_slot", "shot_prob", "shot_xg", "turnover"]
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            frames = batch["frames"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "frames"}

            optimizer.zero_grad()
            outputs = model(frames, seq_len=lengths)
            losses = criterion(outputs, {"frames": frames, **labels})
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k].append(losses[k].item())
            pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_losses.items() if v})

        train_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["total"])

        print(
            f"Epoch {epoch + 1}: train_total={train_metrics.get('total', 0):.4f} "
            f"val_total={val_metrics.get('total', 0):.4f} "
            f"val_pass_acc={val_metrics.get('pass_acc', 0):.4f} "
            f"val_shot_mae={val_metrics.get('shot_mae', 0):.4f} "
            f"val_turnover_acc={val_metrics.get('turnover_acc', 0):.4f}"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_metrics,
            checkpoint_dir / "latest.pt",
            cfg,
        )
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_metrics,
                checkpoint_dir / "best.pt",
                cfg,
            )
            print(f"Saved best checkpoint (val_total={best_val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)
    train(cfg, args)
