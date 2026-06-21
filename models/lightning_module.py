"""PyTorch Lightning module for football state representation pretraining."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ModelConfig
from models import FootballStateModel
from utils.metrics import masked_bce_loss, masked_ce_loss, masked_mse_loss


class FootballPretrainModule(pl.LightningModule):
    """LightningModule that wraps FootballStateModel and the multi-task loss.

    Inputs per batch:
        frames:             [B, T, N, F]
        lengths:            [B]
        mask:               [B, T, 22]
        pass_receiver_xy:   [B, T, 2]
        pass_receiver_slot: [B, T]
        shot_xg:            [B, T, 2]
        turnover:           [B, T]
    """

    def __init__(
        self,
        config: ModelConfig,
        receiver_weight: float = 1.0,
        shot_weight: float = 1.0,
        turnover_weight: float = 1.0,
        shot_pos_weight: float = 1.0,
        turnover_pos_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model = FootballStateModel(config)
        self.receiver_weight = receiver_weight
        self.shot_weight = shot_weight
        self.turnover_weight = turnover_weight
        self.shot_pos_weight = shot_pos_weight
        self.turnover_pos_weight = turnover_pos_weight

    def forward(self, frames: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> dict:
        return self.model(frames, seq_len=seq_len)

    def _build_pos_weight(self, ratio: float) -> torch.Tensor:
        """Heuristic positive weight from observed neg:pos ratio."""
        return torch.tensor(max(1.0, ratio), device=self.device)

    def _compute_losses(self, outputs: dict, batch: dict) -> Dict[str, torch.Tensor]:
        lengths = batch["lengths"]
        losses: Dict[str, torch.Tensor] = {}

        # Pass receiver: end coordinates (MSE) and receiver slot (CE).
        pass_out = outputs["pass_receiver"]
        pass_mask = batch["pass_receiver_slot"] != -1
        if pass_mask.any():
            losses["pass_xy"] = masked_mse_loss(
                pass_out["end_xy"], batch["pass_receiver_xy"], lengths, extra_mask=pass_mask
            )
            losses["pass_slot"] = masked_ce_loss(
                pass_out["receiver_logits"], batch["pass_receiver_slot"], lengths, extra_mask=pass_mask
            )

        # Shot / xG.
        shot_out = outputs["shot_xg"]
        shot_flag = batch["shot_xg"][..., 0]
        shot_value = batch["shot_xg"][..., 1]
        losses["shot_prob"] = masked_bce_loss(
            shot_out["shot_logits"].squeeze(-1),
            shot_flag,
            lengths,
            pos_weight=torch.tensor(self.shot_pos_weight, device=shot_out["shot_logits"].device),
        )
        shot_value_mask = shot_flag > 0
        if shot_value_mask.any():
            losses["shot_xg"] = masked_mse_loss(
                shot_out["xg"].squeeze(-1), shot_value, lengths, extra_mask=shot_value_mask
            )

        # Turnover.
        losses["turnover"] = masked_bce_loss(
            outputs["turnover"]["turnover_logits"].squeeze(-1),
            batch["turnover"],
            lengths,
            pos_weight=torch.tensor(self.turnover_pos_weight, device=outputs["turnover"]["turnover_logits"].device),
        )

        # Balance the three core losses so each task contributes comparable
        # gradient magnitudes. Empirical unweighted magnitudes at init:
        #   pass_slot ~ 3.09, shot_prob ~ 0.69, turnover ~ 0.69,
        # but with positive re-weighting shot/turnover become ~3-8. We scale
        # pass losses down and keep shot/turnover weighted by their ratios.
        total = (
            self.receiver_weight * (losses.get("pass_xy", 0) + losses.get("pass_slot", 0)) * 0.5
            + self.shot_weight * losses.get("shot_prob", 0)
            + self.turnover_weight * losses["turnover"]
        )
        # xG is an auxiliary dense regression only on positive shots; keep it in
        # metrics but outside the main total so it does not drown the core tasks.
        losses["total"] = total
        return losses

    def _compute_metrics(self, outputs: dict, batch: dict) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        pass_out = outputs["pass_receiver"]
        pass_mask = batch["pass_receiver_slot"] != -1
        if pass_mask.any():
            preds = pass_out["receiver_logits"].argmax(dim=-1)
            correct = ((preds == batch["pass_receiver_slot"]) & pass_mask).float().sum()
            metrics["pass_acc"] = (correct / (pass_mask.sum() + 1e-8)).item()

        shot_flag = batch["shot_xg"][..., 0]
        shot_value = batch["shot_xg"][..., 1]
        if shot_flag.any():
            mae = (
                (outputs["shot_xg"]["xg"].squeeze(-1) - shot_value).abs() * shot_flag
            ).sum() / (shot_flag.sum() + 1e-8)
            metrics["shot_mae"] = mae.item()

        turnover_pred = outputs["turnover"]["turnover_prob"].squeeze(-1)
        lengths = batch["lengths"]
        max_len = turnover_pred.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        t_correct = (((turnover_pred > 0.5).long() == batch["turnover"].long()) & mask).float().sum()
        metrics["turnover_acc"] = (t_correct / (mask.sum() + 1e-8)).item()
        return metrics

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch["frames"], seq_len=batch["lengths"])
        losses = self._compute_losses(outputs, batch)
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=key == "total", batch_size=batch["frames"].size(0))
        # Log GPU utilisation and batch timing for diagnostics.
        self.log("train/batch_size", float(batch["frames"].size(0)), on_step=True, on_epoch=False, prog_bar=False)
        return losses["total"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch["frames"], seq_len=batch["lengths"])
        losses = self._compute_losses(outputs, batch)
        metrics = self._compute_metrics(outputs, batch)
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=key == "total", batch_size=batch["frames"].size(0))
        for key, value in metrics.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, batch_size=batch["frames"].size(0))
        return losses["total"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3),
            "monitor": "val/total",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
