"""Masked metric helpers for variable-length football sequences."""

import torch
import torch.nn.functional as F


def _length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return boolean mask [B, T] where True indicates valid timesteps."""
    batch_size = lengths.size(0)
    device = lengths.device
    seq_range = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    return seq_range < lengths.unsqueeze(1)


def _combine_masks(lengths: torch.Tensor, extra_mask: torch.Tensor = None) -> torch.Tensor:
    """Combine length mask with an optional extra boolean mask.

    Args:
        lengths: [B]
        extra_mask: [B, T] or [B, T, *] bool mask (e.g. positive-class mask).

    Returns:
        [B, T] boolean mask.
    """
    max_len = extra_mask.size(1) if extra_mask is not None else lengths.max().item()
    mask = _length_to_mask(lengths, int(max_len))
    if extra_mask is not None:
        # Collapse any trailing dims.
        if extra_mask.dim() > 2:
            extra_mask = extra_mask.any(dim=list(range(2, extra_mask.dim())))
        mask = mask & extra_mask
    return mask


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    extra_mask: torch.Tensor = None,
) -> torch.Tensor:
    """MSE averaged over valid timesteps only.

    Args:
        predictions: [B, T, D] or [B, T].
        targets: same shape as predictions for valid positions.
        lengths: [B] number of valid timesteps per sequence.
        extra_mask: optional [B, T] bool mask to further restrict samples.

    Returns:
        Scalar loss tensor.
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(-1)
        targets = targets.unsqueeze(-1)

    mask = _combine_masks(lengths, extra_mask).unsqueeze(-1)
    squared_errors = F.mse_loss(predictions, targets, reduction="none")
    masked_errors = squared_errors * mask.float()
    loss = masked_errors.sum() / (mask.sum() + 1e-8)
    return loss


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    extra_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Binary cross entropy averaged over valid timesteps."""
    if logits.dim() == 2:
        logits = logits.unsqueeze(-1)
        targets = targets.unsqueeze(-1)

    mask = _combine_masks(lengths, extra_mask).unsqueeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    masked_bce = bce * mask.float()
    loss = masked_bce.sum() / (mask.sum() + 1e-8)
    return loss


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    extra_mask: torch.Tensor = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Cross-entropy for multi-class predictions averaged over valid timesteps.

    Args:
        logits: [B, T, C]
        targets: [B, T] long class labels (ignore_index ignored automatically).
        lengths: [B]
        extra_mask: optional [B, T] bool mask of additional valid positions.

    Returns:
        Scalar loss tensor.
    """
    B, T, C = logits.shape
    flat_logits = logits.reshape(B * T, C)
    flat_targets = targets.reshape(B * T)
    ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="none")
    ce = ce.view(B, T)

    mask = _combine_masks(lengths, extra_mask)
    masked_ce = ce * mask.float()
    loss = masked_ce.sum() / (mask.sum() + 1e-8)
    return loss


def masked_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> float:
    """Classification accuracy over valid timesteps for binary logits."""
    max_len = logits.size(1)
    mask = _length_to_mask(lengths, max_len)
    preds = (torch.sigmoid(logits) > 0.5).long()
    correct = ((preds == targets.long()) & mask).float().sum()
    total = mask.sum().float()
    return (correct / (total + 1e-8)).item()


def masked_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> float:
    """Mean absolute error over valid timesteps."""
    max_len = predictions.size(1)
    mask = _length_to_mask(lengths, max_len)
    mae = (predictions - targets).abs()
    if mae.dim() == 3:
        mae = mae.mean(dim=-1)
    masked_mae = mae * mask.float()
    return (masked_mae.sum() / (mask.sum() + 1e-8)).item()
