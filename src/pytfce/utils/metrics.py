"""Detection metrics for pTFCE variant comparison."""

from __future__ import annotations

import numpy as np


def detection_metrics(detected: np.ndarray, ground_truth: np.ndarray,
                      brain_mask: np.ndarray) -> dict:
    """Compute TPR, FPR, Dice, TP, FP, FN, TN within *brain_mask*."""
    det = detected.astype(bool) & brain_mask
    gt = ground_truth.astype(bool) & brain_mask
    bm = brain_mask.astype(bool)

    tp = int((det & gt).sum())
    fp = int((det & ~gt & bm).sum())
    fn = int((~det & gt).sum())
    tn = int((~det & ~gt & bm).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"tpr": tpr, "fpr": fpr, "dice": dice, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compare_methods(results: dict[str, np.ndarray],
                    ground_truth: np.ndarray,
                    brain_mask: np.ndarray) -> dict[str, dict]:
    """Compare multiple methods.  *results* maps method name to detection mask."""
    return {name: detection_metrics(det, ground_truth, brain_mask)
            for name, det in results.items()}


def soft_dice_torch(pred: "torch.Tensor", target: "torch.Tensor",
                    eps: float = 1e-7) -> "torch.Tensor":
    """Differentiable soft Dice for torch tensors (higher is better)."""
    import torch
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)
