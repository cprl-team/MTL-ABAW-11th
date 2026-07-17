"""Masked multi-task loss + homoscedastic uncertainty weighting."""
from src.losses.mtl import MultiTaskLoss, ccc_loss_1d

__all__ = ["MultiTaskLoss", "ccc_loss_1d"]
