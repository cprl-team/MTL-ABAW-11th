from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FineTuneCfg:
    freeze_frac: float = 0.0
    warmup_epochs: int = 0
    backbone_lr_mult: float = 1.0
    llrd_groups: int = 0
    llrd_decay: float = 1.0
    peft: str = "none"
    ema: bool = False
    ema_decay: float = 0.999

    @classmethod
    def from_cfg(cls, d: dict) -> "FineTuneCfg":
        d = dict(d or {})
        ft = cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})
        if ft.peft == "lora":
            raise NotImplementedError(
                "LoRA fine-tuning is implemented for the ViT backbone (FaRL) path, "
                "not CNNs; use peft: bitfit|norm here, or switch backbone first.")
        if ft.peft not in ("none", "bitfit", "norm"):
            raise ValueError(f"unknown peft '{ft.peft}'")
        return ft


def _backbone_named_params(model):
    return list(model.backbone.named_parameters())


def _norm_param_ids(backbone) -> set:
    ids = set()
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
    for m in backbone.modules():
        if isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                ids.add(id(p))
    return ids


def freeze_all_backbone(model, flag: bool = True):
    for p in model.backbone.parameters():
        p.requires_grad_(not flag)


def apply_freeze(model, ft: FineTuneCfg):
    """Set backbone requires_grad per the configured strategy (post-warmup state)."""
    bb = _backbone_named_params(model)
    if ft.peft in ("bitfit", "norm"):
        for _, p in bb:
            p.requires_grad_(False)
        if ft.peft == "bitfit":
            for nm, p in bb:
                if nm.endswith("bias"):
                    p.requires_grad_(True)
        else:
            norm_ids = _norm_param_ids(model.backbone)
            for _, p in bb:
                if id(p) in norm_ids:
                    p.requires_grad_(True)
        return
    n = len(bb)
    cut = int(round(ft.freeze_frac * n))
    for i, (_, p) in enumerate(bb):
        p.requires_grad_(i >= cut)


def build_param_groups(model, base_lr: float, ft: FineTuneCfg):
    """Optimizer param groups: heads at base_lr; backbone at base_lr*mult, optionally
    layer-wise-decayed across llrd_groups (shallow layers get smaller LR)."""
    bb = _backbone_named_params(model)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": base_lr}]
    n = len(bb)
    if ft.llrd_groups and ft.llrd_decay < 1.0 and n > 0:
        k = ft.llrd_groups
        for gi in range(k):
            lo, hi = gi * n // k, (gi + 1) * n // k
            ps = [p for _, p in bb[lo:hi]]
            if not ps:
                continue
            depth_from_top = (k - 1 - gi)
            lr = base_lr * ft.backbone_lr_mult * (ft.llrd_decay ** depth_from_top)
            groups.append({"params": ps, "lr": lr})
    else:
        groups.append({"params": [p for _, p in bb], "lr": base_lr * ft.backbone_lr_mult})
    return groups


class ModelEMA:
    """Exponential moving average of weights; evaluate this to smooth the val curve."""
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow
