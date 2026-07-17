from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ccc_loss_1d(pred, target):
    """1 - CCC over a 1-D batch (already masked to valid entries). Returns a scalar
    tensor; 0 if fewer than 2 valid points (no gradient signal there)."""
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pm, tm = pred.mean(), target.mean()
    pv, tv = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - pm) * (target - tm)).mean()
    ccc = 2 * cov / (pv + tv + (pm - tm) ** 2 + 1e-8)
    return 1.0 - ccc


def focal_ce(logits, target, weight=None, gamma=2.0):
    """Multi-class focal cross-entropy. weight is the per-class alpha (imbalance)."""
    logp = F.log_softmax(logits, dim=1)
    logpt = logp.gather(1, target[:, None]).squeeze(1)
    pt = logpt.exp()
    loss = -((1.0 - pt) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def ldam_ce(logits, target, margins, weight=None, s=30.0):
    """Label-Distribution-Aware Margin loss (Cao et al. 2019): subtract a per-class
    margin (larger for rarer classes) from the true-class logit, then scaled CE.
    Enlarges decision boundaries for tail classes — a margin-based alternative to
    focal/logit-adjustment for the long-tailed EXPR task. margins: (C,) tensor."""
    batch_m = margins.to(logits.dtype)[target].unsqueeze(1)
    true_logit = logits.gather(1, target[:, None])
    logits_m = logits.scatter(1, target[:, None], true_logit - batch_m)
    return F.cross_entropy(s * logits_m, target, weight=weight)


def asl_loss(logits, targets, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
    """Asymmetric Loss for multi-label classification (Ben-Baruch et al. 2020):
    focuses harder on positives than negatives and clips easy negatives — well suited
    to the AU task's heavy per-unit imbalance (rare AU15/23/24). targets in {0,1}."""
    p = torch.sigmoid(logits)
    xs_pos = p
    xs_neg = 1.0 - p
    if clip and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)
    los_pos = targets * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg
    pt = xs_pos * targets + xs_neg * (1.0 - targets)
    gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    loss = loss * (1.0 - pt) ** gamma
    return -loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, expr_class_weights=None, au_pos_weight=None,
                 uncertainty_weighting=True, expr_loss="ce", focal_gamma=2.0,
                 expr_prior=None, logit_adjust_tau=0.0, au_loss="bce",
                 ldam_max_m=0.5, ldam_s=30.0, asl_gamma_neg=4.0, asl_gamma_pos=0.0,
                 asl_clip=0.05):
        super().__init__()
        self.register_buffer(
            "expr_w", torch.as_tensor(expr_class_weights, dtype=torch.float32)
            if expr_class_weights is not None else None)
        self.register_buffer(
            "au_pos_w", torch.as_tensor(au_pos_weight, dtype=torch.float32)
            if au_pos_weight is not None else None)
        self.register_buffer(
            "expr_log_prior",
            torch.log(torch.clamp(torch.as_tensor(expr_prior, dtype=torch.float32), 1e-8))
            if expr_prior is not None else None)
        if expr_prior is not None:
            inv = torch.clamp(torch.as_tensor(expr_prior, dtype=torch.float32), 1e-8) ** -0.25
            self.register_buffer("ldam_margins", ldam_max_m * inv / inv.max())
        else:
            self.ldam_margins = None
        self.ldam_s = ldam_s
        self.logit_adjust_tau = logit_adjust_tau
        self.uncertainty_weighting = uncertainty_weighting
        if expr_loss not in ("ce", "focal", "ldam"):
            raise ValueError(f"expr_loss must be ce|focal|ldam, got '{expr_loss}'")
        if au_loss not in ("bce", "asl"):
            raise ValueError(f"au_loss must be bce|asl, got '{au_loss}'")
        self.expr_loss = expr_loss
        self.au_loss = au_loss
        self.focal_gamma = focal_gamma
        self.asl_params = (asl_gamma_neg, asl_gamma_pos, asl_clip)
        if uncertainty_weighting:
            self.log_var = nn.Parameter(torch.zeros(3))

    def _va(self, pred_va, valence, arousal, mask):
        if mask.sum() < 2:
            return pred_va.sum() * 0.0
        v = ccc_loss_1d(pred_va[mask, 0], valence[mask])
        a = ccc_loss_1d(pred_va[mask, 1], arousal[mask])
        return 0.5 * (v + a)

    def _expr(self, logits, target, mask):
        if mask.sum() == 0:
            return logits.sum() * 0.0
        if self.logit_adjust_tau and self.expr_log_prior is not None:
            logits = logits + self.logit_adjust_tau * self.expr_log_prior
        w = self.expr_w if self.expr_w is not None else None
        lg, tg = logits[mask], target[mask]
        if self.expr_loss == "ldam":
            return ldam_ce(lg, tg, self.ldam_margins, weight=w, s=self.ldam_s)
        elif self.expr_loss == "focal":
            return focal_ce(lg, tg, weight=w, gamma=self.focal_gamma)
        return F.cross_entropy(lg, tg, weight=w)

    def _au(self, logits, target, mask):
        if mask.sum() == 0:
            return logits.sum() * 0.0
        lg, tg = logits[mask], target[mask]
        valid = tg != -1
        if valid.sum() == 0:
            return logits.sum() * 0.0
        if self.au_loss == "asl":
            gn, gp, clip = self.asl_params
            return asl_loss(lg[valid], tg[valid].float(), gamma_neg=gn, gamma_pos=gp, clip=clip)
        loss = F.binary_cross_entropy_with_logits(
            lg, tg.float().clamp(min=0),
            pos_weight=self.au_pos_w if self.au_pos_w is not None else None,
            reduction="none")
        return loss[valid].mean()

    def task_losses(self, outputs, targets, masks):
        """The 3 raw per-task losses [VA, EXPR, AU] as tensors — for gradient surgery
        (PCGrad), which needs each task's gradient separately."""
        return [self._va(outputs["va"], targets["valence"], targets["arousal"], masks["va"]),
                self._expr(outputs["expr"], targets["expr"], masks["expr"]),
                self._au(outputs["au"], targets["au"], masks["au"])]

    def forward(self, outputs, targets, masks):
        """outputs: {va,(B,2) expr,(B,8) au,(B,12)}; targets: valence,arousal,expr,au;
        masks: va,expr,au boolean (B,). Returns (total_loss, parts_dict)."""
        l_va, l_expr, l_au = self.task_losses(outputs, targets, masks)
        parts = {"va": float(l_va.detach()), "expr": float(l_expr.detach()),
                 "au": float(l_au.detach())}
        if self.uncertainty_weighting:
            losses = torch.stack([l_va, l_expr, l_au])
            total = (0.5 * torch.exp(-self.log_var) * losses + 0.5 * self.log_var).sum()
            parts["log_var"] = self.log_var.detach().tolist()
        else:
            total = l_va + l_expr + l_au
        return total, parts
