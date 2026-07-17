from __future__ import annotations

import torch
import torch.nn as nn

_BACKBONES: dict = {}


def register_backbone(name):
    def deco(fn):
        _BACKBONES[name] = fn
        return fn
    return deco


def build_backbone(name: str, pretrained: bool = True, freeze: bool = False):
    """Return (feature_extractor: nn.Module, feat_dim: int) mapping (B,3,H,W) -> (B,feat_dim)."""
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone '{name}'; registered: {sorted(_BACKBONES)}")
    backbone, feat_dim = _BACKBONES[name](pretrained)
    if freeze:
        for p in backbone.parameters():
            p.requires_grad_(False)
    return backbone, feat_dim


@register_backbone("resnet18")
def _resnet18(pretrained):
    from torchvision import models
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=w)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim


@register_backbone("resnet50")
def _resnet50(pretrained):
    from torchvision import models
    w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=w)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim


class MTLHead(nn.Module):
    """Optional bottleneck + linear head."""
    def __init__(self, in_dim, out_dim, hidden=0, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        if hidden:
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        if dropout:
            layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RegionAUHead(nn.Module):
    """Per-AU spatial attention over region-pooled ViT patch tokens (Stage-A1 finetuning objective).
    The AU loss on this head forces the backbone to LOCALIZE action units in the patch grid (brow/eye/
    lip regions) -- the global CLS objective never does, which is why frozen patch features carried no
    extra AU signal. Discarded at extraction time; the finetuned backbone keeps the localized signal."""
    def __init__(self, d, R=3, n_au=12):
        super().__init__()
        self.R = R; self.d = d
        self.q = nn.Parameter(torch.randn(n_au, d) * 0.02)
        self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.w = nn.Parameter(torch.randn(n_au, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_au))

    def forward(self, patches):
        B, P, D = patches.shape
        g = int(round(P ** 0.5))
        pg = patches.reshape(B, g, g, D).permute(0, 3, 1, 2)
        reg = nn.functional.adaptive_avg_pool2d(pg, self.R).flatten(2).transpose(1, 2)
        k = self.k(reg); v = self.v(reg)
        att = torch.softmax(self.q @ k.transpose(-1, -2) / self.d ** 0.5, dim=-1)
        return (att @ v * self.w).sum(-1) + self.b


class MTLModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, freeze_backbone=False,
                 head_hidden=0, dropout=0.0, n_expr=8, n_au=12, weights_path=None, region_au=0,
                 lora=None):
        super().__init__()
        if backbone == "farl":
            from src.models.farl import build_farl
            self.backbone, feat_dim = build_farl(weights_path)
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
        elif backbone == "fsfm":
            from src.models.fsfm import build_fsfm
            self.backbone, feat_dim = build_fsfm(weights_path)
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
        elif backbone in ("fmae", "maeface"):
            from src.models.fmae import build_fmae
            self.backbone, feat_dim = build_fmae(weights_path)
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
        elif backbone == "dinov2":
            from src.models.dinov2 import build_dinov2
            self.backbone, feat_dim = build_dinov2(weights_path)
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
        elif backbone.startswith("timm:"):
            import timm
            self.backbone = timm.create_model(backbone.split(":", 1)[1],
                                              pretrained=pretrained, num_classes=0,
                                              global_pool="avg")
            feat_dim = self.backbone.num_features
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
        else:
            self.backbone, feat_dim = build_backbone(backbone, pretrained, freeze_backbone)
        if lora:
            from src.models.lora import wrap_lora, set_lora_trainable
            self.backbone = wrap_lora(self.backbone, rank=lora.get("rank", 16),
                                      alpha=lora.get("alpha", 32), dropout=lora.get("dropout", 0.0),
                                      targets=lora.get("targets"))
            f, t = set_lora_trainable(self.backbone)
            print(f"  [lora] peft LoRA rank={lora.get('rank', 16)}: base frozen ({f}), adapters ({t})")
        self.feat_dim = feat_dim
        self.va_head = MTLHead(feat_dim, 2, head_hidden, dropout)
        self.expr_head = MTLHead(feat_dim, n_expr, head_hidden, dropout)
        self.au_head = MTLHead(feat_dim, n_au, head_hidden, dropout)
        self.region_au = region_au
        self.region_au_head = RegionAUHead(feat_dim, region_au, n_au) if region_au else None

    def forward(self, x):
        if self.region_au:
            tok = self.backbone.forward_features(x)
            f = self.backbone.forward_head(tok, pre_logits=True)
            return {"va": torch.tanh(self.va_head(f)), "expr": self.expr_head(f),
                    "au": self.au_head(f), "au_region": self.region_au_head(tok[:, 1:])}
        f = self.backbone(x)
        return {"va": torch.tanh(self.va_head(f)),
                "expr": self.expr_head(f),
                "au": self.au_head(f)}


def build_mtl_model(cfg: dict) -> MTLModel:
    cfg = dict(cfg)
    return MTLModel(
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=cfg.get("pretrained", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
        head_hidden=cfg.get("head_hidden", 0),
        dropout=cfg.get("dropout", 0.0),
        weights_path=cfg.get("weights_path"),
        region_au=cfg.get("region_au", 0),
        lora=cfg.get("lora"),
    )
