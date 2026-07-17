from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    """CLIP ViT-B/16 visual tower. forward -> (B, output_dim) image embedding."""
    def __init__(self, input_resolution=224, patch_size=16, width=768,
                 layers=12, heads=12, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        scale = width ** -0.5
        n_patches = (input_resolution // patch_size) ** 2
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj


def build_farl(weights_path: str | None = None):
    """Return (VisualTransformer, feat_dim=512). Loads FaRL 'visual.*' weights if
    a path is given; raises a clear error if the file is missing."""
    model = VisualTransformer()
    if weights_path:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(
                f"FaRL weights not found at {p}. Fetch them (MIT, GitHub release):\n"
                "  curl -fL https://github.com/FacePerceiver/FaRL/releases/download/"
                "pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth -o " + str(p))
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        vis = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        missing, unexpected = model.load_state_dict(vis, strict=False)
        loaded = len(vis) - len(unexpected)
        print(f"[FaRL] loaded {loaded}/{len(vis)} visual tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if loaded < 100:
            raise RuntimeError(
                f"FaRL load looks wrong (only {loaded} tensors matched). Check the "
                "checkpoint key prefix (expected 'visual.*').")
    return model, model.output_dim
