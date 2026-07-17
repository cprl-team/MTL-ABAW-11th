from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "ddamfn"


class DDAMFNBackbone(nn.Module):
    out_dim = 512

    def __init__(self, weights_path, num_head=2):
        super().__init__()
        if str(_VENDOR) not in sys.path:
            sys.path.insert(0, str(_VENDOR))
        from networks.DDAM import DDAMNet
        self.net = DDAMNet(num_class=8, num_head=num_head, pretrained=False)
        ckpt = torch.load(weights_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        print(f"[DDAMFN] loaded {weights_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    def forward(self, x):
        n = self.net
        f = n.features(x)
        heads = [getattr(n, f"cat_head{i}")(f) for i in range(n.num_head)]
        y = heads[0]
        for i in range(1, n.num_head):
            y = torch.max(y, heads[i])
        y = f * y
        return n.flatten(n.Linear(y))

    def forward_features(self, x):
        return self.forward(x)


def build_ddamfn(weights_path, **_):
    m = DDAMFNBackbone(weights_path)
    return m, m.out_dim
