from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.affectnet import parse_affectnet
from src.engine.extract_features import collect_feats


class _Wrap:
    """collect_feats() calls model.eval() and model.backbone(x); wrap the raw FSFM encoder."""
    def __init__(self, bb):
        self.backbone = bb

    def eval(self):
        self.backbone.eval()
        return self


def main():
    import torch
    from src.models.fsfm import build_fsfm
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/fsfm_vitb_vf2_600e.pth")
    ap.add_argument("--manifest", default="data/affectnet/aligned_112_manifest.csv")
    ap.add_argument("--root", default="data/affectnet")
    ap.add_argument("--out", default="runs/affectnet_fsfm_feats")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb, fd = build_fsfm(args.weights)
    bb = bb.to(device)
    model = _Wrap(bb)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in ["val", "train"]:
        ann = parse_affectnet(args.manifest, split, root=args.root)
        d = collect_feats(model, ann, args.root, device, args.img_size)
        np.savez(out / f"{split}.npz", **d)
        nva = int(d["m_va"].sum())
        print(f"wrote {out}/{split}.npz  images={len(d['F'])}  dim={d['F'].shape[1]}  "
              f"VA-labeled={nva} ({100*nva/len(d['F']):.0f}%)  EXPR-labeled={int(d['m_expr'].sum())}")


if __name__ == "__main__":
    main()
