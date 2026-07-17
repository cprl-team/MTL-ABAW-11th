from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.rafdb import parse_rafdb_compound
from src.engine.extract_features import collect_feats
from src.engine.extract_affectnet import _Wrap


def main():
    import torch
    from src.models.fsfm import build_fsfm
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/fsfm_vitb_vf2_600e.pth")
    ap.add_argument("--labels", default="data/RAF-DB/compound/EmoLabel/list_patition_label.txt")
    ap.add_argument("--root", default="data/RAF-DB")
    ap.add_argument("--out", default="runs/rafdb_comp_feats")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb, _ = build_fsfm(args.weights)
    model = _Wrap(bb.to(device))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        ann = parse_rafdb_compound(args.labels, split, root=args.root)
        d = collect_feats(model, ann, args.root, device, args.img_size)
        np.savez(out / f"{split}.npz", F=d["F"], label=d["expr"])
        print(f"wrote {out}/{split}.npz  images={len(d['F'])}  dim={d['F'].shape[1]}  "
              f"classes={len(np.unique(d['expr']))}")


if __name__ == "__main__":
    main()
