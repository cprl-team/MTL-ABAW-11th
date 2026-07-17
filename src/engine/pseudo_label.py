from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.engine.train_temporal import TemporalMTL, group_videos


def load_split(feat_dirs, split):
    parts = [np.load(d / f"{split}.npz", allow_pickle=True) for d in feat_dirs]
    base = parts[0]
    d = {"F": np.concatenate([p["F"] for p in parts], axis=1)}
    for k in ("valence", "arousal", "expr", "au", "m_va", "m_expr", "m_au",
              "videos", "images"):
        d[k] = base[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--task", default="expr", choices=["expr"])
    ap.add_argument("--thr", type=float, default=0.9, help="min teacher confidence")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dirs = [Path(p) for p in args.feats]

    tr = load_split(feat_dirs, "train")
    feat_dim = tr["F"].shape[1]
    seqs = group_videos(tr)
    models = []
    for c in args.teachers:
        m = TemporalMTL(feat_dim, args.hidden, args.layers, 0.0).to(device)
        m.load_state_dict(torch.load(c, map_location=device)); m.eval()
        models.append(m)

    N = len(tr["F"])
    prob = np.zeros((N, 8))
    with torch.no_grad():
        for s in seqs:
            x = torch.from_numpy(s["F"]).float().unsqueeze(0).to(device)
            acc = None
            for m in models:
                p = torch.softmax(m(x)["expr"][0], 1)
                acc = p if acc is None else acc + p
            prob[s["rows"]] = (acc / len(models)).cpu().numpy()

    conf = prob.max(1); pred = prob.argmax(1)
    missing = ~tr["m_expr"]
    add = missing & (conf >= args.thr)
    expr2 = tr["expr"].copy(); mexpr2 = tr["m_expr"].copy()
    expr2[add] = pred[add].astype(expr2.dtype); mexpr2[add] = True
    print(f"missing-EXPR frames: {int(missing.sum())}  "
          f"pseudo-labeled (conf>={args.thr}): {int(add.sum())}  "
          f"-> EXPR labeled {int(tr['m_expr'].sum())} -> {int(mexpr2.sum())} "
          f"(+{100*add.sum()/N:.1f}% of frames)")
    import collections
    print("  pseudo-class counts:", dict(collections.Counter(pred[add].tolist())))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = feat_dirs[0]
    np.savez(out / "train.npz", F=tr["F"], valence=tr["valence"], arousal=tr["arousal"],
             expr=expr2, au=tr["au"], m_va=tr["m_va"], m_expr=mexpr2, m_au=tr["m_au"],
             videos=tr["videos"], images=tr["images"])
    shutil.copy(src / "val.npz", out / "val.npz")
    print(f"wrote {out}/train.npz (+ copied val.npz)")


if __name__ == "__main__":
    main()
