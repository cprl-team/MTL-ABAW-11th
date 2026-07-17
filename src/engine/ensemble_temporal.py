from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.engine.train_temporal import TemporalMTL, TemporalMTLLatent, group_videos
from src.metrics import all_metrics


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
    ap.add_argument("--member", action="append", required=True,
                    help="a feature-group of models: 'featdir1[,featdir2]::ckpt1[,ckpt2,...]'. "
                         "Repeat --member to ensemble across different feature groups "
                         "(e.g. single-backbone + cross-backbone-concat).")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--save", default=None, help="write the ensemble metrics JSON here")
    ap.add_argument("--dump", default=None,
                    help="write per-frame predictions+labels+videos NPZ here (for significance tests)")
    ap.add_argument("--latent", action="store_true",
                    help="members are D2 shared-latent heads (TemporalMTLLatent), e.g. EXPR dumps")
    ap.add_argument("--zdim", type=int, default=96)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"],
                    help="feats split to run (test feats are extracted as the 'val' split by convention)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    groups = []
    N = None; ref = None; total = 0
    for spec in args.member:
        fpart, cpart = spec.split("::")
        feat_dirs = [Path(p) for p in fpart.split(",")]
        ckpts = cpart.split(",")
        va = load_split(feat_dirs, args.split)
        seqs = group_videos(va)
        feat_dim = va["F"].shape[1]
        models = []
        for c in ckpts:
            m = (TemporalMTLLatent(feat_dim, args.hidden, args.layers, 0.0, zdim=args.zdim)
                 if args.latent else TemporalMTL(feat_dim, args.hidden, args.layers, 0.0)).to(device)
            m.load_state_dict(torch.load(c, map_location=device), strict=False); m.eval()
            models.append(m)
        groups.append((seqs, models)); total += len(models)
        if ref is None:
            ref = va; N = len(va["F"])
        print(f"  group: feat_dim={feat_dim}  models={len(models)}")
    print(f"ensembling {total} temporal heads across {len(groups)} feature group(s)")

    pv = np.zeros((N, 2)); pe = np.zeros((N, 8)); pa = np.zeros((N, 12))
    with torch.no_grad():
        for seqs, models in groups:
            for s in seqs:
                x = torch.from_numpy(s["F"]).float().unsqueeze(0).to(device)
                for m in models:
                    o = m(x)
                    pv[s["rows"]] += o["va"][0].cpu().numpy()
                    pe[s["rows"]] += torch.softmax(o["expr"][0], 1).cpu().numpy()
                    pa[s["rows"]] += torch.sigmoid(o["au"][0]).cpu().numpy()
    pv /= total; pe /= total; pa /= total
    va = ref
    m = all_metrics(va["valence"], pv[:, 0], va["arousal"], pv[:, 1],
                    va["expr"], pe.argmax(1), va["au"].astype(int), (pa >= 0.5).astype(int))
    print(f"  ENSEMBLE P_MTL={m['P_MTL']:.4f}  VA={m['VA']:.4f} "
          f"(v{m['CCC_valence']:.3f}/a{m['CCC_arousal']:.3f})  "
          f"EXPR={m['EXPR_macroF1']:.4f}  AU={m['AU_macroF1']:.4f}")
    if args.save:
        import json
        Path(args.save).write_text(json.dumps(m, indent=2))
        print(f"  wrote {args.save}")
    if args.dump:
        np.savez(args.dump, pv=pv, pe=pe, pa=pa,
                 valence=va["valence"], arousal=va["arousal"],
                 expr=va["expr"], au=va["au"].astype(int),
                 m_va=va["m_va"], m_expr=va["m_expr"], m_au=va["m_au"],
                 videos=va["videos"])
        print(f"  dumped per-frame preds -> {args.dump}")


if __name__ == "__main__":
    main()
