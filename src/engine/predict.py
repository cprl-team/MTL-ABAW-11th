from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.engine.eval_ckpt import collect
from src.eval.smoothing import smooth_streams
from src.data import parse_annotations
from src.metrics.mtl import AU_NAMES, N_AU


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dir(s); >1 = ensemble")
    ap.add_argument("--ann", required=True, help="annotation/image-list file to predict on")
    ap.add_argument("--out", default="submission.txt")
    ap.add_argument("--smooth", default="gaussian", choices=["none", "box", "gaussian"])
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--au-thresholds", default=None,
                    help="metrics_smoothed.json with calibrated per-AU thresholds (else 0.5)")
    args = ap.parse_args()

    from src.models import build_mtl_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = parse_annotations(args.ann)

    vas, eps, aps = [], [], []
    for r in args.runs:
        run = Path(r)
        cfg = json.loads((run / "provenance.json").read_text())["config"]
        dcfg = cfg["data"]
        image_root = Path(dcfg["data_dir"]) / dcfg.get("image_subdir", "cropped_aligned")
        model = build_mtl_model(cfg["model"]).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        va, ep, apb, *_ = collect(model, ann, image_root, device, dcfg.get("img_size", 112))
        vas.append(va); eps.append(ep); aps.append(apb)
        print(f"  inferred {run.name}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    va = np.mean(vas, 0); ep = np.mean(eps, 0); apb = np.mean(aps, 0)
    if args.smooth != "none":
        va, ep, apb = smooth_streams(ann.videos, ann.images, va, ep, apb,
                                     kind=args.smooth, sigma=args.sigma)

    thr = np.full(N_AU, 0.5)
    if args.au_thresholds:
        t = json.loads(Path(args.au_thresholds).read_text())
        loaded = (t.get("au_thresholds") or t.get("AU_thresholds")
                  or t.get("smoothing", {}).get("au_thresholds"))
        if loaded is None:
            raise SystemExit(f"no au_thresholds found in {args.au_thresholds}")
        thr = np.array(loaded, dtype=float)
        print(f"  using calibrated AU thresholds from {args.au_thresholds}")

    v = np.clip(va[:, 0], -1, 1)
    a = np.clip(va[:, 1], -1, 1)
    expr = ep.argmax(1)
    au = (apb >= thr).astype(int)

    out = Path(args.out)
    with open(out, "w") as f:
        f.write("image,valence,arousal,expression,aus\n")
        for i in range(len(ann)):
            aus = ",".join(str(int(x)) for x in au[i])
            f.write(f"{ann.images[i]},{v[i]:.4f},{a[i]:.4f},{int(expr[i])},{aus}\n")
    print(f"\nwrote {len(ann)} predictions -> {out}")
    print(f"  expr dist: {np.bincount(expr, minlength=8).tolist()}")
    print(f"  AU pos-rate: {[round(float(au[:,k].mean()),2) for k in range(N_AU)]} ({','.join(AU_NAMES)})")
    print(f"  V range [{v.min():.2f},{v.max():.2f}]  A range [{a.min():.2f},{a.max():.2f}]")


if __name__ == "__main__":
    main()
