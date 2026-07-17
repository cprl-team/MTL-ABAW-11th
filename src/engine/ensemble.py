from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.engine.eval_ckpt import collect
from src.engine.train import calibrate_au_thresholds
from src.eval.smoothing import smooth_streams
from src.metrics import all_metrics
from src.data import parse_annotations


def _metrics(va, expr_prob, au_prob, vt, at, et, au_t):
    return all_metrics(vt, va[:, 0], at, va[:, 1], et, expr_prob.argmax(1),
                       au_t, (au_prob >= 0.5).astype(int))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (each w/ best.pt + provenance)")
    ap.add_argument("--smooth", default="gaussian", choices=["none", "box", "gaussian"])
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--save", default=None, help="write metrics+calibrated AU thresholds JSON")
    args = ap.parse_args()

    from src.models import build_mtl_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vas, eps, aps = [], [], []
    truth = None
    va_ann = None
    for r in args.runs:
        run = Path(r)
        cfg = json.loads((run / "provenance.json").read_text())["config"]
        dcfg = cfg["data"]
        root = Path(dcfg["data_dir"])
        if va_ann is None:
            va_ann = parse_annotations(root / dcfg.get("val_ann", "validation_set_annotations.txt"))
        image_root = root / dcfg.get("image_subdir", "cropped_aligned")
        model = build_mtl_model(cfg["model"]).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        va, ep, apb, vt, at, et, au_t = collect(
            model, va_ann, image_root, device, dcfg.get("img_size", 112))
        vas.append(va); eps.append(ep); aps.append(apb)
        truth = (vt, at, et, au_t)
        print(f"  collected {run.name} ({cfg['model'].get('backbone')})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    va = np.mean(vas, 0); ep = np.mean(eps, 0); apb = np.mean(aps, 0)
    vt, at, et, au_t = truth
    raw = _metrics(va, ep, apb, vt, at, et, au_t)
    va_s, ep_s, ap_s = smooth_streams(va_ann.videos, va_ann.images, va, ep, apb,
                                      kind=args.smooth, window=args.window, sigma=args.sigma)
    sm = _metrics(va_s, ep_s, ap_s, vt, at, et, au_t)
    thr_cal, au_cal = calibrate_au_thresholds(au_t, apb)
    p_cal = raw["VA"] + raw["EXPR_macroF1"] + au_cal

    print(f"\n=== ENSEMBLE of {len(args.runs)} models ===")
    print(f"  raw       P_MTL={raw['P_MTL']:.4f}  VA={raw['VA']:.4f}  EXPR={raw['EXPR_macroF1']:.4f}  AU={raw['AU_macroF1']:.4f}")
    print(f"  +AU calib P_MTL={p_cal:.4f}  (AU {au_cal:.4f})")
    print(f"  +smoothed P_MTL={sm['P_MTL']:.4f}  VA={sm['VA']:.4f}  EXPR={sm['EXPR_macroF1']:.4f}  AU={sm['AU_macroF1']:.4f}")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({
            "runs": args.runs, "smooth": {"kind": args.smooth, "sigma": args.sigma,
                                          "window": args.window},
            "raw": raw, "smoothed": sm, "P_MTL_calibrated": p_cal,
            "au_thresholds": [float(x) for x in thr_cal]}, indent=2))
        print(f"  saved -> {args.save}")


if __name__ == "__main__":
    main()
