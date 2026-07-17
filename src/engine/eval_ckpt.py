from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import (build_torch_dataset, class_balance,
                      parse_annotations)
from src.metrics import all_metrics
from src.metrics.mtl import N_EXPR
from src.eval.smoothing import smooth_streams
from src.eval.calibrate import expr_logit_adjust


def collect(model, ann, image_root, device, img_size, bs=128, workers=6):
    import torch
    from torch.utils.data import DataLoader
    ds = build_torch_dataset(ann, image_root, train=False, img_size=img_size)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=(device == "cuda"))
    V, A, Ep, Ap, Vt, At, Et, At_au = ([] for _ in range(8))
    model.eval()
    with torch.no_grad():
        for x, t, m in ld:
            x = x.to(device, non_blocking=True)
            out = model(x)
            va = out["va"].float().cpu().numpy()
            V.append(va[:, 0]); A.append(va[:, 1])
            Ep.append(torch.softmax(out["expr"], 1).float().cpu().numpy())
            Ap.append(torch.sigmoid(out["au"]).float().cpu().numpy())
            Vt.append(t["valence"].numpy()); At.append(t["arousal"].numpy())
            Et.append(t["expr"].numpy()); At_au.append(t["au"].numpy().astype(int))
    cat = lambda L: np.concatenate(L, 0)
    return (np.stack([cat(V), cat(A)], 1), cat(Ep), cat(Ap),
            cat(Vt), cat(At), cat(Et), cat(At_au))


def _metrics(va, expr_prob, au_prob, vt, at, et, au_t):
    return all_metrics(vt, va[:, 0], at, va[:, 1], et, expr_prob.argmax(1),
                       au_t, (au_prob >= 0.5).astype(int))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with best.pt + provenance.json")
    ap.add_argument("--smooth", default="gaussian", choices=["none", "box", "gaussian"])
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--sigma", type=float, default=2.0)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "provenance.json").read_text())["config"]
    dcfg = cfg["data"]
    root = Path(dcfg["data_dir"])
    va_ann = parse_annotations(root / dcfg.get("val_ann", "validation_set_annotations.txt"))
    image_root = root / dcfg.get("image_subdir", "cropped_aligned")
    img_size = dcfg.get("img_size", 112)

    from src.models import build_mtl_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mtl_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))

    va, ep, apb, vt, at, et, au_t = collect(model, va_ann, image_root, device, img_size)
    raw = _metrics(va, ep, apb, vt, at, et, au_t)
    va_s, ep_s, ap_s = smooth_streams(va_ann.videos, va_ann.images, va, ep, apb,
                                      kind=args.smooth, window=args.window, sigma=args.sigma)
    sm = _metrics(va_s, ep_s, ap_s, vt, at, et, au_t)

    def line(tag, m):
        return (f"  {tag:9s} P_MTL={m['P_MTL']:.4f}  VA={m['VA']:.4f} "
                f"(v{m['CCC_valence']:.3f}/a{m['CCC_arousal']:.3f})  "
                f"EXPR={m['EXPR_macroF1']:.4f}  AU={m['AU_macroF1']:.4f}")
    print(f"checkpoint: {run}/best.pt   smoothing: {args.smooth} (win={args.window}, sigma={args.sigma})")
    print(line("raw", raw))
    print(line("smoothed", sm))
    print(f"  delta     P_MTL={sm['P_MTL']-raw['P_MTL']:+.4f}  "
          f"VA={sm['VA']-raw['VA']:+.4f}  EXPR={sm['EXPR_macroF1']-raw['EXPR_macroF1']:+.4f}  "
          f"AU={sm['AU_macroF1']-raw['AU_macroF1']:+.4f}")

    tr_ann = parse_annotations(root / dcfg.get("train_ann", "training_set_annotations.txt"))
    cb = class_balance(tr_ann)
    prior = np.array([cb["expr_counts"].get(c, 0) for c in range(N_EXPR)], float)
    prior = prior / prior.sum()
    names = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]
    tau, adj_macro, adj_pc, _ = expr_logit_adjust(et, ep, prior)
    print(f"\nEXPR logit-adjustment (tau*={tau:.2f}): macro-F1 "
          f"{raw['EXPR_macroF1']:.4f} -> {adj_macro:.4f}  (delta {adj_macro-raw['EXPR_macroF1']:+.4f})")
    base_pc = [raw["EXPR_per_class"][str(c)] for c in range(N_EXPR)]
    for c in range(N_EXPR):
        mark = "  <-- rare" if names[c] in ("Anger", "Fear", "Disgust", "Sadness") else ""
        print(f"    {names[c]:9s}: {base_pc[c]:.3f} -> {adj_pc[c]:.3f}{mark}")
    print(f"  P_MTL with adjusted EXPR: {raw['VA']+adj_macro+raw['AU_macroF1']:.4f} (raw {raw['P_MTL']:.4f})")

    (run / "metrics_smoothed.json").write_text(json.dumps(
        {"smoothing": {"kind": args.smooth, "window": args.window, "sigma": args.sigma},
         "raw": raw, "smoothed": sm,
         "expr_logit_adjust": {"tau": tau, "macro_f1": adj_macro,
                               "per_class": {names[c]: adj_pc[c] for c in range(N_EXPR)}}},
        indent=2))


if __name__ == "__main__":
    main()
