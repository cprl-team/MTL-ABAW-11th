from __future__ import annotations

import argparse

import numpy as np

from src.metrics.mtl import au_score, expr_score, va_score


def _macro(dump, task):
    """Return (labels, per-frame hard predictions, videos) for the task.
    For VA, 'labels'/'preds' are (N,2) stacks of (valence, arousal)."""
    if task == "expr":
        return dump["expr"], dump["pe"].argmax(1), dump["videos"]
    if task == "au":
        return dump["au"].astype(int), (dump["pa"] >= 0.5).astype(int), dump["videos"]
    if task == "va":
        yt = np.stack([dump["valence"], dump["arousal"]], 1)
        return yt, dump["pv"], dump["videos"]
    raise ValueError(task)


def _score(task, yt, yp):
    if task == "expr":
        return expr_score(yt, yp)[0]
    if task == "au":
        return au_score(yt, yp)[0]
    return va_score(yt[:, 0], yp[:, 0], yt[:, 1], yp[:, 1])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="ours (e.g. +FMAE) dump npz")
    ap.add_argument("--b", required=True, help="baseline dump npz")
    ap.add_argument("--task", choices=["expr", "au", "va"], required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A = np.load(args.a, allow_pickle=True)
    B = np.load(args.b, allow_pickle=True)
    yt_a, yp_a, vids = _macro(A, args.task)
    yt_b, yp_b, vids_b = _macro(B, args.task)
    assert len(yt_a) == len(yt_b) and (vids == vids_b).all(), "dumps are not frame-aligned"

    obs = _score(args.task, yt_a, yp_a) - _score(args.task, yt_b, yp_b)
    uniq = np.unique(vids)
    rows_by_vid = {v: np.where(vids == v)[0] for v in uniq}
    rng = np.random.default_rng(args.seed)
    diffs = np.empty(args.n_boot)
    for i in range(args.n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_vid[v] for v in pick])
        diffs[i] = (_score(args.task, yt_a[idx], yp_a[idx])
                    - _score(args.task, yt_b[idx], yp_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one = float((diffs <= 0).mean())
    print(f"task={args.task}  videos={len(uniq)}  n_boot={args.n_boot}")
    print(f"  observed delta (A-B) = {obs:+.4f}")
    print(f"  bootstrap mean       = {diffs.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  one-sided p (delta<=0) = {p_one:.4f}   "
          f"{'(significant, p<0.05)' if p_one < 0.05 else '(not significant at 0.05)'}")


if __name__ == "__main__":
    main()
