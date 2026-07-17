from __future__ import annotations

import argparse

import numpy as np

from src.eval.calibrate import expr_logit_adjust
from src.metrics.mtl import au_score, expr_score, va_score


def _va_moment_match(true, pred, valid):
    """Affine-rescale pred to the valid-frame mean/std of true. Returns full-length pred'."""
    t, p = true[valid], pred[valid]
    sp = p.std()
    if sp < 1e-8:
        return pred
    out = pred.copy().astype(float)
    out[valid] = (p - p.mean()) / sp * t.std() + t.mean()
    return out



def _fit_va(true, pred, valid):
    t, p = true[valid], pred[valid]
    return (p.mean(), p.std() if p.std() > 1e-8 else 1.0, t.mean(), t.std())


def _apply_va(pred, params):
    mp, sp, mt, st = params
    return (pred - mp) / sp * st + mt


def _fit_au_thr(pa, au):
    grid = np.linspace(0.05, 0.95, 19)
    thr = np.full(pa.shape[1], 0.5)
    for k in range(pa.shape[1]):
        yt = au[:, k].astype(int); m = yt != -1
        best_f, best_t = -1.0, 0.5
        for t in grid:
            pred = (pa[m, k] >= t).astype(int); y = yt[m]
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            f = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
            if f > best_f:
                best_f, best_t = f, t
        thr[k] = best_t
    return thr


def _crossfit(VA, EX, AU, prior, seed=0):
    """2-fold video cross-fit: each video scored with params fit on the OTHER fold,
    then score the pooled full-val predictions. Honest held-out estimate."""
    vids = VA["videos"]; uniq = np.unique(vids)
    rng = np.random.default_rng(seed); perm = rng.permutation(len(uniq))
    foldA = set(uniq[perm[: len(uniq) // 2]])
    inA = np.array([v in foldA for v in vids])
    vp, ap_ = VA["pv"][:, 0].astype(float), VA["pv"][:, 1].astype(float)
    vt, at, mva = VA["valence"], VA["arousal"], VA["m_va"].astype(bool)
    vp2, ap2 = vp.copy(), ap_.copy()
    for fit_mask, app_mask in [(~inA, inA), (inA, ~inA)]:
        vp2[app_mask] = _apply_va(vp[app_mask], _fit_va(vt, vp, mva & fit_mask))
        ap2[app_mask] = _apply_va(ap_[app_mask], _fit_va(at, ap_, mva & fit_mask))
    va_cf = va_score(vt, vp2, at, ap2)[0]
    evids = EX["videos"]; einA = np.array([v in foldA for v in evids])
    pe = EX["pe"]; pred_e = pe.argmax(1).copy()
    for fit_mask, app_mask in [(~einA, einA), (einA, ~einA)]:
        tau = expr_logit_adjust(EX["expr"][fit_mask], pe[fit_mask], prior)[0]
        logp = np.log(np.clip(pe[app_mask], 1e-8, 1.0)); logpi = np.log(np.clip(prior, 1e-8, 1.0))
        pred_e[app_mask] = (logp - tau * logpi).argmax(1)
    ex_cf = expr_score(EX["expr"], pred_e)[0]
    avids = AU["videos"]; ainA = np.array([v in foldA for v in avids])
    pa = AU["pa"]; pred_a = (pa >= 0.5).astype(int)
    for fit_mask, app_mask in [(~ainA, ainA), (ainA, ~ainA)]:
        thr = _fit_au_thr(pa[fit_mask], AU["au"][fit_mask])
        pred_a[app_mask] = (pa[app_mask] >= thr[None, :]).astype(int)
    au_cf = au_score(AU["au"].astype(int), pred_a)[0]
    return va_cf, ex_cf, au_cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--va", required=True)
    ap.add_argument("--expr", required=True)
    ap.add_argument("--au", required=True)
    ap.add_argument("--expr-prior", required=True)
    args = ap.parse_args()

    VA = np.load(args.va, allow_pickle=True)
    EX = np.load(args.expr, allow_pickle=True)
    AU = np.load(args.au, allow_pickle=True)
    prior = np.load(args.expr_prior)

    vt, at = VA["valence"], VA["arousal"]
    vp, ap_ = VA["pv"][:, 0], VA["pv"][:, 1]
    mva = VA["m_va"].astype(bool)
    va0 = va_score(vt, vp, at, ap_)[0]
    vp2 = _va_moment_match(vt, vp, mva)
    ap2 = _va_moment_match(at, ap_, mva)
    va1 = va_score(vt, vp2, at, ap2)[0]

    ex0 = expr_score(EX["expr"], EX["pe"].argmax(1))[0]
    tau, ex1, _, _ = expr_logit_adjust(EX["expr"], EX["pe"], prior)

    au0 = au_score(AU["au"].astype(int), (AU["pa"] >= 0.5).astype(int))[0]
    thr = np.full(AU["pa"].shape[1], 0.5)
    grid = np.linspace(0.05, 0.95, 19)
    for k in range(AU["pa"].shape[1]):
        best_f, best_t = -1.0, 0.5
        for t in grid:
            pred = (AU["pa"][:, k] >= t).astype(int)
            yt = AU["au"][:, k].astype(int)
            m = yt != -1
            tp = int(((pred[m] == 1) & (yt[m] == 1)).sum())
            fp = int(((pred[m] == 1) & (yt[m] == 0)).sum())
            fn = int(((pred[m] == 0) & (yt[m] == 1)).sum())
            f = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
            if f > best_f:
                best_f, best_t = f, t
        thr[k] = best_t
    pred_au = (AU["pa"] >= thr[None, :]).astype(int)
    au1 = au_score(AU["au"].astype(int), pred_au)[0]

    cf = np.array([_crossfit(VA, EX, AU, prior, seed=s) for s in range(5)])
    va_cf, ex_cf, au_cf = cf.mean(0)

    print("Tier-0 post-hoc calibration:")
    print(f"  {'':6} {'base':>8} {'dev-fit':>9} {'crossfit':>9}   (crossfit = honest held-out)")
    print(f"  VA   : {va0:8.4f} {va1:9.4f} {va_cf:9.4f}   [moment-match]")
    print(f"  EXPR : {ex0:8.4f} {ex1:9.4f} {ex_cf:9.4f}   [logit-adjust tau={tau:.2f}]")
    print(f"  AU   : {au0:8.4f} {au1:9.4f} {au_cf:9.4f}   [per-AU thresholds]")
    print(f"  P_MTL: {va0+ex0+au0:8.4f} {va1+ex1+au1:9.4f} {va_cf+ex_cf+au_cf:9.4f}")
    print(f"  dev-fit delta  = {(va1+ex1+au1)-(va0+ex0+au0):+.4f}  (optimistic, fit==score)")
    print(f"  crossfit delta = {(va_cf+ex_cf+au_cf)-(va0+ex0+au0):+.4f}  (honest held-out estimate)")
    print(f"  AU thresholds (full-val): {np.round(thr, 2).tolist()}")


if __name__ == "__main__":
    main()
