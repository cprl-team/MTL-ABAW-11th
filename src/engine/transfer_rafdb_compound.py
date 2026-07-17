from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.engine.train_affectnet_latent import StaticLatent, _simulate_partial
from src.losses.mtl import ccc_loss_1d

ROOT = Path(__file__).resolve().parents[2]


def _train_source(Xtr, Etr, Vtr, me, mv, latent, seed, device, epochs=25, bs=512, beta=0.05, kw=8):
    torch.manual_seed(seed)
    model = StaticLatent(Xtr.shape[1], zdim=96, latent=latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    N = len(Xtr)
    me_t, mv_t = torch.tensor(me), torch.tensor(mv)
    for ep in range(epochs):
        model.train()
        b = beta * min(1.0, (ep + 1) / kw) if latent else 0.0
        order = torch.randperm(N)
        for i in range(0, N, bs):
            idx = order[i:i + bs]
            x = Xtr[idx].to(device); out = model(x)
            m_e, m_v = me_t[idx].to(device), mv_t[idx].to(device)
            loss = x.new_zeros(())
            if m_e.any():
                loss = loss + F.cross_entropy(out["expr"][m_e], Etr[idx][m_e.cpu()].to(device))
            if m_v.any():
                pv = out["va"][m_v]; tv = Vtr[idx][m_v.cpu()].to(device)
                loss = loss + ccc_loss_1d(pv[:, 0], tv[:, 0]) + ccc_loss_1d(pv[:, 1], tv[:, 1])
            if latent and b > 0:
                mu, lv = out["mu"], out["logvar"]
                loss = loss + b * (0.5 * (mu.pow(2) + lv.exp() - 1 - lv)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def _probe(rep_tr, ytr, rep_te, yte, seed):
    sc = StandardScaler().fit(rep_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(rep_tr), ytr)
    return f1_score(yte, clf.predict(sc.transform(rep_te)), average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--an", default="runs/affectnet_fsfm_feats")
    ap.add_argument("--comp", default="runs/rafdb_comp_feats")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--subsample", type=int, default=60000)
    ap.add_argument("--out", default="runs/rafdb_transfer")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    an = np.load(Path(args.an) / "train.npz", allow_pickle=True)
    Xan = torch.tensor(an["F"], dtype=torch.float32)
    Ean = torch.tensor(an["expr"], dtype=torch.long)
    Van = torch.tensor(np.stack([an["valence"], an["arousal"]], 1), dtype=torch.float32)
    me0, mv0 = an["m_expr"].astype(bool), an["m_va"].astype(bool)

    ctr = np.load(Path(args.comp) / "train.npz"); cte = np.load(Path(args.comp) / "test.npz")
    Ctr = torch.tensor(ctr["F"], dtype=torch.float32).to(device)
    Cte = torch.tensor(cte["F"], dtype=torch.float32).to(device)
    ytr, yte = ctr["label"], cte["label"]

    raw_f1 = _probe(ctr["F"], ytr, cte["F"], yte, 0)
    lat, base = [], []
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        if args.subsample and args.subsample < len(Xan):
            sel = rng.choice(len(Xan), args.subsample, replace=False)
            Xs, Es, Vs, mes, mvs = Xan[sel], Ean[sel], Van[sel], me0[sel], mv0[sel]
        else:
            Xs, Es, Vs, mes, mvs = Xan, Ean, Van, me0, mv0
        if args.overlap < 1.0:
            mes, mvs = _simulate_partial(mes, mvs, args.overlap, rng)
        for latent, store in [(True, lat), (False, base)]:
            m = _train_source(Xs, Es, Vs, mes, mvs, latent, s, device)
            rtr = m.represent(Ctr).cpu().numpy(); rte = m.represent(Cte).cpu().numpy()
            store.append(_probe(rtr, ytr, rte, yte, s))
        print(f"  seed {s}: latent {lat[-1]:.4f}  base {base[-1]:.4f}  (raw {raw_f1:.4f})")

    lat, base = np.array(lat), np.array(base)
    from scipy.stats import wilcoxon
    p = float(wilcoxon(lat, base).pvalue) if args.seeds >= 6 else float("nan")
    rec = {"raw_fsfm_f1": float(raw_f1), "latent_f1": float(lat.mean()), "latent_sd": float(lat.std()),
           "base_f1": float(base.mean()), "base_sd": float(base.std()),
           "delta": float((lat - base).mean()), "wins": int((lat > base).sum()),
           "seeds": args.seeds, "p": p, "overlap": args.overlap, "subsample": args.subsample}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rec, indent=2))
    print(f"\nRAF-DB compound transfer (macro-F1, {args.seeds} seeds): "
          f"latent {rec['latent_f1']:.4f}+-{rec['latent_sd']:.4f}  "
          f"base {rec['base_f1']:.4f}+-{rec['base_sd']:.4f}  "
          f"delta {rec['delta']:+.4f}  wins {rec['wins']}/{args.seeds}  p={rec['p']:.3f}  "
          f"(raw FSFM floor {raw_f1:.4f})")


if __name__ == "__main__":
    main()
