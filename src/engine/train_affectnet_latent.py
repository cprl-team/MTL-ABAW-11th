from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.losses.mtl import ccc_loss_1d
from src.metrics.mtl import expr_score, va_score


class StaticLatent(nn.Module):
    """Shared affect latent (latent=True) or masked-loss baseline (latent=False), over per-image
    features. The latent mirrors TemporalMTLLatent without the BiGRU: a stochastic z mediates both
    task decoders, trained with the masked loss + beta*KL."""

    def __init__(self, d, zdim=96, hidden=256, latent=True, dropout=0.3):
        super().__init__()
        self.latent = latent
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        if latent:
            self.zdim = zdim
            self.enc = nn.Linear(hidden, 2 * zdim)
            self.expr = nn.Linear(zdim, 8)
            self.va = nn.Linear(zdim, 2)
        else:
            self.expr = nn.Linear(hidden, 8)
            self.va = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.proj(x)
        if not self.latent:
            return {"expr": self.expr(h), "va": torch.tanh(self.va(h))}
        mu, logvar = self.enc(h).chunk(2, dim=-1)
        logvar = logvar.clamp(-6, 6)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return {"expr": self.expr(z), "va": torch.tanh(self.va(z)), "mu": mu, "logvar": logvar}

    @torch.no_grad()
    def represent(self, x):
        """The representation the EXPR decoder reads: z (=mu) for the latent, h for the baseline.
        Used for the RAF-DB-compound transfer probe."""
        h = self.proj(x)
        return self.enc(h).chunk(2, dim=-1)[0] if self.latent else h


def _simulate_partial(m_expr, m_va, overlap, rng):
    """Keep both labels for `overlap` of the (both-labeled) rows; split the rest EXPR-only/VA-only."""
    both = m_expr & m_va
    idx = np.where(both)[0]
    rng.shuffle(idx)
    n_keep = int(round(overlap * len(idx)))
    drop = idx[n_keep:]
    half = len(drop) // 2
    me, mv = m_expr.copy(), m_va.copy()
    mv[drop[:half]] = False
    me[drop[half:]] = False
    return me, mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="runs/affectnet_fsfm_feats")
    ap.add_argument("--latent", action="store_true")
    ap.add_argument("--zdim", type=int, default=96)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--kl-warmup", type=int, default=8)
    ap.add_argument("--overlap", type=float, default=1.0, help="1.0 = full labels (no simulation)")
    ap.add_argument("--subsample", type=int, default=0, help="subsample train to N images (0=all); "
                    "the latent's cross-task coupling should matter under label scarcity")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = Path(args.feats)

    def load(split):
        d = np.load(feats / f"{split}.npz", allow_pickle=True)
        return (torch.tensor(d["F"], dtype=torch.float32),
                torch.tensor(d["expr"], dtype=torch.long),
                torch.tensor(np.stack([d["valence"], d["arousal"]], 1), dtype=torch.float32),
                d["m_expr"].astype(bool), d["m_va"].astype(bool))

    Xtr, Etr, Vtr, me_tr, mv_tr = load("train")
    Xva, Eva, Vva, me_va, mv_va = load("val")
    if args.subsample and args.subsample < len(Xtr):
        sel = rng.choice(len(Xtr), args.subsample, replace=False)
        Xtr, Etr, Vtr, me_tr, mv_tr = Xtr[sel], Etr[sel], Vtr[sel], me_tr[sel], mv_tr[sel]
    if args.overlap < 1.0:
        me_tr, mv_tr = _simulate_partial(me_tr, mv_tr, args.overlap, rng)
    me_t = torch.tensor(me_tr); mv_t = torch.tensor(mv_tr)

    model = StaticLatent(Xtr.shape[1], zdim=args.zdim, latent=args.latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    N = len(Xtr)
    best = {"P": -1e9}
    for ep in range(args.epochs):
        model.train()
        beta = args.beta * min(1.0, (ep + 1) / max(1, args.kl_warmup)) if args.latent else 0.0
        order = torch.randperm(N)
        for i in range(0, N, args.bs):
            b = order[i:i + args.bs]
            x = Xtr[b].to(device)
            out = model(x)
            me, mv = me_t[b].to(device), mv_t[b].to(device)
            loss = x.new_zeros(())
            if me.any():
                loss = loss + F.cross_entropy(out["expr"][me], Etr[b][me.cpu()].to(device))
            if mv.any():
                pv = out["va"][mv]; tv = Vtr[b][mv.cpu()].to(device)
                loss = loss + ccc_loss_1d(pv[:, 0], tv[:, 0]) + ccc_loss_1d(pv[:, 1], tv[:, 1])
            if args.latent and beta > 0:
                mu, lv = out["mu"], out["logvar"]
                loss = loss + beta * (0.5 * (mu.pow(2) + lv.exp() - 1 - lv)).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            o = model(Xva.to(device))
            pe = o["expr"].argmax(1).cpu().numpy()
            pv = o["va"].cpu().numpy()
        ef = expr_score(Eva.numpy(), pe)[0]
        vc = va_score(Vva[:, 0].numpy(), pv[:, 0], Vva[:, 1].numpy(), pv[:, 1])[0]
        P = ef + vc
        if P > best["P"]:
            best = {"P": float(P), "EXPR_F1": float(ef), "VA_CCC": float(vc), "epoch": ep}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = {**best, "latent": args.latent, "overlap": args.overlap, "beta": args.beta,
           "zdim": args.zdim, "seed": args.seed,
           "train_expr_labeled": int(me_tr.sum()), "train_va_labeled": int(mv_tr.sum())}
    (out / "metrics.json").write_text(json.dumps(rec, indent=2))
    print(f"[{'LATENT' if args.latent else 'BASE  '} o={args.overlap} s={args.seed}] "
          f"EXPR={best['EXPR_F1']:.4f} VA={best['VA_CCC']:.4f} P={best['P']:.4f} @ep{best['epoch']}")


if __name__ == "__main__":
    main()
