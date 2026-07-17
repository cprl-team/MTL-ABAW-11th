from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import class_balance, parse_annotations
from src.engine.train import _expr_class_weights, _au_pos_weight
from src.eval.smoothing import _frame_num
from src.metrics import all_metrics
from src.utils.seeding import set_all_seeds


class _GradReverse(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward (DANN/IAT)."""
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb=1.0):
    return _GradReverse.apply(x, lamb)


class TemporalMTL(nn.Module):
    """BiGRU over per-frame feature sequences -> per-frame VA/EXPR/AU.

    With n_ids set, also carries an Identity-Adversarial head (IAT): an identity
    classifier fed through a gradient-reversal layer, so the temporal representation
    is pushed to be identity-invariant. The downstream paper finding (FMAE-IAT, Norface)
    is that removing the subject/appearance shortcut is the lever that helps AU generalize
    in-the-wild. We use the source-video index as the identity proxy."""

    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2, n_ids=0, text_dim=0):
        super().__init__()
        self.rnn = nn.GRU(feat_dim, hidden, layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        d = 2 * hidden
        self.drop = nn.Dropout(dropout)
        self.va = nn.Linear(d, 2); self.expr = nn.Linear(d, 8); self.au = nn.Linear(d, 12)
        self.id_head = nn.Linear(d, n_ids) if n_ids else None
        self.text_proj = nn.Linear(d, text_dim) if text_dim else None

    def forward(self, x):
        h, _ = self.rnn(x); h = self.drop(h)
        va = torch.tanh(self.va(h))
        expr = self.expr(h)
        return {"va": va, "expr": expr, "au": self.au(h), "feat": h}


class TemporalMTLFusion(nn.Module):
    """Cross-task feature fusion: the EXPR head consumes the shared temporal feature
    plus the VA and AU branch outputs, so expression uses the complementary affect
    signal (the Progressive-Learning EXPR recipe). VA/AU heads stay independent."""

    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(feat_dim, hidden, layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        d = 2 * hidden
        self.drop = nn.Dropout(dropout)
        self.va = nn.Linear(d, 2)
        self.au = nn.Linear(d, 12)
        self.expr = nn.Sequential(nn.Linear(d + 2 + 12, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 8))

    def forward(self, x):
        h, _ = self.rnn(x); h = self.drop(h)
        va = torch.tanh(self.va(h)); au = self.au(h)
        expr = self.expr(torch.cat([h, va, torch.sigmoid(au)], dim=-1))
        return {"va": va, "expr": expr, "au": au}


class TemporalMTLCoupled(nn.Module):
    """Bidirectional AU<->EXPR coupling at the REPRESENTATION level (vs TemporalMTLFusion, which fuses
    AU/VA OUTPUTS into EXPR one-way -- flat/negative). After the BiGRU, EXPR and AU each get a branch
    feature; a gated cross-exchange lets each inform the other before its head, so the muscle-level AU
    signal shapes the rare-expression representation and vice versa (the FACS/EMFACS structure linking
    action units and expressions). Gates init at 0 -> starts decoupled (a plain 2-branch MTL) and only
    couples if training rewards it. VA stays independent."""

    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2, cdim=256):
        super().__init__()
        self.rnn = nn.GRU(feat_dim, hidden, layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        d = 2 * hidden
        self.drop = nn.Dropout(dropout)
        self.va = nn.Linear(d, 2)
        self.expr_pre = nn.Sequential(nn.Linear(d, cdim), nn.ReLU())
        self.au_pre = nn.Sequential(nn.Linear(d, cdim), nn.ReLU())
        self.ae = nn.Linear(cdim, cdim)
        self.ea = nn.Linear(cdim, cdim)
        self.g_ae = nn.Parameter(torch.zeros(1))
        self.g_ea = nn.Parameter(torch.zeros(1))
        self.expr = nn.Linear(cdim, 8)
        self.au = nn.Linear(cdim, 12)

    def forward(self, x):
        h, _ = self.rnn(x); h = self.drop(h)
        eh = self.expr_pre(h); ah = self.au_pre(h)
        ec = eh + self.g_ae * torch.tanh(self.ae(ah))
        ac = ah + self.g_ea * torch.tanh(self.ea(eh))
        return {"va": torch.tanh(self.va(h)), "expr": self.expr(ec), "au": self.au(ac), "feat": h}


class TemporalMTLTransformer(nn.Module):
    """Self-attention temporal head (vs the BiGRU): a different inductive bias -> decorrelated errors,
    for the near-peer ENSEMBLE lever (diversity across architectures, not just seeds). Same output dim
    d=2*hidden and per-frame heads as TemporalMTL so it drops into the ensemble/eval path unchanged.
    Sinusoidal positions (any length); padded frames (F all-zero) are masked out of attention. Eval
    runs whole videos, so attention is global over the clip (train sees length-`window` chunks)."""

    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2, heads=8):
        super().__init__()
        d = 2 * hidden
        self.proj = nn.Linear(feat_dim, d)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.va = nn.Linear(d, 2); self.expr = nn.Linear(d, 8); self.au = nn.Linear(d, 12)

    def _pe(self, T, d, device):
        pos = torch.arange(T, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-np.log(10000.0) / d))
        pe = torch.zeros(T, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        pad = (x.abs().sum(-1) == 0)
        h = self.proj(x) + self._pe(x.size(1), self.proj.out_features, x.device)
        h = self.tr(h, src_key_padding_mask=pad)
        h = self.drop(h)
        return {"va": torch.tanh(self.va(h)), "expr": self.expr(h), "au": self.au(h), "feat": h}


class _TCNBlock(nn.Module):
    """Residual dilated-conv block (non-causal / bidirectional: symmetric padding, offline whole-video)."""
    def __init__(self, d, k, dil, dropout):
        super().__init__()
        pad = (k - 1) * dil // 2
        self.net = nn.Sequential(
            nn.Conv1d(d, d, k, padding=pad, dilation=dil), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d, d, k, padding=pad, dilation=dil), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x): return x + self.net(x)


class TemporalMTLTCN(nn.Module):
    """Dilated temporal conv head (vs BiGRU): local + multi-scale, a decorrelated near-peer for the
    architecture-diversity ensemble. Small dilations give VA the local smoothing the transformer lacked;
    large dilations reach short AU onset-apex-offset dynamics. Dilations 1,2,4,8 x (kernel 3, 2 convs)
    -> ~60-frame receptive field, comparable to the window-64 BiGRU context. Same output dim + per-frame
    heads, so it drops into the ensemble/eval path unchanged."""
    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2, blocks=4, kernel=3):
        super().__init__()
        d = 2 * hidden
        self.proj = nn.Conv1d(feat_dim, d, 1)
        self.tcn = nn.ModuleList([_TCNBlock(d, kernel, 2 ** i, dropout) for i in range(blocks)])
        self.drop = nn.Dropout(dropout)
        self.va = nn.Linear(d, 2); self.expr = nn.Linear(d, 8); self.au = nn.Linear(d, 12)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for b in self.tcn:
            h = b(h)
        h = self.drop(h.transpose(1, 2))
        return {"va": torch.tanh(self.va(h)), "expr": self.expr(h), "au": self.au(h), "feat": h}


class TemporalMTLTCNms(nn.Module):
    """Multi-scale TCN with per-AU receptive-field selection (domain knowledge). Three parallel dilated
    branches with short/mid/long receptive fields (~5/~15/~50 frames) are concatenated per frame, so each
    task head -- and each of the 12 AU logits via its own row of the AU weight matrix -- learns its own
    mix of time-scales. Action units differ in temporal extent (a blink is transient, a smile sustained),
    so letting each AU pick its receptive field should sharpen AU. Extends single-scale --arch tcn."""
    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2, kernel=3):
        super().__init__()
        d = 2 * hidden
        self.proj = nn.Conv1d(feat_dim, d, 1)
        self.short = _TCNBlock(d, kernel, 1, dropout)
        self.mid = nn.Sequential(_TCNBlock(d, kernel, 1, dropout), _TCNBlock(d, kernel, 2, dropout))
        self.long = nn.Sequential(_TCNBlock(d, kernel, 4, dropout), _TCNBlock(d, kernel, 8, dropout))
        self.drop = nn.Dropout(dropout)
        dc = 3 * d
        self.va = nn.Linear(dc, 2); self.expr = nn.Linear(dc, 8); self.au = nn.Linear(dc, 12)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        h = torch.cat([self.short(h), self.mid(h), self.long(h)], dim=1)
        h = self.drop(h.transpose(1, 2))
        return {"va": torch.tanh(self.va(h)), "expr": self.expr(h), "au": self.au(h), "feat": h}


def _ssm_conv(x, k):
    """Causal long-convolution y[b,c,t] = sum_{s<=t} k[c,t-s] x[b,c,s] via FFT.
    x [B,d,T], k [d,K] -> y [B,d,T]. O(T log T); no custom CUDA kernel (sm_120-safe)."""
    T = x.shape[-1]; K = k.shape[-1]
    n = 1 << (T + K - 1).bit_length()
    Xf = torch.fft.rfft(x, n=n)
    Kf = torch.fft.rfft(k, n=n)
    return torch.fft.irfft(Xf * Kf.unsqueeze(0), n=n)[..., :T]


class _S4D(nn.Module):
    """Diagonal state-space mixer (S4D-real / linear-recurrent-unit form). Each of the d channels
    carries an independent N-state diagonal SSM h_t = a h_{t-1} + x_t, y_t = sum_n C_n h_t^{(n)};
    the impulse response is k[t] = sum_n C_n a_n^t, a sum-of-exponentials over learned retention
    poles a_n in (0,1). This is a GLOBAL, smooth, multi-timescale filter -- a different inductive
    bias from the TCN's local dilated taps and the BiGRU's gated recurrence -> decorrelated errors
    for the architecture-diversity ensemble. Bidirectional (separate forward/backward poles), applied
    as an FFT long-convolution so it is stable and fast on any sequence length."""

    def __init__(self, d, n_state=16, bidir=True, kmax=2048):
        super().__init__()
        self.d, self.n, self.bidir, self.kmax = d, n_state, bidir, kmax
        dirs = 2 if bidir else 1
        a0 = torch.linspace(0.05, 0.95, n_state)
        logit = torch.log(a0 / (1 - a0))
        self.pole = nn.Parameter(logit.repeat(dirs, d, 1).clone()
                                 + 0.01 * torch.randn(dirs, d, n_state))
        self.C = nn.Parameter(torch.randn(dirs, d, n_state) / math.sqrt(n_state))
        self.D = nn.Parameter(torch.ones(d))

    def _kernel(self, di, K, device):
        a = torch.sigmoid(self.pole[di])
        loga = torch.log(a)
        t = torch.arange(K, device=device, dtype=loga.dtype)
        apow = torch.exp(t[None, None, :] * loga[:, :, None])
        return (self.C[di][:, :, None] * apow).sum(1)

    def forward(self, x):
        xt = x.transpose(1, 2)
        K = min(x.shape[1], self.kmax)
        y = _ssm_conv(xt, self._kernel(0, K, x.device))
        if self.bidir:
            y = y + _ssm_conv(xt.flip(-1), self._kernel(1, K, x.device)).flip(-1)
        y = y + self.D[None, :, None] * xt
        return y.transpose(1, 2)


class _S4DBlock(nn.Module):
    """Canonical S4 block: a pre-norm state-space time-mixing sublayer (SSM -> GLU) and a pre-norm
    position-wise FFN channel-mixing sublayer, each with its own residual. The FFN is what mixes
    channels -- the diagonal SSM mixes only along time per channel -- so it is needed for the head
    to be a strength-parity peer of the BiGRU (which mixes channels at every step)."""
    def __init__(self, d, n_state, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.ssm = _S4D(d, n_state)
        self.glu = nn.Linear(d, 2 * d)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(F.glu(self.glu(self.ssm(self.n1(x))), dim=-1))
        return x + self.drop(self.ffn(self.n2(x)))


class TemporalMTLSSM(nn.Module):
    """State-space temporal head (vs BiGRU and TCN): a stack of diagonal S4D blocks, a third,
    decorrelated near-peer architecture for the diversity ensemble. Same output dim d=2*hidden and
    per-frame heads as the others, so it drops into the ensemble/eval path unchanged. Padded frames
    are all-zero and masked out by the loss (as for the GRU/TCN heads)."""
    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2, n_state=64):
        super().__init__()
        d = 2 * hidden
        self.proj = nn.Linear(feat_dim, d)
        self.blocks = nn.ModuleList([_S4DBlock(d, n_state, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.va = nn.Linear(d, 2); self.expr = nn.Linear(d, 8); self.au = nn.Linear(d, 12)

    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        h = self.drop(self.norm(h))
        return {"va": torch.tanh(self.va(h)), "expr": self.expr(h), "au": self.au(h), "feat": h}


class TemporalMTLLatent(nn.Module):
    """D2: partial-label shared affect-latent. A stochastic latent z (VAE encoder over the
    BiGRU feature) mediates ALL three task decoders. Trained with the masked multi-task loss
    (the mask marginalizes missing-task labels) + a small KL. Because every task is decoded
    from the SAME z, a frame labeled for only one task still shapes the shared affect
    representation the other tasks read from -- turning the 63%-partial-label structure into
    cross-task supervision instead of masked-off (discarded) gradient. Predicts all tasks at
    inference (z = posterior mean)."""

    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.2, zdim=32, va_skip=False, seq=False):
        super().__init__()
        self.rnn = nn.GRU(feat_dim, hidden, layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        d = 2 * hidden
        self.drop = nn.Dropout(dropout)
        self.enc = nn.Linear(d, 2 * zdim)
        self.zdim = zdim
        self.va_skip = va_skip
        self.va = nn.Linear(d if va_skip else zdim, 2)
        self.expr = nn.Linear(zdim, 8); self.au = nn.Linear(zdim, 12)
        self.seq = seq
        if seq:
            self.trans = nn.Sequential(nn.Linear(zdim, zdim), nn.ReLU(), nn.Linear(zdim, 2 * zdim))

    def forward(self, x):
        h, _ = self.rnn(x); h = self.drop(h)
        mu, logvar = self.enc(h).chunk(2, dim=-1)
        logvar = logvar.clamp(-6, 6)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        va = torch.tanh(self.va(h if self.va_skip else z))
        out = {"va": va, "expr": self.expr(z), "au": self.au(z),
               "feat": h, "mu": mu, "logvar": logvar}
        if self.seq:
            pm, plv = self.trans(z[:, :-1]).chunk(2, dim=-1)
            plv = plv.clamp(-6, 6)
            zpad = torch.zeros_like(mu[:, :1])
            pm = torch.cat([zpad, pm], dim=1)
            plv = torch.cat([zpad, plv], dim=1)
            out["klseq"] = (0.5 * (plv - logvar + (logvar.exp() + (mu - pm).pow(2))
                                   / plv.exp() - 1)).mean()
        return out


def group_videos(npz):
    """-> list of per-video dicts with frame-ordered arrays + original row indices."""
    videos = npz["videos"]; images = npz["images"]
    seqs = []
    for v in np.unique(videos):
        idx = np.where(videos == v)[0]
        keys = [(_frame_num(images[i]) if _frame_num(images[i]) is not None else j, j, i)
                for j, i in enumerate(idx)]
        order = [i for _, _, i in sorted(keys)]
        order = np.asarray(order)
        seqs.append(dict(rows=order,
                         F=npz["F"][order], valence=npz["valence"][order],
                         arousal=npz["arousal"][order], expr=npz["expr"][order],
                         au=npz["au"][order], m_va=npz["m_va"][order],
                         m_expr=npz["m_expr"][order], m_au=npz["m_au"][order]))
    return seqs


def make_windows(seqs, L, stride):
    """Chunk each video into (possibly zero-padded) length-L windows for batching.
    Each window records its source-video index ('vid') as an identity label for IAT."""
    W = []
    for vid, s in enumerate(seqs):
        T = len(s["F"])
        for start in range(0, max(1, T), stride):
            sl = slice(start, start + L)
            n = len(s["F"][sl])
            if n == 0:
                break
            pad = L - n
            def take(a, fill):
                a = a[sl]
                if pad:
                    a = np.concatenate([a, np.full((pad,) + a.shape[1:], fill, a.dtype)], 0)
                return a
            W.append(dict(
                vid=vid,
                F=take(s["F"], 0.0), valence=take(s["valence"], -5.0),
                arousal=take(s["arousal"], -5.0), expr=take(s["expr"], -1),
                au=take(s["au"], -1),
                m_va=np.concatenate([s["m_va"][sl], np.zeros(pad, bool)]),
                m_expr=np.concatenate([s["m_expr"][sl], np.zeros(pad, bool)]),
                m_au=np.concatenate([s["m_au"][sl], np.zeros(pad, bool)])))
            if start + L >= T:
                break
    return W


def main():
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="source run (for cfg + class balance)")
    ap.add_argument("--feats", nargs="+", default=None,
                    help="one or more feats dirs; multiple are CONCATENATED per frame "
                         "(cross-backbone temporal). Default <run>/feats")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--fusion", action="store_true",
                    help="EXPR head fuses VA+AU branch outputs (cross-task feature fusion)")
    ap.add_argument("--couple", action="store_true",
                    help="bidirectional AU<->EXPR representation coupling (gated cross-exchange)")
    ap.add_argument("--arch", default="gru", choices=["gru", "transformer", "tcn", "tcnms", "ssm"],
                    help="temporal head architecture (tcnms = multi-scale TCN, per-AU receptive fields; "
                         "ssm = diagonal state-space / S4D head)")
    ap.add_argument("--tasks", default="va,expr,au",
                    help="comma subset of va,expr,au to TRAIN (specialist = one task). "
                         "Other tasks' losses are masked off; best is selected by the "
                         "specialist's own metric.")
    ap.add_argument("--iat", action="store_true",
                    help="Identity-Adversarial Training: gradient-reversed video-identity head "
                         "pushes the temporal features to be identity-invariant (AU generalization).")
    ap.add_argument("--iat-lambda", type=float, default=1.0, help="GRL adversarial strength")
    ap.add_argument("--text-anchor", action="store_true",
                    help="EXPR text-anchor auxiliary loss: pull the projected feature toward the "
                         "frozen CLIP text embedding of its expression class (rare-class lever).")
    ap.add_argument("--ta-anchors", default="weights/expr_text_anchors.npy")
    ap.add_argument("--ta-alpha", type=float, default=0.5, help="text-anchor loss weight")
    ap.add_argument("--ta-temp", type=float, default=0.07, help="cosine softmax temperature")
    ap.add_argument("--latent", action="store_true",
                    help="D2: shared affect-latent (VAE bottleneck z mediates all 3 decoders) so "
                         "partial-label frames couple across tasks. Use with --tasks va,expr,au.")
    ap.add_argument("--zdim", type=int, default=32, help="latent dimension")
    ap.add_argument("--beta", type=float, default=0.01, help="KL weight (small = prediction-constrained)")
    ap.add_argument("--kl-warmup", type=int, default=0,
                    help="anneal KL weight linearly 0->beta over this many epochs (0 = no annealing)")
    ap.add_argument("--va-skip", action="store_true",
                    help="route the VA head off the BiGRU feature (skip the latent) to preserve VA "
                         "quality while EXPR/AU keep the shared-latent coupling")
    ap.add_argument("--z-smooth", type=float, default=0.0,
                    help="temporal-smoothness prior on the latent: penalize ||z_t - z_{t-1}||^2 "
                         "(affect evolves smoothly) by this weight")
    ap.add_argument("--seq-latent", action="store_true",
                    help="sequential latent: learned transition prior p(z_t|z_{t-1}) instead of the "
                         "static N(0,I) prior, so the latent models affect dynamics (VRNN-style)")
    ap.add_argument("--dump-val", default=None,
                    help="after training, dump the best model's val predictions+labels NPZ here "
                         "(for significance tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    keep = set(args.tasks.split(","))
    sel_metric = ("VA" if keep == {"va"} else "EXPR_macroF1" if keep == {"expr"}
                  else "AU_macroF1" if keep == {"au"} else "P_MTL")
    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run)
    feat_dirs = [Path(p) for p in args.feats] if args.feats else [run / "feats"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def load_split(split):
        parts = [np.load(d / f"{split}.npz", allow_pickle=True) for d in feat_dirs]
        base = parts[0]
        for p in parts[1:]:
            assert (p["images"] == base["images"]).all(), "feature files are misaligned"
        d = {"F": np.concatenate([p["F"] for p in parts], axis=1)}
        for k in ("valence", "arousal", "expr", "au", "m_va", "m_expr", "m_au",
                  "videos", "images"):
            d[k] = base[k]
        return d

    cfg = json.loads((run / "provenance.json").read_text())["config"]
    dcfg = cfg["data"]; tcfg = cfg["train"]; root = Path(dcfg["data_dir"])
    cb = class_balance(parse_annotations(root / dcfg.get("train_ann", "training_set_annotations.txt")))
    expr_w = _expr_class_weights(cb); au_w = _au_pos_weight(cb)

    tr = load_split("train")
    va = load_split("val")
    feat_dim = tr["F"].shape[1]
    tr_seqs, va_seqs = group_videos(tr), group_videos(va)
    windows = make_windows(tr_seqs, args.window, args.stride)
    print(f"feat_dim={feat_dim}  train videos={len(tr_seqs)} windows={len(windows)}  val videos={len(va_seqs)}")

    class WinDS(Dataset):
        def __len__(self): return len(windows)
        def __getitem__(self, i):
            w = windows[i]
            return (torch.from_numpy(w["F"]).float(),
                    {"valence": torch.from_numpy(w["valence"]).float(),
                     "arousal": torch.from_numpy(w["arousal"]).float(),
                     "expr": torch.from_numpy(w["expr"]).long(),
                     "au": torch.from_numpy(w["au"]).float()},
                    {"va": torch.from_numpy(w["m_va"]), "expr": torch.from_numpy(w["m_expr"]),
                     "au": torch.from_numpy(w["m_au"])},
                    w["vid"])

    from src.losses import MultiTaskLoss
    if (args.iat or args.text_anchor) and args.fusion:
        raise SystemExit("--iat/--text-anchor are for plain TemporalMTL, not --fusion")
    n_ids = len(tr_seqs) if args.iat else 0
    anchors = None
    text_dim = 0
    if args.text_anchor:
        anchors = torch.from_numpy(np.load(args.ta_anchors)).float().to(device)
        text_dim = anchors.shape[1]
    if args.latent:
        model = TemporalMTLLatent(feat_dim, args.hidden, args.layers, args.dropout,
                                  zdim=args.zdim, va_skip=args.va_skip, seq=args.seq_latent).to(device)
        print(f"[latent] shared affect-latent zdim={args.zdim}, beta={args.beta}"
              f"{', va-skip' if args.va_skip else ''}{', seq-prior' if args.seq_latent else ''}")
    elif args.iat or args.text_anchor:
        model = TemporalMTL(feat_dim, args.hidden, args.layers, args.dropout,
                            n_ids=n_ids, text_dim=text_dim).to(device)
    elif args.arch == "transformer":
        model = TemporalMTLTransformer(feat_dim, args.hidden, args.layers, args.dropout).to(device)
        print("[arch] transformer temporal head (self-attention)")
    elif args.arch == "tcn":
        model = TemporalMTLTCN(feat_dim, args.hidden, args.layers, args.dropout).to(device)
        print("[arch] TCN temporal head (dilated conv, RF~60)")
    elif args.arch == "tcnms":
        model = TemporalMTLTCNms(feat_dim, args.hidden, args.layers, args.dropout).to(device)
        print("[arch] multi-scale TCN temporal head (per-AU receptive fields)")
    elif args.arch == "ssm":
        model = TemporalMTLSSM(feat_dim, args.hidden, args.layers, args.dropout).to(device)
        print("[arch] diagonal state-space (S4D) temporal head")
    elif args.couple:
        model = TemporalMTLCoupled(feat_dim, args.hidden, args.layers, args.dropout).to(device)
        print("[couple] bidirectional AU<->EXPR representation coupling (gates init 0)")
    else:
        Net = TemporalMTLFusion if args.fusion else TemporalMTL
        model = Net(feat_dim, args.hidden, args.layers, args.dropout).to(device)
    if args.iat:
        print(f"[IAT] identity-adversarial head over {n_ids} videos, lambda={args.iat_lambda}")
    if args.text_anchor:
        print(f"[text-anchor] {text_dim}-d CLIP anchors, alpha={args.ta_alpha}, temp={args.ta_temp}")
    crit = MultiTaskLoss(expr_w, au_w, uncertainty_weighting=tcfg.get("uncertainty_weighting", True),
                         expr_loss=tcfg.get("expr_loss", "focal"),
                         focal_gamma=tcfg.get("focal_gamma", 2.0)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(crit.parameters()), lr=args.lr,
                            weight_decay=1e-4)
    ld = DataLoader(WinDS(), batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    def flat(d, keys):
        return {k: d[k].reshape(-1, *d[k].shape[2:]).to(device) for k in keys}

    @torch.no_grad()
    def evaluate():
        model.eval()
        N = len(va["F"])
        pv = np.zeros((N, 2)); pe = np.zeros((N, 8)); pa = np.zeros((N, 12))
        for s in va_seqs:
            x = torch.from_numpy(s["F"]).float().unsqueeze(0).to(device)
            o = model(x)
            pv[s["rows"]] = o["va"][0].cpu().numpy()
            pe[s["rows"]] = torch.softmax(o["expr"][0], 1).cpu().numpy()
            pa[s["rows"]] = torch.sigmoid(o["au"][0]).cpu().numpy()
        m = all_metrics(va["valence"], pv[:, 0], va["arousal"], pv[:, 1],
                        va["expr"], pe.argmax(1), va["au"].astype(int), (pa >= 0.5).astype(int))
        return m, pv, pe, pa

    import torch.nn.functional as F
    best = {"P_MTL": -1}
    for ep in range(args.epochs):
        model.train()
        for x, t, m, vid in ld:
            x = x.to(device)
            o = model(x)
            of = {k: o[k].reshape(-1, o[k].shape[-1]) for k in ("va", "expr", "au")}
            tf = flat(t, ["valence", "arousal", "expr", "au"])
            mf = {k: m[k].reshape(-1).to(device) for k in m}
            for tk in ("va", "expr", "au"):
                if tk not in keep:
                    mf[tk] = torch.zeros_like(mf[tk])
            loss, _ = crit(of, tf, mf)
            if args.latent:
                mu, logvar = o["mu"], o["logvar"]
                if args.seq_latent:
                    kl = o["klseq"]
                else:
                    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                beta = args.beta * (min(1.0, (ep + 1) / args.kl_warmup) if args.kl_warmup else 1.0)
                loss = loss + beta * kl
                if args.z_smooth:
                    loss = loss + args.z_smooth * (mu[:, 1:] - mu[:, :-1]).pow(2).mean()
            if args.iat:
                pooled = o["feat"].mean(1)
                id_logits = model.id_head(grad_reverse(pooled, args.iat_lambda))
                loss = loss + F.cross_entropy(id_logits, vid.to(device))
            if args.text_anchor:
                feat = o["feat"].reshape(-1, o["feat"].shape[-1])
                proj = F.normalize(model.text_proj(feat), dim=-1)
                txt_logits = proj @ F.normalize(anchors, dim=-1).t() / args.ta_temp
                ye = tf["expr"]; me = mf["expr"].bool()
                if me.any():
                    loss = loss + args.ta_alpha * F.cross_entropy(txt_logits[me], ye[me])
            opt.zero_grad(); loss.backward(); opt.step()
        mt, _, _, _ = evaluate()
        tag = ""
        if mt[sel_metric] > best.get(sel_metric, -1):
            best = mt; tag = "  <-- best"
            torch.save(model.state_dict(), out / "best.pt")
        print(f"ep{ep:02d} P_MTL={mt['P_MTL']:.4f} VA={mt['VA']:.4f} "
              f"EXPR={mt['EXPR_macroF1']:.4f} AU={mt['AU_macroF1']:.4f}{tag}")
    (out / "metrics.json").write_text(json.dumps(best, indent=2))
    print(f"\nBEST temporal: P_MTL={best['P_MTL']:.4f} VA={best['VA']:.4f} "
          f"EXPR={best['EXPR_macroF1']:.4f} AU={best['AU_macroF1']:.4f}")
    if args.dump_val:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device), strict=False)
        _, pv, pe, pa = evaluate()
        np.savez(args.dump_val, pv=pv, pe=pe, pa=pa,
                 valence=va["valence"], arousal=va["arousal"], expr=va["expr"],
                 au=va["au"].astype(int), m_va=va["m_va"], m_expr=va["m_expr"],
                 m_au=va["m_au"], videos=va["videos"])
        print(f"  dumped val preds -> {args.dump_val}")


if __name__ == "__main__":
    main()
