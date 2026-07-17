from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.data import parse_annotations, class_balance
from src.data.abaw import N_AU
from src.engine.train import _au_pos_weight, _expr_class_weights
from src.engine.train_temporal import TemporalMTLLatent, TemporalMTL, _frame_num
from src.losses import MultiTaskLoss
from src.metrics import all_metrics
from src.models import build_mtl_model
from src.utils.seeding import set_all_seeds

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _video_order(ann):
    """-> list of per-video ordered row-index arrays (frame order within each video)."""
    seqs = []
    for v in np.unique(ann.videos):
        idx = np.where(ann.videos == v)[0]
        keys = [(_frame_num(ann.images[i]) if _frame_num(ann.images[i]) is not None else j, j, i)
                for j, i in enumerate(idx)]
        seqs.append(np.asarray([i for _, _, i in sorted(keys)]))
    return seqs


def _windows(seqs, L, stride):
    W = []
    for s in seqs:
        T = len(s)
        for start in range(0, max(1, T), stride):
            W.append(s[start:start + L])
            if start + L >= T:
                break
    return W


def _make_tf(train, img_size, aug):
    from torchvision import transforms
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2)]
    ops += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


def main():
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_all_seeds(seed)
    tcfg = cfg["train"]; dcfg = cfg["data"]; mcfg = cfg["model"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = dcfg.get("img_size", 224)
    root = Path(dcfg["data_dir"])
    image_root = root / dcfg.get("image_subdir", "cropped_aligned")

    tr_ann = parse_annotations(root / dcfg.get("train_ann", "training_set_annotations.txt"))
    va_ann = parse_annotations(root / dcfg.get("val_ann", "validation_set_annotations.txt"))
    cb = class_balance(tr_ann)
    expr_w = _expr_class_weights(cb) if tcfg.get("class_weighting", True) else None
    au_w = _au_pos_weight(cb) if tcfg.get("class_weighting", True) else None

    holder = build_mtl_model(mcfg).to(device)
    backbone, feat_dim = holder.backbone, holder.feat_dim
    init_ckpt = mcfg.get("init_backbone_from")
    if init_ckpt:
        sd = torch.load(init_ckpt, map_location=device, weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        bk = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        bbk = set(backbone.state_dict().keys())
        def remap(k):
            if k in bbk:
                return k
            c1 = f"base_model.model.{k}"
            if c1 in bbk:
                return c1
            pre, _, leaf = k.rpartition(".")
            c2 = f"base_model.model.{pre}.base_layer.{leaf}"
            return c2 if c2 in bbk else k
        miss, unexp = backbone.load_state_dict({remap(k): v for k, v in bk.items()}, strict=False)
        print(f"  [init] backbone <- {init_ckpt} ({len(bk)} tensors, {len(miss)} missing)")
    if mcfg.get("lora"):
        from src.models.lora import set_lora_trainable
        f, t = set_lora_trainable(backbone)
        print(f"  [lora] base frozen ({f}), adapters trainable ({t})")
    for obj in (backbone, getattr(backbone, "base_model", None),
                getattr(getattr(backbone, "base_model", None), "model", None)):
        if obj is not None and hasattr(obj, "set_grad_checkpointing"):
            try:
                obj.set_grad_checkpointing(True); print("  [e2e] grad checkpointing on"); break
            except Exception:
                pass

    latent = tcfg.get("latent", True)
    head = (TemporalMTLLatent(feat_dim, tcfg.get("hidden", 256), tcfg.get("layers", 2),
                              tcfg.get("dropout", 0.2), zdim=tcfg.get("zdim", 96))
            if latent else TemporalMTL(feat_dim, tcfg.get("hidden", 256), tcfg.get("layers", 2),
                                       tcfg.get("dropout", 0.2))).to(device)
    crit = MultiTaskLoss(expr_w, au_w, uncertainty_weighting=tcfg.get("uncertainty_weighting", True),
                         expr_loss=tcfg.get("expr_loss", "focal"),
                         focal_gamma=tcfg.get("focal_gamma", 2.0)).to(device)
    params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters()) \
        + list(crit.parameters())
    opt = torch.optim.AdamW(params, lr=tcfg.get("lr", 5e-4), weight_decay=tcfg.get("weight_decay", 1e-4))
    n_train = sum(p.numel() for p in params if p.requires_grad)
    print(f"  [e2e] trainable params: {n_train:,}  (LoRA adapters + temporal head)")

    L = tcfg.get("window", 32); stride = tcfg.get("stride", 32)
    tf_tr = _make_tf(True, img_size, dcfg.get("augment", "standard"))
    tf_va = _make_tf(False, img_size, None)
    tr_windows = _windows(_video_order(tr_ann), L, stride)
    va_seqs = _video_order(va_ann)

    def load_imgs(rows, ann, tf):
        return torch.stack([tf(Image.open(image_root / ann.images[r]).convert("RGB")) for r in rows])

    class WinDS(Dataset):
        def __len__(self): return len(tr_windows)
        def __getitem__(self, w):
            rows = tr_windows[w]; n = len(rows)
            x = load_imgs(rows, tr_ann, tf_tr)
            if n < L:
                x = torch.cat([x, torch.zeros(L - n, *x.shape[1:])], 0)
            def pad(a, fill, dt):
                a = torch.as_tensor(a, dtype=dt)
                return a if n == L else torch.cat([a, torch.full((L - n, *a.shape[1:]), fill, dtype=dt)], 0)
            t = {"valence": pad(tr_ann.valence[rows], -5., torch.float32),
                 "arousal": pad(tr_ann.arousal[rows], -5., torch.float32),
                 "expr": pad(tr_ann.expr[rows], -1, torch.long),
                 "au": pad(tr_ann.au[rows], -1, torch.float32)}
            m = {"va": pad(tr_ann.va_mask[rows], 0, torch.bool),
                 "expr": pad(tr_ann.expr_mask[rows], 0, torch.bool),
                 "au": pad(tr_ann.au_mask[rows], 0, torch.bool)}
            return x, t, m

    ld = DataLoader(WinDS(), batch_size=tcfg.get("batch_size", 2), shuffle=True,
                    num_workers=tcfg.get("num_workers", 4), drop_last=True, pin_memory=True)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    beta = tcfg.get("beta", 0.05); warm = tcfg.get("kl_warmup", 8)
    eval_bs = tcfg.get("eval_bs", 64)

    max_steps = tcfg.get("max_steps")
    eval_videos = tcfg.get("eval_videos")

    @torch.no_grad()
    def evaluate():
        backbone.eval(); head.eval()
        N = len(va_ann); pv = np.zeros((N, 2)); pe = np.zeros((N, 8)); pa = np.zeros((N, 12))
        seqs = va_seqs[:eval_videos] if eval_videos else va_seqs
        for s in seqs:
            feats = []
            for i in range(0, len(s), eval_bs):
                x = load_imgs(s[i:i + eval_bs], va_ann, tf_va).to(device)
                with torch.autocast("cuda", enabled=use_amp):
                    feats.append(backbone(x).float())
            f = torch.cat(feats, 0).unsqueeze(0)
            o = head(f)
            pv[s] = o["va"][0].cpu().numpy()
            pe[s] = torch.softmax(o["expr"][0], 1).cpu().numpy()
            pa[s] = torch.sigmoid(o["au"][0]).cpu().numpy()
        return all_metrics(va_ann.valence, pv[:, 0], va_ann.arousal, pv[:, 1],
                           va_ann.expr, pe.argmax(1), va_ann.au.astype(int), (pa >= 0.5).astype(int))

    import torch.nn.functional as F
    accum = tcfg.get("grad_accum", 1)
    best = {"P_MTL": -1.0}; history = []
    for ep in range(tcfg.get("epochs", 4)):
        backbone.train(); head.train(); t0 = time.time()
        kl_w = beta * min(1.0, (ep + 1) / max(1, warm))
        opt.zero_grad(set_to_none=True)
        for step, (x, t, m) in enumerate(ld):
            if max_steps and step >= max_steps:
                break
            B, T = x.shape[:2]
            x = x.to(device, non_blocking=True).view(B * T, *x.shape[2:])
            with torch.autocast("cuda", enabled=use_amp):
                feat = backbone(x).view(B, T, -1)
                o = head(feat)
                of = {k: o[k].reshape(-1, o[k].shape[-1]) for k in ("va", "expr", "au")}
                tf_ = {k: v.to(device).reshape(-1, *v.shape[2:]) for k, v in t.items()}
                mf = {k: v.to(device).reshape(-1) for k, v in m.items()}
                loss, _ = crit(of, tf_, mf)
                if latent:
                    mu, lv = o["mu"], o["logvar"]
                    loss = loss + kl_w * (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean()
                loss = loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        val = evaluate(); val["epoch"] = ep; val["sec"] = round(time.time() - t0, 1)
        history.append(val)
        print(f"epoch {ep}: P_MTL={val['P_MTL']:.4f} VA={val['VA']:.4f} "
              f"EXPR={val['EXPR_macroF1']:.4f} AU={val['AU_macroF1']:.4f} ({val['sec']}s)"
              + ("  <-- best" if val["P_MTL"] > best["P_MTL"] else ""))
        if val["P_MTL"] > best["P_MTL"]:
            best = val
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()}, out / "best.pt")
            (out / "metrics.json").write_text(json.dumps(val, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest P_MTL={best['P_MTL']:.4f} (epoch {best.get('epoch')}) -> {out}/metrics.json")


if __name__ == "__main__":
    main()
