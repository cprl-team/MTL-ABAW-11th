from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import (build_torch_dataset, class_balance,
                      parse_annotations, subset_by_videos)
from src.metrics import all_metrics
from src.utils.provenance import write_provenance
from src.utils.seeding import set_all_seeds
from src.utils.checkpointing import BestTracker, sha256_of
from checks.leakage_check import assert_videos_disjoint


def _fingerprint(*files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def _env(device) -> dict:
    """Record framework + hardware for the headline numbers (reproducibility)."""
    import torch
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)(),
            "device": device}
    if device == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _expr_class_weights(cb, n_expr=8):
    counts = np.array([cb["expr_counts"].get(c, 0) for c in range(n_expr)], dtype=np.float64)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (n_expr * counts)
    return (w / w.mean()).tolist()


def _au_pos_weight(cb, n_au=12):
    pos = np.array(cb["au_pos"], dtype=np.float64)
    tot = float(cb["au_total"])
    neg = np.clip(tot - pos, 1, None)
    pos = np.clip(pos, 1, None)
    return (neg / pos).tolist()


def calibrate_au_thresholds(au_true, au_prob, grid=None):
    """Per-AU decision threshold that maximizes that AU's F1 on the given set
    (AU_IGNORE rows dropped per column). Returns (thresholds[12], macro_f1)."""
    from src.metrics.mtl import AU_IGNORE, N_AU, _binary_f1
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    thr = np.full(N_AU, 0.5)
    f1s = []
    for k in range(N_AU):
        col_t, col_p = au_true[:, k], au_prob[:, k]
        msk = col_t != AU_IGNORE
        t, p = col_t[msk], col_p[msk]
        best_f1, best_th = 0.0, 0.5
        for th in grid:
            pred = (p >= th).astype(int)
            tp = int(((pred == 1) & (t == 1)).sum())
            fp = int(((pred == 1) & (t == 0)).sum())
            fn = int(((pred == 0) & (t == 1)).sum())
            f1 = _binary_f1(tp, fp, fn)
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)
        thr[k] = best_th
        f1s.append(best_f1)
    return thr, float(np.mean(f1s))


def evaluate(model, loader, device, au_threshold=0.5, calibrate=False):
    import torch
    model.eval()
    V_t, V_p, A_t, A_p, E_t, E_p, U_t, U_prob = ([] for _ in range(8))
    with torch.no_grad():
        for x, t, m in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            va = out["va"].float().cpu().numpy()
            V_p.append(va[:, 0]); A_p.append(va[:, 1])
            E_p.append(out["expr"].argmax(1).cpu().numpy())
            U_prob.append(torch.sigmoid(out["au"]).float().cpu().numpy())
            V_t.append(t["valence"].numpy()); A_t.append(t["arousal"].numpy())
            E_t.append(t["expr"].numpy()); U_t.append(t["au"].numpy().astype(int))
    cat = lambda L: np.concatenate(L, 0)
    au_true, au_prob = cat(U_t), cat(U_prob)
    metrics = all_metrics(cat(V_t), cat(V_p), cat(A_t), cat(A_p),
                          cat(E_t), cat(E_p), au_true, (au_prob >= au_threshold).astype(int))
    if calibrate:
        thr, au_cal = calibrate_au_thresholds(au_true, au_prob)
        metrics["AU_macroF1_calibrated"] = au_cal
        metrics["AU_thresholds"] = thr.tolist()
        metrics["P_MTL_calibrated"] = float(metrics["VA"] + metrics["EXPR_macroF1"] + au_cal)
    return metrics


def main():
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/mtl")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (for multi-seed runs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    set_all_seeds(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    root = Path(dcfg["data_dir"])
    if dcfg.get("dataset") == "affectnet":
        from src.data.affectnet import parse_affectnet
        manifest = root / dcfg.get("manifest", "aligned_112_manifest.csv")
        tr_f = va_f = manifest
        image_root = root
        au_map = dcfg.get("faba_au_map")
        tr_ann = parse_affectnet(manifest, "train", root, au_map=au_map)
        va_ann = parse_affectnet(manifest, "val", root, au_map=au_map)
    elif dcfg.get("dataset") == "rafdb":
        from src.data.rafdb import parse_rafdb
        labels = root / dcfg.get("labels", "basic/EmoLabel/list_patition_label.txt")
        tr_f = va_f = labels
        image_root = root
        tr_ann = parse_rafdb(labels, "train", root)
        va_ann = parse_rafdb(labels, "test", root)
    elif dcfg.get("dataset") == "emotionet":
        from src.data.emotionet import parse_emotionet
        csv = root / dcfg.get("csv", "EmotioNet_FACS_aws_2020_24600.csv")
        tr_f = va_f = csv
        image_root = root
        tr_ann = parse_emotionet(csv, "train", root)
        va_ann = parse_emotionet(csv, "val", root)
    elif dcfg.get("dataset") == "unbc":
        from src.data.unbc import parse_unbc
        manifest = root / dcfg.get("manifest", "manifest.json")
        tr_f = va_f = manifest
        image_root = root
        tr_ann = parse_unbc(manifest, "train", root)
        va_ann = parse_unbc(manifest, "val", root)
    else:
        tr_f = root / dcfg.get("train_ann", "training_set_annotations.txt")
        va_f = root / dcfg.get("val_ann", "validation_set_annotations.txt")
        image_root = root / dcfg.get("image_subdir", "cropped_aligned")
        tr_ann = parse_annotations(tr_f)
        va_ann = parse_annotations(va_f)
        assert_videos_disjoint(tr_ann.videos, va_ann.videos)

    lim = dcfg.get("limit_train_videos")
    if lim:
        keep_tr = set(sorted(set(tr_ann.videos))[:lim])
        keep_va = set(sorted(set(va_ann.videos))[:max(1, lim // 4)])
        tr_ann = subset_by_videos(tr_ann, keep_tr)
        va_ann = subset_by_videos(va_ann, keep_va)

    cb = class_balance(tr_ann)
    fp = _fingerprint(tr_f, va_f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prov = {**cfg, "dataset_fingerprint": fp, "env": _env(device),
            "train_summary": tr_ann.summary(), "val_summary": va_ann.summary()}
    write_provenance(out, prov, seed)

    img_size = dcfg.get("img_size", 112)
    aug = dcfg.get("augment", "standard")
    tr_ds = build_torch_dataset(tr_ann, image_root, train=True, img_size=img_size, aug=aug)
    va_ds = build_torch_dataset(va_ann, image_root, train=False, img_size=img_size)

    tcfg = cfg.get("train", {})
    bs = tcfg.get("batch_size", 128)
    workers = tcfg.get("num_workers", 4)

    g = torch.Generator().manual_seed(seed)

    def _worker_init(wid):
        np.random.seed(seed + wid)

    sampler = None
    if tcfg.get("sampler") == "balanced_expr":
        from src.data import expr_sample_weights
        w = torch.as_tensor(expr_sample_weights(tr_ann), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(w), replacement=True, generator=g)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
                       num_workers=workers, pin_memory=(device == "cuda"), drop_last=True,
                       generator=g, worker_init_fn=_worker_init)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=workers,
                       pin_memory=(device == "cuda"), worker_init_fn=_worker_init)

    from src.models import build_mtl_model
    from src.losses import MultiTaskLoss
    model = build_mtl_model(cfg["model"]).to(device)
    init_ckpt = cfg["model"].get("init_backbone_from")
    if init_ckpt:
        import torch as _torch
        _sd = _torch.load(init_ckpt, map_location=device, weights_only=False)
        if isinstance(_sd, dict) and "state_dict" in _sd:
            _sd = _sd["state_dict"]
        _bk = {k[len("backbone."):]: v for k, v in _sd.items()
               if k.startswith("backbone.")}
        if not _bk:
            raise ValueError(f"init_backbone_from={init_ckpt}: no 'backbone.*' tensors")
        _bbk = set(model.backbone.state_dict().keys())
        def _remap(k):
            if k in _bbk:
                return k
            c1 = f"base_model.model.{k}"
            if c1 in _bbk:
                return c1
            pre, _, leaf = k.rpartition(".")
            c2 = f"base_model.model.{pre}.base_layer.{leaf}"
            return c2 if c2 in _bbk else k
        _bk = {_remap(k): v for k, v in _bk.items()}
        _miss, _unexp = model.backbone.load_state_dict(_bk, strict=False)
        print(f"  [init] backbone <- {init_ckpt}: loaded {len(_bk)} tensors "
              f"({len(_miss)} missing, {len(_unexp)} unexpected)")
    distill_w = tcfg.get("distill_weight", 0.0)
    teacher_bb = None
    _sfeat: dict = {}
    if distill_w > 0:
        import copy as _copy
        teacher_bb = _copy.deepcopy(model.backbone).to(device).eval()
        for _p in teacher_bb.parameters():
            _p.requires_grad_(False)
        model.backbone.register_forward_hook(lambda mod, inp, out: _sfeat.__setitem__("f", out))
        print(f"  [distill] feature distillation to init backbone, weight={distill_w}")
    expr_w = _expr_class_weights(cb) if tcfg.get("class_weighting", True) else None
    au_w = _au_pos_weight(cb) if tcfg.get("class_weighting", True) else None
    expr_counts = np.array([cb["expr_counts"].get(c, 0) for c in range(8)], dtype=np.float64)
    expr_prior = (expr_counts / expr_counts.sum()).tolist()
    criterion = MultiTaskLoss(expr_w, au_w,
                              uncertainty_weighting=tcfg.get("uncertainty_weighting", True),
                              expr_loss=tcfg.get("expr_loss", "ce"),
                              focal_gamma=tcfg.get("focal_gamma", 2.0),
                              expr_prior=expr_prior,
                              logit_adjust_tau=tcfg.get("logit_adjust_tau", 0.0),
                              au_loss=tcfg.get("au_loss", "bce")).to(device)

    from src.engine.finetune import (FineTuneCfg, ModelEMA, apply_freeze,
                                     build_param_groups, freeze_all_backbone)
    ft = FineTuneCfg.from_cfg(tcfg.get("finetune", {}))
    base_lr = tcfg.get("lr", 1e-4)
    if ft.warmup_epochs > 0:
        freeze_all_backbone(model, True)
    else:
        apply_freeze(model, ft)
    if cfg["model"].get("lora"):
        from src.models.lora import set_lora_trainable
        _f, _t = set_lora_trainable(model.backbone)
        print(f"  [lora] base frozen ({_f} tensors), adapters trainable ({_t} tensors)")
    param_groups = build_param_groups(model, base_lr, ft)
    param_groups.append({"params": list(criterion.parameters()), "lr": base_lr})
    opt = torch.optim.AdamW(param_groups, lr=base_lr,
                            weight_decay=tcfg.get("weight_decay", 1e-4))
    ema = ModelEMA(model, ft.ema_decay) if ft.ema else None
    epochs = tcfg.get("epochs", 10)
    max_steps = tcfg.get("max_steps")
    steps_per_epoch = max_steps if max_steps else len(tr_ld)
    sched = None
    if tcfg.get("lr_schedule", "cosine") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs * steps_per_epoch))
    grad_surgery = tcfg.get("grad_surgery", "none")
    if grad_surgery not in ("none", "pcgrad"):
        raise ValueError(f"grad_surgery must be 'none' or 'pcgrad', got '{grad_surgery}'")
    use_amp = (device == "cuda") and tcfg.get("amp", True) and grad_surgery == "none"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    au_thr = tcfg.get("au_threshold", 0.5)
    mixup_a = tcfg.get("mixup_alpha", 0.0)
    if mixup_a > 0:
        print(f"  [mixup] alpha={mixup_a}")
    if grad_surgery == "pcgrad":
        import random as _random
        from src.engine.mtl_opt import pcgrad_backward
        pc_rng = _random.Random(seed)

    tracker = BestTracker(higher_is_better=True)
    best = {"P_MTL": -1.0}
    history = []
    for ep in range(epochs):
        if ft.warmup_epochs > 0 and ep == ft.warmup_epochs:
            apply_freeze(model, ft)
            print(f"  [finetune] unfroze backbone at epoch {ep} (warmup done)")
        model.train()
        t0 = time.time()
        for step, (x, t, m) in enumerate(tr_ld):
            x = x.to(device, non_blocking=True)
            targets = {k: v.to(device) for k, v in t.items()}
            masks = {k: v.to(device) for k, v in m.items()}
            opt.zero_grad(set_to_none=True)
            if grad_surgery == "pcgrad":
                outputs = model(x)
                ltasks = criterion.task_losses(outputs, targets, masks)
                shared = [p for p in model.backbone.parameters() if p.requires_grad]
                heads = [p for n, p in model.named_parameters()
                         if not n.startswith("backbone.") and p.requires_grad]
                pcgrad_backward(ltasks, shared, heads, pc_rng)
                opt.step()
            else:
                mix_idx, lam = None, 1.0
                if mixup_a > 0:
                    lam = float(np.random.beta(mixup_a, mixup_a))
                    mix_idx = torch.randperm(x.size(0), device=device)
                    x = lam * x + (1.0 - lam) * x[mix_idx]
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                    enabled=use_amp):
                    outputs = model(x)
                    loss, parts = criterion(outputs, targets, masks)
                    if mix_idx is not None:
                        t2 = {k: v[mix_idx] for k, v in targets.items()}
                        m2 = {k: v[mix_idx] for k, v in masks.items()}
                        loss = lam * loss + (1.0 - lam) * criterion(outputs, t2, m2)[0]
                    if "au_region" in outputs:
                        rt = targets["au"]; am = masks["au"].float()
                        am = am.unsqueeze(1) if am.dim() == 1 else am
                        rv = (rt >= 0).float() * am
                        if rv.sum() > 0:
                            rl = torch.nn.functional.binary_cross_entropy_with_logits(
                                outputs["au_region"], rt.clamp(0, 1).float(), reduction="none")
                            loss = loss + tcfg.get("region_au_weight", 0.5) * (rl * rv).sum() / rv.sum()
                    if teacher_bb is not None and _sfeat.get("f") is not None:
                        with torch.no_grad():
                            tfeat = teacher_bb(x)
                        loss = loss + distill_w * (1 - torch.nn.functional.cosine_similarity(
                            _sfeat["f"], tfeat, dim=-1)).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            if ema is not None:
                ema.update(model)
            if sched is not None:
                sched.step()
            if max_steps and step + 1 >= max_steps:
                break
        eval_sd = None
        if ema is not None:
            eval_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict())
        val = evaluate(model, va_ld, device, au_thr,
                       calibrate=tcfg.get("calibrate_au", True))
        val["epoch"] = ep
        val["sec"] = round(time.time() - t0, 1)
        history.append(val)
        print(f"epoch {ep}: P_MTL={val['P_MTL']:.4f} "
              f"VA={val['VA']:.4f} EXPR={val['EXPR_macroF1']:.4f} "
              f"AU={val['AU_macroF1']:.4f} ({val['sec']}s)")
        if tracker.update(val["P_MTL"]):
            ckpt = out / "best.pt"
            torch.save(model.state_dict(), ckpt)
            val["checkpoint"] = str(ckpt)
            val["checkpoint_sha256"] = sha256_of(ckpt)
            val["seed"] = seed
            best = val
            (out / "metrics.json").write_text(json.dumps(val, indent=2))
        if eval_sd is not None:
            model.load_state_dict(eval_sd)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest P_MTL={best['P_MTL']:.4f} (epoch {best.get('epoch')}) -> {out}/metrics.json")


if __name__ == "__main__":
    main()
