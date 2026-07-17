from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import build_torch_dataset, parse_annotations


def collect_feats(model, ann, image_root, device, img_size, bs=256, workers=6, flip=False, regions=0):
    import torch
    from torch.utils.data import DataLoader
    ds = build_torch_dataset(ann, image_root, train=False, img_size=img_size)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=(device == "cuda"))
    F, V, A, E, AU, MV, ME, MAU = ([] for _ in range(8))
    model.eval()
    with torch.no_grad():
        for x, t, m in ld:
            x = x.to(device, non_blocking=True)
            if flip:
                x = torch.flip(x, dims=[3])
            feat = model.backbone(x)
            if regions:
                tok = model.backbone.forward_features(x)
                p = tok[:, 1:]
                g = int(round(p.shape[1] ** 0.5)); D = p.shape[2]
                pg = p.reshape(p.shape[0], g, g, D).permute(0, 3, 1, 2)
                reg = torch.nn.functional.adaptive_avg_pool2d(pg, regions)
                reg = reg.permute(0, 2, 3, 1).reshape(p.shape[0], -1)
                feat = torch.cat([feat, reg], dim=1)
            F.append(feat.float().cpu().numpy())
            V.append(t["valence"].numpy()); A.append(t["arousal"].numpy())
            E.append(t["expr"].numpy()); AU.append(t["au"].numpy().astype(np.int64))
            MV.append(m["va"].numpy()); ME.append(m["expr"].numpy()); MAU.append(m["au"].numpy())
    cat = lambda L: np.concatenate(L, 0)
    return dict(F=cat(F), valence=cat(V), arousal=cat(A), expr=cat(E), au=cat(AU),
                m_va=cat(MV), m_expr=cat(ME), m_au=cat(MAU),
                videos=np.asarray(ann.videos), images=np.asarray(ann.images))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="output dir (default <run>/feats)")
    ap.add_argument("--flip", action="store_true",
                    help="horizontal-flip images before the backbone (flip-TTA; additive, off by default)")
    ap.add_argument("--split", default=None, choices=["train", "val"],
                    help="extract only this split (default both)")
    ap.add_argument("--regions", type=int, default=0,
                    help="R: concat RxR region-pooled patch features onto CLS (AU spatial locality; 0=off)")
    ap.add_argument("--data-config", default=None,
                    help="YAML whose 'data' block to use instead of the run's own (e.g. extract Aff-Wild2 "
                         "features from a backbone that was fine-tuned on a different dataset).")
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out) if args.out else run / "feats"
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run / "provenance.json").read_text())["config"]
    dcfg = cfg["data"]
    if args.data_config:
        import yaml
        dcfg = yaml.safe_load(Path(args.data_config).read_text())["data"]
        print(f"[extract] data from {args.data_config}: {dcfg.get('data_dir')}")
    root = Path(dcfg["data_dir"])
    image_root = root / dcfg.get("image_subdir", "cropped_aligned")
    img_size = dcfg.get("img_size", 112)

    from src.models import build_mtl_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mtl_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))

    for split, ann_key, default in [("train", "train_ann", "training_set_annotations.txt"),
                                    ("val", "val_ann", "validation_set_annotations.txt")]:
        if args.split and split != args.split:
            continue
        ann = parse_annotations(root / dcfg.get(ann_key, default))
        d = collect_feats(model, ann, image_root, device, img_size, flip=args.flip, regions=args.regions)
        np.savez(out / f"{split}.npz", **d)
        print(f"wrote {out}/{split}.npz  frames={len(d['F'])}  feat_dim={d['F'].shape[1]}")


if __name__ == "__main__":
    main()
