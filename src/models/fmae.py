from __future__ import annotations

from pathlib import Path

import torch

_DROP = ("decoder", "mask_token", "id_head", "id_", "head.", "fc_norm")
_STRIP = ("encoder.", "base_encoder.", "module.", "model.")


def build_fmae(weights_path: str | None = None, model_name: str = "vit_base_patch16_224"):
    """Return (timm ViT-B/16 feature extractor, feat_dim=768). Loads the FMAE encoder
    weights if a path is given; raises loudly if the match looks wrong."""
    import timm
    model = timm.create_model(model_name, pretrained=False, num_classes=0,
                              global_pool="token", img_size=224)
    if weights_path:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(
                f"FMAE weights not found at {p}. Fetch from HuggingFace:\n"
                "  forever208/FMAE-IAT : FMAE_ViT_base.pth")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt
        if isinstance(ckpt, dict):
            for k in ("model", "state_dict", "model_state_dict"):
                if k in ckpt:
                    sd = ckpt[k]
                    break
        enc = {}
        for k, v in sd.items():
            if any(k.startswith(d) or d in k for d in _DROP):
                continue
            kk = k
            for pre in _STRIP:
                if kk.startswith(pre):
                    kk = kk[len(pre):]
            enc[kk] = v
        missing, unexpected = model.load_state_dict(enc, strict=False)
        loaded = len(enc) - len(unexpected)
        print(f"[FMAE] loaded {loaded}/{len(enc)} encoder tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if loaded < 100:
            raise RuntimeError(
                f"FMAE load looks wrong (only {loaded} matched). Inspect checkpoint keys "
                "(top-level key / prefix) and adjust _DROP/_STRIP in src/models/fmae.py.")
    return model, model.num_features
