from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VA_IGNORE = -5.0
EXPR_IGNORE = -1
AU_IGNORE = -1
N_AU = 12
N_EXPR = 8


@dataclass
class AbawAnnotations:
    """Parsed annotations. All arrays are row-aligned (one row per frame)."""
    images: np.ndarray
    videos: np.ndarray
    valence: np.ndarray
    arousal: np.ndarray
    expr: np.ndarray
    au: np.ndarray
    va_mask: np.ndarray
    expr_mask: np.ndarray
    au_mask: np.ndarray

    def __len__(self):
        return len(self.images)

    def summary(self) -> dict:
        return {
            "frames": int(len(self)),
            "videos": int(len(np.unique(self.videos))),
            "va_valid": int(self.va_mask.sum()),
            "expr_valid": int(self.expr_mask.sum()),
            "au_valid": int(self.au_mask.sum()),
        }


def parse_annotations(path: str | Path) -> AbawAnnotations:
    """Parse an ABAW MTL annotation file into row-aligned arrays + masks."""
    import pandas as pd

    path = Path(path)
    au_cols = [f"AU{i}" for i in range(N_AU)]
    names = ["image", "valence", "arousal", "expression"] + au_cols
    df = pd.read_csv(path, header=None, skiprows=1, names=names)
    if df.shape[1] != 4 + N_AU:
        raise ValueError(f"{path.name}: expected {4 + N_AU} columns, got {df.shape[1]}")

    images = df["image"].to_numpy(dtype=str)
    videos = np.array([s.split("/", 1)[0] for s in images])
    valence = df["valence"].to_numpy(dtype=np.float32)
    arousal = df["arousal"].to_numpy(dtype=np.float32)
    expr = df["expression"].to_numpy(dtype=np.int64)
    au = df[au_cols].to_numpy(dtype=np.int64)

    va_mask = (valence != VA_IGNORE) & (arousal != VA_IGNORE)
    expr_mask = expr != EXPR_IGNORE
    au_mask = au[:, 0] != AU_IGNORE

    return AbawAnnotations(images, videos, valence, arousal, expr, au,
                           va_mask, expr_mask, au_mask)


def class_balance(ann: AbawAnnotations) -> dict:
    """Expression class counts and per-AU positive rates over VALID rows.
    Used to build class weights / samplers (metric is macro-F1, imbalance-sensitive)."""
    e = ann.expr[ann.expr_mask]
    expr_counts = {int(c): int((e == c).sum()) for c in range(N_EXPR)}
    au = ann.au[ann.au_mask]
    au_pos = (au == 1).sum(0).astype(int).tolist()
    au_tot = int(au.shape[0])
    return {"expr_counts": expr_counts, "au_pos": au_pos, "au_total": au_tot}


def expr_sample_weights(ann: AbawAnnotations) -> np.ndarray:
    """Per-frame sampling weights to oversample rare EXPR classes (for a
    WeightedRandomSampler). Frames with a valid EXPR label get inverse-frequency
    weight (class-mean ~1, so rare emotions get >1); EXPR-invalid frames get 1.0 so
    VA/AU-only frames are still sampled. Attacks the rare-class collapse at the
    sampling level without discarding the other tasks' supervision."""
    counts = np.array([int((ann.expr[ann.expr_mask] == c).sum()) for c in range(N_EXPR)],
                      dtype=np.float64)
    counts = np.clip(counts, 1, None)
    inv = counts.sum() / (N_EXPR * counts)
    w = np.ones(len(ann), dtype=np.float64)
    valid = ann.expr_mask
    w[valid] = inv[ann.expr[valid]]
    return w


def subset_by_videos(ann: AbawAnnotations, videos: set[str]) -> AbawAnnotations:
    """Return a row subset whose video id is in `videos` (for tiny smoke subsets)."""
    keep = np.array([v in videos for v in ann.videos])
    return AbawAnnotations(
        ann.images[keep], ann.videos[keep], ann.valence[keep], ann.arousal[keep],
        ann.expr[keep], ann.au[keep], ann.va_mask[keep], ann.expr_mask[keep],
        ann.au_mask[keep])


def build_torch_dataset(ann: AbawAnnotations, image_root, train: bool,
                        img_size: int = 112, aug: str = "standard"):
    """Factory that constructs the torch Dataset, importing torch lazily so this
    module stays usable without it.

    aug (train only): 'standard' (flip + mild jitter) or 'strong' (random-resized
    crop + small rotation + stronger jitter + random erasing) for regularization.
    """
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train and aug == "strong":
        tf = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])
    elif train:
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    image_root = Path(image_root)

    class _AbawMtlDataset(Dataset):
        def __init__(self):
            self.ann = ann
            self.root = image_root
            self.tf = tf

        def __len__(self):
            return len(self.ann)

        def __getitem__(self, i):
            a = self.ann
            try:
                img = Image.open(self.root / a.images[i]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (img_size, img_size), (128, 128, 128))
            x = self.tf(img)
            targets = {
                "valence": torch.tensor(a.valence[i], dtype=torch.float32),
                "arousal": torch.tensor(a.arousal[i], dtype=torch.float32),
                "expr": torch.tensor(a.expr[i], dtype=torch.long),
                "au": torch.tensor(a.au[i], dtype=torch.float32),
            }
            masks = {
                "va": torch.tensor(bool(a.va_mask[i])),
                "expr": torch.tensor(bool(a.expr_mask[i])),
                "au": torch.tensor(bool(a.au_mask[i])),
            }
            return x, targets, masks

    return _AbawMtlDataset()
