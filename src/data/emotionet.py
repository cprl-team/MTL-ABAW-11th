from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.data.abaw import AbawAnnotations

ABAW_AU = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]


def _aucol(df, n):
    for c in df.columns:
        if c.strip().strip("'") == f"AU {n}":
            return c
    raise KeyError(f"AU {n} column not found")


def parse_emotionet(csv: str | Path, split: str,
                    root: str | Path = "data/emotionnet",
                    val_every: int = 20) -> AbawAnnotations:
    import pandas as pd
    root = Path(root)
    df = pd.read_csv(csv)
    url = df.columns[0]
    df["_stem"] = df[url].map(lambda u: (m.group(1) if (m := re.search(r"(N_\d+_\d+)\.jpg", str(u))) else None))
    df = df[df["_stem"].notna()].copy()
    imgdir = root / "aligned_labeled"
    have = {p.stem for p in imgdir.glob("*.jpg")}
    df = df[df["_stem"].isin(have)].reset_index(drop=True)
    is_val = (np.arange(len(df)) % val_every == 0)
    df = df[is_val] if split == "val" else df[~is_val]
    df = df.reset_index(drop=True)

    n = len(df)
    images = np.array([f"aligned_labeled/{s}.jpg" for s in df["_stem"]], dtype=str)
    videos = np.array([f"img{i}" for i in range(n)])
    au = df[[_aucol(df, a) for a in ABAW_AU]].to_numpy(dtype=np.int64)
    au = np.where(au == 999, -1, au)
    au_mask = (au != -1).any(axis=1)
    valence = np.full(n, -5.0, dtype=np.float32)
    arousal = np.full(n, -5.0, dtype=np.float32)
    expr = np.full(n, -1, dtype=np.int64)
    return AbawAnnotations(images, videos, valence, arousal, expr, au,
                           np.zeros(n, bool), np.zeros(n, bool), au_mask)
