from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.abaw import AbawAnnotations

ABAW_AU = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]
UNBC_CODED = [4, 6, 7, 10, 12, 25, 26]


def parse_unbc(manifest: str | Path, split: str,
               root: str | Path = "data/unbc", val_every: int = 20) -> AbawAnnotations:
    root = Path(root)
    entries = json.loads(Path(manifest).read_text())
    coded_seqs = {e["seq"] for e in entries if e.get("au")}
    rows = [e for e in entries if e["seq"] in coded_seqs]
    subs = sorted({e["subject"] for e in rows})
    val_subs = set(subs[::val_every]) if val_every else set()
    rows = [e for e in rows if (e["subject"] in val_subs) == (split == "val")]

    n = len(rows)
    idx = {au: ABAW_AU.index(au) for au in ABAW_AU}
    images = np.empty(n, dtype=object)
    au = np.full((n, 12), -1, dtype=np.int64)
    for j in UNBC_CODED:
        au[:, idx[j]] = 0
    videos = np.empty(n, dtype=object)
    for i, e in enumerate(rows):
        crop = e["crop"]
        images[i] = crop[crop.index("aligned_112/"):]
        videos[i] = e["seq"]
        for k, v in (e.get("au") or {}).items():
            a = int(k)
            if a in idx and a in UNBC_CODED and float(v) > 0:
                au[i, idx[a]] = 1
    images = images.astype(str); videos = videos.astype(str)
    valence = np.full(n, -5.0, dtype=np.float32)
    arousal = np.full(n, -5.0, dtype=np.float32)
    expr = np.full(n, -1, dtype=np.int64)
    au_mask = (au != -1).any(axis=1)
    return AbawAnnotations(images, videos, valence, arousal, expr, au,
                           np.zeros(n, bool), np.zeros(n, bool), au_mask)
