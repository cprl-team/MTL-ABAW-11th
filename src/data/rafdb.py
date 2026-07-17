from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.abaw import AbawAnnotations

_RAF2AN = {1: 3,
           2: 4,
           3: 5,
           4: 1,
           5: 2,
           6: 6,
           7: 0}
RAFDB_PRESENT = [0, 1, 2, 3, 4, 5, 6]


def parse_rafdb(label_file: str | Path, split: str,
                root: str | Path = "data/RAF-DB") -> AbawAnnotations:
    """split: 'train' or 'test' (official partition, filename prefix)."""
    names, exprs = [], []
    for line in Path(label_file).read_text().splitlines():
        if not line.strip():
            continue
        fn, lab = line.split()
        if not fn.startswith(split):
            continue
        stem = fn.rsplit(".", 1)[0]
        names.append(f"basic/Image/aligned/{stem}_aligned.jpg")
        exprs.append(_RAF2AN[int(lab)])
    n = len(names)
    images = np.array(names, dtype=str)
    videos = np.array([f"img{i}" for i in range(n)])
    expr = np.array(exprs, dtype=np.int64)
    valence = np.full(n, -5.0, dtype=np.float32)
    arousal = np.full(n, -5.0, dtype=np.float32)
    au = np.full((n, 12), -1, dtype=np.int64)
    return AbawAnnotations(images, videos, valence, arousal, expr, au,
                           np.zeros(n, bool), np.ones(n, bool), np.zeros(n, bool))


RAFDB_COMPOUND_N = 11


def parse_rafdb_compound(label_file: str | Path, split: str,
                         root: str | Path = "data/RAF-DB") -> AbawAnnotations:
    """split: 'train' or 'test' (official partition, filename prefix). EXPR field = compound 0-10."""
    names, exprs = [], []
    for line in Path(label_file).read_text().splitlines():
        if not line.strip():
            continue
        fn, lab = line.split()
        if not fn.startswith(split):
            continue
        stem = fn.rsplit(".", 1)[0]
        names.append(f"compound/Image/aligned/{stem}_aligned.jpg")
        exprs.append(int(lab) - 1)
    n = len(names)
    images = np.array(names, dtype=str)
    videos = np.array([f"img{i}" for i in range(n)])
    expr = np.array(exprs, dtype=np.int64)
    z = np.full(n, -5.0, dtype=np.float32)
    au = np.full((n, 12), -1, dtype=np.int64)
    return AbawAnnotations(images, videos, z, z.copy(), expr, au,
                           np.zeros(n, bool), np.ones(n, bool), np.zeros(n, bool))
