from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.abaw import AbawAnnotations

AFFECTNET_NAMES = ["Neutral", "Happiness", "Sadness", "Surprise",
                   "Fear", "Disgust", "Anger", "Contempt"]
N_EXPR_AN = 8


def parse_affectnet(manifest: str | Path, split: str,
                    root: str | Path = "data/affectnet",
                    au_map: str | Path | None = None) -> AbawAnnotations:
    import pandas as pd
    df = pd.read_csv(manifest)
    df = df[df["split"] == split]
    df = df[df["expression"] <= 7]
    images = df["out"].to_numpy(dtype=str)
    videos = np.array([f"img{i}" for i in range(len(df))])
    valence = df["valence"].to_numpy(dtype=np.float32)
    arousal = df["arousal"].to_numpy(dtype=np.float32)
    expr = df["expression"].to_numpy(dtype=np.int64)
    va_mask = (valence > -1.5) & (arousal > -1.5)
    valence = np.where(va_mask, valence, np.float32(-5.0))
    arousal = np.where(va_mask, arousal, np.float32(-5.0))
    au = np.full((len(df), 12), -1, dtype=np.int64)
    au_mask = np.zeros(len(df), bool)
    if au_map is not None:
        import json
        amap = json.loads(Path(au_map).read_text())
        for i, o in enumerate(images):
            h = str(o).split("/")[-1].split(".")[0]
            v = amap.get(h)
            if v is not None and v[0] != -1:
                au[i] = np.asarray(v, dtype=np.int64)
                au_mask[i] = True
    return AbawAnnotations(images, videos, valence, arousal, expr, au,
                           va_mask, np.ones(len(df), bool), au_mask)
