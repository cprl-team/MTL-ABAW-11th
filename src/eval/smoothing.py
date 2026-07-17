from __future__ import annotations

from pathlib import Path

import numpy as np


def _frame_num(img: str):
    stem = Path(img).stem
    try:
        return int(stem)
    except ValueError:
        return None


def _filter1d(seq: np.ndarray, kind: str, window: int, sigma: float) -> np.ndarray:
    """Filter along axis 0 (time) of a (T, C) array with edge padding."""
    T = seq.shape[0]
    if T < 2 or kind == "none":
        return seq
    if kind == "box":
        w = max(1, int(window) | 1)
        r = w // 2
        kernel = np.ones(w) / w
    elif kind == "gaussian":
        r = max(1, int(round(3 * sigma)))
        x = np.arange(-r, r + 1)
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
    else:
        raise ValueError(f"unknown smoothing kind '{kind}'")
    if 2 * r + 1 > T:
        r = (T - 1) // 2
        if r < 1:
            return seq
        kernel = kernel[len(kernel) // 2 - r: len(kernel) // 2 + r + 1]
        kernel = kernel / kernel.sum()
    out = np.empty_like(seq, dtype=np.float64)
    padded = np.pad(seq, ((r, r), (0, 0)), mode="edge")
    for c in range(seq.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


def smooth_streams(videos, images, va, expr_prob, au_prob,
                   kind="gaussian", window=7, sigma=2.0):
    """Smooth per-frame streams within each video, in frame order.

    videos: (N,) video id per frame; images: (N,) "video/frame.jpg".
    va: (N,2) valence/arousal preds; expr_prob: (N,8) softmax; au_prob: (N,12) sigmoid.
    Returns smoothed (va, expr_prob, au_prob), row-aligned to the inputs.
    """
    va, expr_prob, au_prob = (np.asarray(a, dtype=np.float64).copy()
                              for a in (va, expr_prob, au_prob))
    videos = np.asarray(videos)
    fnum = np.array([_frame_num(im) for im in images], dtype=object)
    for v in np.unique(videos):
        idx = np.where(videos == v)[0]
        keys = fnum[idx]
        order = idx[np.argsort([k if k is not None else j for j, k in enumerate(keys)])]
        for arr in (va, expr_prob, au_prob):
            arr[order] = _filter1d(arr[order], kind, window, sigma)
    return va, expr_prob, au_prob
