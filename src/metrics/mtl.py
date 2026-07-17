from __future__ import annotations

import numpy as np

VA_IGNORE = -5.0
EXPR_IGNORE = -1
AU_IGNORE = -1
N_EXPR = 8
N_AU = 12
AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
            "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient between two 1-D arrays.

    CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2).
    Returns 0.0 if fewer than 2 valid points or zero total variance.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return 0.0
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    denom = vt + vp + (mt - mp) ** 2
    if denom <= 0:
        return 0.0
    return float(2 * cov / denom)


def va_score(v_true, v_pred, a_true, a_pred):
    """Mean CCC for valence and arousal, dropping VA_IGNORE frames per dimension.

    Returns (mean_ccc, ccc_valence, ccc_arousal).
    """
    v_true = np.asarray(v_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    a_true = np.asarray(a_true, dtype=np.float64)
    a_pred = np.asarray(a_pred, dtype=np.float64)
    mv = v_true != VA_IGNORE
    ma = a_true != VA_IGNORE
    ccc_v = ccc(v_true[mv], v_pred[mv])
    ccc_a = ccc(a_true[ma], a_pred[ma])
    return 0.5 * (ccc_v + ccc_a), ccc_v, ccc_a


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def expr_score(y_true, y_pred):
    """Macro-F1 over the 8 expression classes, dropping EXPR_IGNORE frames.

    Returns (macro_f1, per_class_f1 list of length N_EXPR). Macro-F1 is the mean
    over all 8 classes — matching the official (1/8) sum_c F1_c definition.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = y_true != EXPR_IGNORE
    yt, yp = y_true[m], y_pred[m]
    per_class = []
    for c in range(N_EXPR):
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        per_class.append(_binary_f1(tp, fp, fn))
    macro = float(np.mean(per_class))
    return macro, per_class


def au_score(y_true, y_pred):
    """Macro-F1 over the 12 AUs (multi-label), dropping AU_IGNORE rows.

    y_true, y_pred: arrays of shape (N, 12) in {0, 1} with AU_IGNORE marking
    frames whose AU labels are invalid (the whole 12-vector is masked together in
    this dataset). Returns (macro_f1, per_au_f1 list of length N_AU).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_au = []
    for k in range(N_AU):
        col_t = y_true[:, k]
        col_p = y_pred[:, k]
        m = col_t != AU_IGNORE
        t, p = col_t[m], col_p[m]
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        per_au.append(_binary_f1(tp, fp, fn))
    macro = float(np.mean(per_au)) if per_au else 0.0
    return macro, per_au


def mtl_score(va, expr_macro, au_macro):
    """Combine the three task scores into the overall P_MTL in [0, 3]."""
    return float(va + expr_macro + au_macro)


def all_metrics(v_true, v_pred, a_true, a_pred, expr_true, expr_pred,
                au_true, au_pred) -> dict:
    """Compute the full metric dict from pooled predictions. This is what the
    training/eval loop writes into metrics.json."""
    va, ccc_v, ccc_a = va_score(v_true, v_pred, a_true, a_pred)
    expr_macro, expr_pc = expr_score(expr_true, expr_pred)
    au_macro, au_pc = au_score(au_true, au_pred)
    return {
        "P_MTL": mtl_score(va, expr_macro, au_macro),
        "VA": va, "CCC_valence": ccc_v, "CCC_arousal": ccc_a,
        "EXPR_macroF1": expr_macro,
        "EXPR_per_class": {str(i): expr_pc[i] for i in range(N_EXPR)},
        "AU_macroF1": au_macro,
        "AU_per_unit": {AU_NAMES[i]: au_pc[i] for i in range(N_AU)},
    }
