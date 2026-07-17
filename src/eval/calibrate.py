from __future__ import annotations

import numpy as np

from src.metrics.mtl import expr_score


def expr_logit_adjust(expr_true, expr_prob, train_prior, taus=None):
    """expr_prob: (N,8) softmax. train_prior: (8,) class frequencies in TRAIN.
    Returns (best_tau, macro_f1, per_class_f1, predictions). tau=0 is plain argmax."""
    if taus is None:
        taus = np.linspace(0.0, 3.0, 31)
    logp = np.log(np.clip(np.asarray(expr_prob, float), 1e-8, 1.0))
    logpi = np.log(np.clip(np.asarray(train_prior, float), 1e-8, 1.0))
    best_tau, (best_macro, best_pc) = 0.0, expr_score(expr_true, logp.argmax(1))
    best_pred = logp.argmax(1)
    for t in taus:
        pred = (logp - t * logpi).argmax(1)
        m, pc = expr_score(expr_true, pred)
        if m > best_macro:
            best_macro, best_pc, best_tau, best_pred = m, pc, float(t), pred
    return best_tau, best_macro, best_pc, best_pred
