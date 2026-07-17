import argparse
import json
from pathlib import Path

import numpy as np

from .significance import paired_permutation_test


def _per_seed(run, metric):
    files = sorted(Path(run).glob("seed_*/metrics.json"))
    if not files:
        single = Path(run) / "metrics.json"
        files = [single] if single.exists() else []
    if not files:
        raise SystemExit(f"no seed_*/metrics.json or metrics.json under {run}")
    return np.array([json.loads(f.read_text())[metric] for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="ours")
    ap.add_argument("--run-b", required=True, help="baseline")
    ap.add_argument("--metric", default="P_MTL")
    args = ap.parse_args()

    a, b = _per_seed(args.run_a, args.metric), _per_seed(args.run_b, args.metric)
    if a.shape != b.shape:
        raise SystemExit(f"seed-count mismatch: {a.shape} vs {b.shape}")
    diff, p = paired_permutation_test(a, b)
    print(f"metric: {args.metric}  (paired over {len(a)} seeds)")
    print(f"  A {a.mean():.3f} ± {a.std():.3f}   B {b.mean():.3f} ± {b.std():.3f}")
    print(f"  mean diff (A-B) = {diff:+.3f}   p = {p:.4f}   "
          f"{'(significant, p<0.05)' if p < 0.05 else '(not significant at 0.05)'}")


if __name__ == "__main__":
    main()
