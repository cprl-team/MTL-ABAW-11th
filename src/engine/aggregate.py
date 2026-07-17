from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCALARS = ["P_MTL", "P_MTL_calibrated", "VA", "CCC_valence", "CCC_arousal",
           "EXPR_macroF1", "AU_macroF1", "AU_macroF1_calibrated"]


def aggregate(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("seed_*/metrics.json"))
    if not files:
        single = run_dir / "metrics.json"
        if single.exists():
            files = [single]
        else:
            raise FileNotFoundError(f"no seed_*/metrics.json or metrics.json under {run_dir}")
    rows = [json.loads(f.read_text()) for f in files]
    summary = {"n_seeds": len(rows), "seeds": [r.get("seed", i) for i, r in enumerate(rows)]}
    for k in SCALARS:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        if vals.size:
            summary[k] = {"mean": float(vals.mean()),
                          "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                          "n": int(vals.size)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with seed_*/metrics.json")
    args = ap.parse_args()
    s = aggregate(args.run)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
