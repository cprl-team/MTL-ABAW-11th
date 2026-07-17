from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    parts = [np.load(d, allow_pickle=True) for d in args.dumps]
    base = parts[0]
    avg = {k: np.mean([p[k] for p in parts], axis=0) for k in ("pv", "pe", "pa")}
    keep = {k: base[k] for k in ("valence", "arousal", "expr", "au",
                                 "m_va", "m_expr", "m_au", "videos")}
    np.savez(args.out, **avg, **keep)
    print(f"averaged {len(parts)} dumps -> {args.out}")


if __name__ == "__main__":
    main()
