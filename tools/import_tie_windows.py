#!/usr/bin/env python3
"""Copy OBD's old tie-point window cache (data/tie_windows/) into the store as tie positions.

That cache (2026-10-08..09, OBD-core 0.6.0 and 0.6.1) holds, per n, the p intervals that
cusp-proximity --points ties searched with screen(collect_all=True) and every tie point found:
exactly what Store.positions computes, so it becomes positions parts without recomputing.  Screen
did not change between 0.6.0 and 0.6.1 (only tie_table's n_tied_pairs dtype did), so all of it is
epoch 1.  Windows that ran past p = 1 are clamped to 1 (no tie point lies beyond it).  n that already
have positions in the store are skipped.  Afterwards, spot-check with Store.check("positions", ns=[...]).

    python tools/import_tie_windows.py ~/git/OBD/data/tie_windows [--store ~/obd-data]
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from obd_core import store


def _read_dir(path, cols):
    out = defaultdict(list)
    for name in sorted(os.listdir(path)):
        if name.endswith(".parquet"):
            t = pq.read_table(os.path.join(path, name), columns=cols)
            for c in cols:
                out[c].append(t[c].to_numpy())
    return {c: np.concatenate(v) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--store", default=None)
    args = ap.parse_args()
    s = store.Store(args.store)
    cov = _read_dir(os.path.join(args.cache, "coverage"), ["n", "lo", "hi", "min_pair_mass"])
    pts = _read_dir(os.path.join(args.cache, "ties"), ["n", "i", "j", "pstar"])
    if np.any(cov["min_pair_mass"] != 0):
        raise SystemExit("the cache holds searches with a min_pair_mass; this tool copies complete searches only")
    by_n = defaultdict(list)
    for n, lo, hi in zip(cov["n"].tolist(), cov["lo"].tolist(), cov["hi"].tolist()):
        by_n[n].append((max(lo, 0.5), min(hi, 1.0)))
    order = np.argsort(pts["n"], kind="stable")
    pn = pts["n"][order]
    starts = np.searchsorted(pn, sorted(by_n), side="left")
    ends = np.searchsorted(pn, sorted(by_n), side="right")
    done = skipped = rows = 0
    for (n, a, b) in zip(sorted(by_n), starts, ends):
        if s.coverage("positions", n)["parts"]:
            skipped += 1
            continue
        wins = store._merge(by_n[n])
        k = order[a:b]
        i, j, p = pts["i"][k].astype(np.int64), pts["j"][k].astype(np.int64), pts["pstar"][k]
        inside = np.zeros(p.size, bool)
        for lo, hi in wins:
            inside |= (p >= lo) & (p <= hi)
        key = i[inside] * (n + 1) + j[inside]
        _, first = np.unique(key, return_index=True)
        sel = np.flatnonzero(inside)[np.sort(first)]
        sel = sel[np.argsort(p[sel], kind="stable")]
        s._save("positions", n, {"i": i[sel], "j": j[sel], "pstar": p[sel]}, wins, 0.0)
        done += 1
        rows += sel.size
        if done % 10000 == 0:
            print(f"  {done} n copied", flush=True)
    print(f"copied {done} n ({rows:,} tie points), skipped {skipped} already in the store")


if __name__ == "__main__":
    main()
