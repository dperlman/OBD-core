#!/usr/bin/env python3
"""What the shared store holds, on one screen.

    python ~/git/OBD-core/tools/store_summary.py [--store ~/obd-data]

For each product (ties, cusps, positions): how many n it covers completely (all of p* >= 1/2) and
which, how many it covers only in windows, its rows and its size on disk; then any parts from
another numeric epoch (which the store would refuse) and the last few Store.check runs.  Reads only
the coverage files, so it takes a few seconds even with 100,000 n.
"""
import argparse
import json
import os

import obd_core
from obd_core import store


def runs(ns, limit=12):
    """'2-1000, 2000, 3000, ...' for a sorted list of n, shortened in the middle when long."""
    parts, a = [], None
    for k, n in enumerate(ns):
        if a is None:
            a = b = n
        elif n == b + 1:
            b = n
        else:
            parts.append(f"{a}" if a == b else f"{a}-{b}")
            a = b = n
        if k == len(ns) - 1:
            parts.append(f"{a}" if a == b else f"{a}-{b}")
    if len(parts) > limit:
        parts = parts[: limit // 2] + [f"... ({len(parts) - limit} more) ..."] + parts[-(limit // 2):]
    return ", ".join(parts)


def size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None)
    a = ap.parse_args()
    s = store.Store(a.store)
    meta = json.load(open(os.path.join(s.root, "STORE.json")))
    print(f"{s.root}   format {meta.get('format')}; this OBD-core: numeric epoch {obd_core.NUMERIC_EPOCH}\n")
    stale, grand = [], 0
    for kind in store.KINDS:
        full, partial, rows = [], [], 0
        d = os.path.join(s.root, kind)
        for n in s.ns(kind):
            cov = s.coverage(kind, n)
            parts = cov["parts"]
            rows += sum(p.get("rows") or 0 for p in parts)
            stale += [(kind, n) for p in parts if p.get("numeric_epoch") != obd_core.NUMERIC_EPOCH]
            ivs = store._merge([iv for p in parts if p.get("min_pair_mass", 0.0) == 0.0 for iv in p["intervals"]])
            (full if ivs == [[0.5, 1.0]] else partial).append(n)
        sz = size(d) if os.path.isdir(d) else 0
        grand += sz
        print(f"{kind:10s} {len(full):7,d} n complete   {rows:15,d} rows   {sz / 1e9:6.2f} GB")
        if full:
            print(f"{'':10s} complete: n = {runs(full)}")
        if partial:
            print(f"{'':10s} windows only: {len(partial):,d} n, n = {runs(partial, 6)}")
        print()
    print(f"total {grand / 1e9:.2f} GB")
    print("parts from another numeric epoch (refused until checked): " +
          ("none" if not stale else f"{len(stale)}, e.g. {stale[:5]}"))
    checks = meta.get("checks", [])
    if checks:
        print("last checks:")
        for c in checks[-3:]:
            print(f"  {c['at']}  {c['kind']:9s} {c['n_checked']:4d} n  OBD-core {c['obd_core_version']}  "
                  f"failures: {len(c['failures']) or 'none'}")


if __name__ == "__main__":
    main()
