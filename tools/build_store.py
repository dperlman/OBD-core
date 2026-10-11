#!/usr/bin/env python3
"""Fill ~/obd-data from OBD-core, checking every older copy on the way (the 2026-10 store migration).

For each n, in priority order, computes tie_table(n) with the installed OBD-core, compares it with
every older table of that n that exists:

  OBD          ~/git/OBD/data/tie_points/n=NNNNN/part.parquet   (both halves, OBD-core 0.3.2)
  cusps-ties   ~/git/ordered-binomial-cusps/data/ties/n=NNNNN/part.parquet
  cusps-csv    ~/git/ordered-binomial-cusps/cusps/nNNNNN.csv     (cusps only, from cusps_fast)

and writes the result into the store through obd_core.store: the full tie table (Store.ties) for
the n whose tables are kept, the cusps alone (Store.cusps) for the rest.  Each comparison is one row
of verification/sweep-<date>.csv.  A rerun skips every n the store already covers; it stops starting
new n after --deadline (unix seconds).

The first run (2026-10-11, until 16:14) used an earlier version of this script that wrote
coverage.json in a short form; run tools/adopt_build_store.py on the store before resuming.

    python tools/build_store.py --store ~/obd-data --deadline 1791760456
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
from importlib.metadata import version
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq


HOME = os.path.expanduser("~")
OBD_TIES = f"{HOME}/git/OBD/data/tie_points/n={{:05d}}/part.parquet"
CUSPS_TIES = f"{HOME}/git/ordered-binomial-cusps/data/ties/n={{:05d}}/part.parquet"
CUSPS_CSV = f"{HOME}/git/ordered-binomial-cusps/cusps/n{{:05d}}.csv"
FORMAT = "obd.store.v1"


# ---------------------------------------------------------------- comparing

def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0:
        return 0.0
    d = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    d[(a == b) | (np.isnan(a) & np.isnan(b))] = 0.0
    return float(np.nanmax(d))


def _by_pair(i, j):
    return np.lexsort((np.asarray(j, np.int64), np.asarray(i, np.int64)))


def compare_obd(n, cur):
    path = OBD_TIES.format(n)
    if not os.path.isfile(path):
        return None
    o = pq.read_table(path, columns=["p", "i", "j", "E", "slope_left", "slope_right", "log10_D", "is_cusp", "decided_by"])
    o = {k: o[k].to_numpy(zero_copy_only=False) for k in o.column_names}
    up = o["p"] > 0.5
    o = {k: v[up] for k, v in o.items()}
    c = {k: np.asarray(cur[k])[1:] for k in ("pstar", "i", "j", "E", "slope_left", "slope_right", "log10_D", "is_cusp", "decided_by")}
    a, b = _by_pair(o["i"], o["j"]), _by_pair(c["i"], c["j"])
    same_pairs = len(a) == len(b) and np.array_equal(o["i"][a], c["i"][b]) and np.array_equal(o["j"][a], c["j"][b])
    if not same_pairs:
        return dict(source="OBD", rows_old=len(a), rows_new=len(b), pairs_same=False)
    r = dict(source="OBD", rows_old=len(a), rows_new=len(b), pairs_same=True,
             cusps_same=bool(np.array_equal(o["is_cusp"][a], c["is_cusp"][b])),
             decided_by_same=bool(np.array_equal(o["decided_by"][a].astype(str), c["decided_by"][b].astype(str))),
             rel_p=_rel(o["p"][a], c["pstar"][b]), rel_E=_rel(o["E"][a], c["E"][b]),
             rel_slopes=max(_rel(o["slope_left"][a], c["slope_left"][b]), _rel(o["slope_right"][a], c["slope_right"][b])),
             abs_log10_D=float(np.nanmax(np.abs(o["log10_D"][a] - c["log10_D"][b]))) if len(a) else 0.0)
    r["bit_identical"] = r["rel_p"] == r["rel_E"] == r["rel_slopes"] == r["abs_log10_D"] == 0.0
    return r


def compare_cusps_ties(n, cur):
    path = CUSPS_TIES.format(n)
    if not os.path.isfile(path):
        return None
    cols = ["i", "j", "pstar", "ln_fi", "S_minus", "F3", "is_cusp", "decided_by"]
    o = pq.read_table(path, columns=cols)
    o = {k: o[k].to_numpy(zero_copy_only=False) for k in cols}
    up = o["pstar"] > 0.5
    o = {k: v[up] for k, v in o.items()}
    c = {k: np.asarray(cur[k])[1:] for k in cols}
    a, b = _by_pair(o["i"], o["j"]), _by_pair(c["i"], c["j"])
    same_pairs = len(a) == len(b) and np.array_equal(o["i"][a], c["i"][b]) and np.array_equal(o["j"][a], c["j"][b])
    if not same_pairs:
        return dict(source="cusps-ties", rows_old=len(a), rows_new=len(b), pairs_same=False)
    rel = {k: _rel(o[k][a], c[k][b]) for k in ("pstar", "ln_fi", "S_minus", "F3")}
    return dict(source="cusps-ties", rows_old=len(a), rows_new=len(b), pairs_same=True,
                cusps_same=bool(np.array_equal(o["is_cusp"][a], c["is_cusp"][b])),
                decided_by_same=bool(np.array_equal(o["decided_by"][a].astype(str), c["decided_by"][b].astype(str))),
                rel_p=rel["pstar"], rel_other=max(rel["ln_fi"], rel["S_minus"], rel["F3"]),
                bit_identical=all(v == 0.0 for v in rel.values()))


def compare_csv(n, cur):
    path = CUSPS_CSV.format(n)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    oi = np.array([int(r["i"]) for r in rows], np.int64)
    oj = np.array([int(r["j"]) for r in rows], np.int64)
    k = np.flatnonzero(np.asarray(cur["is_cusp"])[1:]) + 1          # cusps, without the axis row
    ci, cj = np.asarray(cur["i"])[k].astype(np.int64), np.asarray(cur["j"])[k].astype(np.int64)
    a, b = _by_pair(oi, oj), _by_pair(ci, cj)
    same = len(a) == len(b) and np.array_equal(oi[a], ci[b]) and np.array_equal(oj[a], cj[b])
    r = dict(source="cusps-csv", rows_old=len(a), rows_new=len(b), pairs_same=bool(same), cusps_same=bool(same))
    if not same:
        old, new = set(zip(oi.tolist(), oj.tolist())), set(zip(ci.tolist(), cj.tolist()))
        r["only_old"] = ";".join(f"{x}-{y}" for x, y in sorted(old - new)[:10])
        r["only_new"] = ";".join(f"{x}-{y}" for x, y in sorted(new - old)[:10])
        return r
    f = lambda name: np.array([float(rows[q][name]) for q in a])
    r["rel_p"] = _rel(f("pstar"), np.asarray(cur["pstar"])[k][b])
    r["rel_E"] = _rel(f("E"), np.asarray(cur["E"])[k][b])
    r["rel_slopes"] = max(_rel(f("slope_left"), np.asarray(cur["slope_left"])[k][b]),
                          _rel(f("slope_right"), np.asarray(cur["slope_right"])[k][b]))
    r["rel_other"] = max(_rel(f("S_minus"), np.asarray(cur["S_minus"])[k][b]), _rel(f("F3"), np.asarray(cur["F3"])[k][b]))
    r["bit_identical"] = r["rel_p"] == r["rel_E"] == r["rel_slopes"] == r["rel_other"] == 0.0
    return r


# ---------------------------------------------------------------- plan

def plan():
    """(n, keep_ties) in priority order."""
    order, seen = [], set()

    def add(ns, keep):
        for n in ns:
            if n not in seen:
                seen.add(n)
                order.append((int(n), keep))
    add(range(2, 1001), True)                                            # every n <= 1000: full tables
    add(range(2000, 8001, 1000), True)                                   # the cusps repo's large tables
    anim = sorted(set(np.round(np.geomspace(50, 8000, 100)).astype(int).tolist()))
    add([n for n in anim if n > 1000], True)                             # the animations' n
    add(range(1100, 5001, 100), False)                                   # then cusps: every 100th n,
    add(range(1010, 5001, 10), False)                                    # every 10th,
    add(range(1001, 5001), False)                                        # every n
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(HOME, "obd-data"))
    ap.add_argument("--deadline", type=float, default=float("inf"), help="unix time after which no new n starts")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    store = args.store
    os.makedirs(os.path.join(store, "verification"), exist_ok=True)
    report = os.path.join(store, "verification", f"sweep-{dt.date.today().isoformat()}.csv")
    fields = ["n", "source", "rows_old", "rows_new", "pairs_same", "cusps_same", "decided_by_same", "bit_identical",
              "rel_p", "rel_E", "rel_slopes", "rel_other", "abs_log10_D", "only_old", "only_new"]
    new_file = not os.path.isfile(report)
    out = open(report, "a", newline="")
    w = csv.DictWriter(out, fieldnames=fields)
    if new_file:
        w.writeheader()
    from obd_core import store as obd_store

    s = obd_store.Store(store)
    t0, done, bad = time.time(), 0, 0
    todo = plan()
    print(f"OBD-core {version('obd-core')}; {len(todo)} n planned; store {store}", flush=True)
    with Pool(args.workers) as pool:
        for n, keep in todo:
            full = [[0.5, 1.0]]
            if s.covered("cusps", n) == full and (not keep or s.covered("ties", n) == full):
                continue
            if time.time() > args.deadline:
                print(f"deadline reached; stopping before n = {n}", flush=True)
                break
            t1 = time.time()
            cur = s.ties(n, workers=args.workers, pool=pool) if keep else s.cusps(n, workers=args.workers, pool=pool)
            tc = time.time() - t1
            rows = [r for r in ((compare_obd(n, cur), compare_cusps_ties(n, cur)) if keep else ()) if r]
            rows += [r for r in (compare_csv(n, cur),) if r]           # works on a cusp table too
            for r in rows:
                r["n"] = n
                w.writerow({k: r.get(k, "") for k in fields})
                if not (r.get("pairs_same") and r.get("cusps_same")):
                    bad += 1
                    print(f"  MISMATCH n = {n} vs {r['source']}: {r}", flush=True)
            out.flush()
            size = sum(os.path.getsize(os.path.join(s._dir(k, n), p["file"]))
                       for k in ("ties", "cusps") for p in s.coverage(k, n)["parts"])
            done += 1
            if keep or n % 100 == 0 or done % 50 == 0:
                print(f"[{time.time() - t0:6.0f}s] n = {n}: {tc:.1f}s compute, checked against "
                      f"{','.join(r['source'] for r in rows) or 'nothing'}, wrote {size / 1e6:.1f} MB "
                      f"({done} n done, {bad} mismatches)", flush=True)
    out.close()
    print(f"finished: {done} n written, {bad} mismatches, {time.time() - t0:.0f}s; report {report}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
