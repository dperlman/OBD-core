#!/usr/bin/env python3
"""CLAUDE.md rule 3: show that a change to the kernel leaves every output unchanged, byte for byte.

Recomputes a fixed set of n with the code under test and compares every stored row with it, bit
for bit and dtype for dtype (Store.check), against the shared store (~/obd-data), which holds what
earlier releases of the same numeric epoch computed.  Exit status 0 when nothing differs.

    python tools/regression_check.py [--store ~/obd-data] [--workers 8]

The n are chosen to exercise every route: small n, the n where interval certification decides
verdicts (1055, 1162), large n, and tie positions far past the tie tables.  About 1.5 minutes on 8
cores.  If a change is MEANT to alter output, bump NUMERIC_EPOCH (rule 8) instead of expecting this
to pass.  (Until 2026-10-12 rule 3 ran the cusps repo's cusps_fast.py and dump_ties.py against that
repo's local files; both are thin drivers of this package and their outputs are in the store.)
"""
import argparse
import sys
import time
from multiprocessing import Pool

from obd_core import store

TIES = [3, 7, 100, 135, 217, 274, 1000, 2000, 3000]
CUSPS = [3, 600, 1055, 1162, 2001, 3000, 4000, 5000]
POSITIONS = [1500, 20000, 99999]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    s = store.Store(args.store)
    failed = False
    with Pool(args.workers) as pool:
        for kind, ns in (("ties", TIES), ("cusps", CUSPS), ("positions", POSITIONS)):
            have = [n for n in ns if s.coverage(kind, n)["parts"]]
            missing = sorted(set(ns) - set(have))
            if missing:
                print(f"{kind}: not in the store, so not checked: {missing}")
            t0 = time.time()
            r = s.check(kind, ns=have, workers=args.workers, pool=pool, reference_sample=2)
            print(f"{kind}: {r['n_checked']} n checked in {time.time() - t0:.0f}s; "
                  f"{'identical' if not r['failures'] else 'DIFFERENT: ' + str(r['failures'])}", flush=True)
            failed |= bool(r["failures"])
    print("rule 3: PASS" if not failed else "rule 3: FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
