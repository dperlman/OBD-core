#!/usr/bin/env python3
"""Bring coverage.json files written by tools/build_store.py (2026-10-11) into the obd_core.store format.

build_store.py wrote one whole-table part per n with a short coverage.json ("parts": [file name]).
This rewrites each into the store's form (parts as records with intervals, min_pair_mass and
numeric_epoch), stamping epoch 1 for parts computed by OBD-core 0.6.1 (epoch 1 began with 0.6.1)
and leaving anything else unstamped, so the store refuses it until Store.check has compared it.

    python tools/adopt_build_store.py ~/obd-data
"""
import json
import os
import sys

EPOCH_1 = {"0.6.1"}


def adopt(root):
    changed = 0
    for kind in ("ties", "cusps"):
        d = os.path.join(root, kind)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name, "coverage.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                cov = json.load(f)
            if not cov.get("parts") or isinstance(cov["parts"][0], dict):
                continue
            v = cov.get("obd_core_version")
            parts = [{"file": p, "intervals": cov["intervals"], "min_pair_mass": 0.0, "rows": cov.get("rows"),
                      "numeric_epoch": 1 if v in EPOCH_1 else None, "obd_core_version": v,
                      "written_at": cov.get("written_at")} for p in cov["parts"]]
            new = {"format": cov["format"], "kind": kind, "n": cov["n"], "parts": parts}
            with open(path + ".tmp", "w") as f:
                json.dump(new, f, indent=1)
            os.replace(path + ".tmp", path)
            changed += 1
    print(f"{changed} coverage files adopted under {root}")


if __name__ == "__main__":
    adopt(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/obd-data"))
