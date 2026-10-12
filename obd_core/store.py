"""The shared store: certified tie tables and cusp tables, computed once and read back.

    from obd_core import store
    s = store.Store()                          # $OBD_DATA, or ~/obd-data
    t = s.ties(4000)                           # read it, or compute it with tie_table and save it
    w = s.ties(20000, p_range=(0.6, 0.6002))   # a window: computes only what is not yet covered
    c = s.cusps(4321)                          # certified cusps of one n (saved; the tie table is not)

Every value comes from ``tie_table``: the store computes nothing itself (CLAUDE.md, rule 1), and
what it returns is exactly what ``tie_table`` returns for the same request, row for row and bit for
bit (tests/test_store.py).  Tables cover p* >= 1/2, as ``tie_table`` does by default; the row at
p = 1/2 is the axis.

Layout, one directory per product and n:

    STORE.json                       format and the log of `check` runs
    ties/n=NNNNN/coverage.json       which p intervals are complete, and in which part files
    ties/n=NNNNN/part-*.parquet      tie_table rows (all columns, same dtypes)
    cusps/n=NNNNN/...                the same for the rows with is_cusp

**Trust.**  Every part records the ``NUMERIC_EPOCH`` it was computed under.  The epoch changes only
when OBD-core's outputs change (CLAUDE.md, rule 3 decides that), so stored data stays valid across
releases that keep it, and a part from another epoch is refused with ``StaleData``, never used.
``Store.check`` recomputes stored n and compares them bit for bit, and records the result in
STORE.json.

**Safety.**  Parts are written under a temporary name and renamed; coverage.json is rewritten (also
by rename) only after its part is in place.  An interrupted run leaves no partial data, and the next
request recomputes only what is missing.  One process at a time should write a given n.

Needs pyarrow (``pip install obd-core[store]``).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from importlib.metadata import version as _version

import numpy as np

import obd_core

FORMAT = "obd.store.v1"
KINDS = ("ties", "cusps", "positions")


class StaleData(RuntimeError):
    """A stored part was computed under a different NUMERIC_EPOCH than the running OBD-core."""


def default_root() -> str:
    return os.environ.get("OBD_DATA") or os.path.join(os.path.expanduser("~"), "obd-data")


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _core_version() -> str:
    try:
        return _version("obd-core")
    except Exception:
        return "unknown"


def _merge(intervals):
    """Union of closed intervals, sorted and merged."""
    out = []
    for lo, hi in sorted((float(a), float(b)) for a, b in intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _gaps(lo, hi, covered):
    """The parts of [lo, hi] not inside any covered interval (closed), as closed intervals.

    A tie point exactly on a covered edge is already stored, so recomputing a gap that shares that
    edge only duplicates it; reads drop duplicates by pair."""
    gaps, at = [], lo
    for a, b in _merge(covered):
        if b < at or a > hi:
            continue
        if a > at:
            gaps.append([at, min(a, hi)])
        at = max(at, b)
        if at >= hi:
            break
    if at < hi or (not gaps and not covered and lo == hi):
        gaps.append([at, hi])
    return [g for g in gaps if g[1] >= g[0]]


class Store:
    def __init__(self, root: str | None = None):
        import pyarrow  # noqa: F401  (fail early, with a clear message, if the extra is missing)

        self.root = os.path.abspath(os.path.expanduser(root or default_root()))
        os.makedirs(self.root, exist_ok=True)
        meta = os.path.join(self.root, "STORE.json")
        if not os.path.isfile(meta):
            self._write_json(meta, {"format": FORMAT, "created_at": _now(), "checks": []})

    # ------------------------------------------------------------ paths and coverage

    def _dir(self, kind: str, n: int) -> str:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        return os.path.join(self.root, kind, f"n={int(n):05d}")

    @staticmethod
    def _write_json(path: str, obj) -> None:
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(path + ".tmp", path)

    def coverage(self, kind: str, n: int) -> dict:
        """{"parts": [{"file", "intervals", "min_pair_mass", "numeric_epoch", ...}], ...}, empty if none."""
        path = os.path.join(self._dir(kind, n), "coverage.json")
        if not os.path.isfile(path):
            return {"format": FORMAT, "kind": kind, "n": int(n), "parts": []}
        with open(path) as f:
            return json.load(f)

    def covered(self, kind: str, n: int, min_pair_mass: float = 0.0) -> list:
        """Merged p intervals of n that are complete, counting only parts usable at min_pair_mass."""
        parts = self._usable(kind, n, min_pair_mass)
        return _merge([iv for p in parts for iv in p["intervals"]])

    def _usable(self, kind, n, min_pair_mass):
        parts = self.coverage(kind, n)["parts"]
        stale = [p["file"] for p in parts if p.get("numeric_epoch") != obd_core.NUMERIC_EPOCH]
        if stale:
            raise StaleData(f"{kind} n = {n}: {len(stale)} part(s) were computed under another numeric epoch "
                            f"(running {obd_core.NUMERIC_EPOCH}): {stale[:3]} in {self._dir(kind, n)}. "
                            f"Run Store.check to see whether they still agree, or delete them to recompute.")
        # a part searched with a mass threshold m is complete only for requests that accept m
        return [p for p in parts if p.get("min_pair_mass", 0.0) <= min_pair_mass]

    def ns(self, kind: str) -> list[int]:
        d = os.path.join(self.root, kind)
        if not os.path.isdir(d):
            return []
        return sorted(int(x[2:]) for x in os.listdir(d)
                      if x.startswith("n=") and os.path.isfile(os.path.join(d, x, "coverage.json")))

    # ------------------------------------------------------------ reading and writing parts

    def _save(self, kind, n, table, intervals, min_pair_mass, seconds=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        d = self._dir(kind, n)
        os.makedirs(d, exist_ok=True)
        lo, hi = min(a for a, _ in intervals), max(b for _, b in intervals)
        name = f"part-{lo:.6f}-{hi:.6f}-{time.time_ns()}.parquet"
        cols = {}
        for k, v in table.items():
            v = np.asarray(v)
            cols[k] = pa.array(v.astype(str)).dictionary_encode() if v.dtype == object else pa.array(v)
        tab = pa.table(cols).replace_schema_metadata({"format": FORMAT, "kind": kind, "n": str(n),
                                                      "numeric_epoch": str(obd_core.NUMERIC_EPOCH),
                                                      "obd_core_version": _core_version()})
        floats = [c for c in tab.column_names if pa.types.is_floating(tab.schema.field(c).type)]
        path = os.path.join(d, name)
        pq.write_table(tab, path + ".tmp", compression="zstd",
                       use_dictionary=[c for c in tab.column_names if c not in floats],
                       column_encoding={c: "BYTE_STREAM_SPLIT" for c in floats})
        os.replace(path + ".tmp", path)
        cov = self.coverage(kind, n)
        cov.setdefault("format", FORMAT)
        cov["parts"].append({"file": name, "intervals": [list(map(float, iv)) for iv in intervals],
                             "min_pair_mass": float(min_pair_mass or 0.0), "rows": tab.num_rows,
                             "numeric_epoch": obd_core.NUMERIC_EPOCH, "obd_core_version": _core_version(),
                             "seconds": seconds, "written_at": _now()})
        self._write_json(os.path.join(d, "coverage.json"), cov)

    def _read(self, kind, n, wins, min_pair_mass, columns):
        import pyarrow.parquet as pq

        d = self._dir(kind, n)
        pieces = []
        for p in sorted(self._usable(kind, n, min_pair_mass), key=lambda p: p["intervals"][0][0]):
            if not any(a <= hi and b >= lo for a, b in p["intervals"] for lo, hi in wins):
                continue
            t = pq.read_table(os.path.join(d, p["file"]), columns=None if columns is None else
                              sorted(set(columns) | {"i", "j", "pstar"}))
            pieces.append({k: t[k].to_numpy(zero_copy_only=False) for k in t.column_names})
        if not pieces:
            return None
        out = {k: np.concatenate([q[k] for q in pieces]) for k in pieces[0]}
        keep = np.zeros(out["pstar"].size, bool)
        for lo, hi in wins:
            keep |= (out["pstar"] >= lo) & (out["pstar"] <= hi)
        out = {k: v[keep] for k, v in out.items()}
        # drop pairs stored twice (a tie point on the shared edge of two parts), then order by p* as
        # tie_table does; the sort is stable, so rows with equal p* keep tie_table's own order
        key = out["i"].astype(np.int64) * (int(n) + 1) + out["j"].astype(np.int64)
        first = np.sort(np.unique(key, return_index=True)[1])
        order = first[np.argsort(out["pstar"][first], kind="stable")]
        out = {k: v[order] for k, v in out.items()}
        for k, v in out.items():
            if v.dtype.kind in "OUS":
                out[k] = v.astype(object)
        if columns is not None:
            out = {k: out[k] for k in columns}
        return out

    # ------------------------------------------------------------ the requests

    def _gaps_of(self, kind, n, wins, min_pair_mass):
        have = self.covered(kind, n, min_pair_mass)
        return _merge([g for lo, hi in wins for g in _gaps(lo, hi, have)])

    def _fill(self, kind, n, wins, min_pair_mass, workers, pool, save_ties):
        gaps = self._gaps_of(kind, n, wins, min_pair_mass)
        if not gaps:
            return
        if kind in ("cusps", "positions"):                     # derive from stored ties where they cover
            have = self.covered("ties", n, min_pair_mass)
            for g in list(gaps):
                if not _gaps(g[0], g[1], have):
                    t = self._read("ties", n, [g], min_pair_mass, None)
                    self._save(kind, n, self._derive(kind, t), [g], min_pair_mass)
                    gaps.remove(g)
            if not gaps:
                return
        t0 = time.time()
        if kind == "positions":
            lnC = obd_core.lnC_arr(n)
            parts = [obd_core.screen(n, collect_all=True, lnC=lnC, p_range=tuple(g),
                                     min_pair_mass=min_pair_mass or None) for g in gaps]
            pos = {k: np.concatenate([q[k] for q in parts]) for k in ("i", "j", "pstar")}
            o = np.argsort(pos["pstar"], kind="stable")
            self._save("positions", n, {k: v[o] for k, v in pos.items()}, gaps, min_pair_mass,
                       round(time.time() - t0, 3))
            return
        whole = gaps == [[0.5, 1.0]]
        t = obd_core.tie_table(n, workers=workers, pool=pool,
                               p_range=None if whole else [tuple(g) for g in gaps],
                               min_pair_mass=None if whole or not min_pair_mass else min_pair_mass)
        secs = round(time.time() - t0, 3)
        if kind == "ties" or save_ties:
            self._save("ties", n, t, gaps, min_pair_mass, secs)
        have = self.covered("cusps", n, min_pair_mass)
        if kind == "cusps" or any(_gaps(a, b, have) for a, b in gaps):
            self._save("cusps", n, self._derive("cusps", t), gaps, min_pair_mass, secs)

    @staticmethod
    def _derive(kind, t):
        if kind == "cusps":
            return {k: v[t["is_cusp"]] for k, v in t.items()}
        axis = (t["pstar"] == 0.5) & (t["i"] == 0)               # positions come from screen: no axis row
        return {k: np.asarray(t[k])[~axis] for k in ("i", "j", "pstar")}

    def ties(self, n: int, p_range=None, columns=None, min_pair_mass: float = 0.0, workers: int = 1, pool=None,
             save: bool = True) -> dict:
        """tie_table(n, p_range=...) rows, read from the store; what is missing is computed and saved.

        p_range: None (all of [1/2, 1]), (lo, hi), or a list of such windows (their union), with
        1/2 <= lo <= hi <= 1.  columns: a subset to read.  min_pair_mass as in tie_table (not proved
        to keep every cusp); parts searched with a threshold are used only by requests with at least
        that threshold.  save=False computes without writing (and still reads what is stored)."""
        wins = self._range(p_range)
        if save:
            self._fill("ties", n, wins, min_pair_mass, workers, pool, True)
            return self._read("ties", n, wins, min_pair_mass, columns)
        if not self._gaps_of("ties", n, wins, min_pair_mass):
            return self._read("ties", n, wins, min_pair_mass, columns)
        t = obd_core.tie_table(n, workers=workers, pool=pool, p_range=None if wins == [[0.5, 1.0]] else
                               [tuple(w) for w in wins], min_pair_mass=min_pair_mass or None)
        return t if columns is None else {k: t[k] for k in columns}

    def cusps(self, n: int, p_range=None, columns=None, min_pair_mass: float = 0.0, workers: int = 1, pool=None,
              save_ties: bool = False) -> dict:
        """The rows of tie_table(n, p_range=...) with is_cusp, from the store; missing ones computed and saved.

        Stored tie tables are used when they cover the range.  Otherwise tie_table is computed and
        only its cusps are kept, unless save_ties (a full table is ~n^2/4 rows: about 1 GB at n = 8000)."""
        wins = self._range(p_range)
        self._fill("cusps", n, wins, min_pair_mass, workers, pool, save_ties)
        return self._read("cusps", n, wins, min_pair_mass, columns)

    def positions(self, n: int, p_range=None, min_pair_mass: float = 0.0) -> dict:
        """{i, j, pstar} of every tie point of n with p* in p_range (the axis excluded), sorted by p*.

        Positions only, no verdicts: computed by the screening kernel (screen(collect_all=True)),
        which skips certification, so wide windows at large n cost a fraction of a tie table.  When
        stored tie tables cover the whole request they are read instead, and nothing is saved (it
        would only duplicate them).  The p* values are those of tie_table."""
        wins = self._range(p_range)
        if not self._gaps_of("ties", n, wins, min_pair_mass):
            return self._derive("positions", self._read("ties", n, wins, min_pair_mass, ["i", "j", "pstar"]))
        self._fill("positions", n, wins, min_pair_mass, 1, None, False)
        return self._read("positions", n, wins, min_pair_mass, None)

    @staticmethod
    def _range(p_range):
        if p_range is None:
            wins = [(0.5, 1.0)]
        elif len(p_range) == 2 and np.ndim(p_range[0]) == 0:
            wins = [p_range]
        else:
            wins = list(p_range)
        wins = [(float(lo), float(hi)) for lo, hi in wins]
        if not wins or not all(0.5 <= lo <= hi <= 1.0 for lo, hi in wins):
            raise ValueError("the store holds p* >= 1/2: every window must satisfy 1/2 <= lo <= hi <= 1")
        return _merge(wins)

    # ------------------------------------------------------------ checking

    def check(self, kind: str = "ties", ns=None, workers: int = 1, pool=None, reference_sample: int = 3,
              verbose: bool = False) -> dict:
        """Recompute stored n with the running OBD-core and compare every stored row bit for bit.

        Also spot-checks reference_sample rows per n against the independent reference.  Appends
        the result to STORE.json and returns it.  Parts from another epoch are checked too: if they
        agree they are re-stamped with the current epoch (the data is still right); if not, they are
        reported and left refused."""
        from obd_core import reference

        ns = self.ns(kind) if ns is None else list(ns)
        failures, checked, restamped = [], 0, 0
        for n in ns:
            cov = self.coverage(kind, n)
            ok_n = True
            for p in cov["parts"]:
                saved = self._read_part(kind, n, p)
                mpm = p.get("min_pair_mass") or None
                if kind == "positions":                        # recompute the way positions are made
                    lnC = obd_core.lnC_arr(n)
                    parts = [obd_core.screen(n, collect_all=True, lnC=lnC, p_range=tuple(iv), min_pair_mass=mpm)
                             for iv in p["intervals"]]
                    t = {k: np.concatenate([q[k] for q in parts]) for k in ("i", "j", "pstar")}
                    o, so = np.lexsort((t["j"], t["i"])), np.lexsort((saved["j"], saved["i"]))
                    t, saved = {k: v[o] for k, v in t.items()}, {k: v[so] for k, v in saved.items()}
                else:
                    t = obd_core.tie_table(n, workers=workers, pool=pool, min_pair_mass=mpm,
                                           p_range=[tuple(iv) for iv in p["intervals"]] if p["intervals"] != [[0.5, 1.0]] else None)
                    if kind == "cusps":
                        t = {k: v[t["is_cusp"]] for k, v in t.items()}
                same = set(saved) == set(t) and all(
                    (list(saved[k]) == list(t[k])) if t[k].dtype == object else
                    (saved[k].dtype == t[k].dtype and np.array_equal(saved[k], t[k], equal_nan=True)) for k in t)
                if same and p.get("numeric_epoch") != obd_core.NUMERIC_EPOCH:
                    p["numeric_epoch"], p["restamped_at"] = obd_core.NUMERIC_EPOCH, _now()
                    restamped += 1
                if not same:
                    ok_n = False
                    failures.append({"n": n, "file": p["file"]})
            if restamped:
                self._write_json(os.path.join(self._dir(kind, n), "coverage.json"), cov)
            if ok_n and reference_sample and kind == "ties" and n >= 3:
                full = self._read(kind, n, [[0.5, 1.0]], 0.0, None)
                r = reference.check_tie_table(n, full, sample=reference_sample, seed=n)
                if not (r["verdicts_ok"] and r["within_expected"]):
                    failures.append({"n": n, "reference": {k: r[k] for k in ("verdicts_ok", "within_expected")}})
            checked += 1
            if verbose:
                print(f"check {kind} n = {n}: {'ok' if ok_n else 'DIFFERS'}", flush=True)
        result = {"at": _now(), "kind": kind, "obd_core_version": _core_version(),
                  "numeric_epoch": obd_core.NUMERIC_EPOCH, "n_checked": checked,
                  "n_range": [min(ns), max(ns)] if ns else None, "restamped_parts": restamped,
                  "failures": failures}
        meta = os.path.join(self.root, "STORE.json")
        with open(meta) as f:
            m = json.load(f)
        m.setdefault("checks", []).append(result)
        self._write_json(meta, m)
        return result

    def _read_part(self, kind, n, part):
        import pyarrow.parquet as pq

        t = pq.read_table(os.path.join(self._dir(kind, n), part["file"]))
        out = {k: t[k].to_numpy(zero_copy_only=False) for k in t.column_names}
        return {k: (v.astype(object) if v.dtype.kind in "OUS" else v) for k, v in out.items()}
