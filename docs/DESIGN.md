# OBD data store design

How the three repositories divide the work, where every computed number lives, how to check it,
and what is deliberately left undone. Kept up to date by hand; the same content is published as the
page "OBD Data Store Design" (https://claude.ai/artifact/FgtACNaNpxvsap5rSqFyS7). As of 2026-10-12.

A passion project's infrastructure: as much as keeps the data trustworthy and the plots quick, and
no more. No CI-style automation, no scheduled jobs, nothing that republishes itself.

## Three repositories

| repo | role | rule |
|---|---|---|
| **OBD-core** (`~/git/OBD-core`) | the mathematics (tie points, cusp verdicts, E and its slopes), the independent rigorous `reference`, and the store: its code, format and tools | code that computes a tie point, a verdict, a slope or E lives here and only here; tagged releases, the others pin one (v0.8.1) |
| **OBD** (`~/git/OBD`) | plots, animations, explorers, the gallery in `docs/plots` | reads data only through the store; keeps only pictures and render-side caches |
| **ordered-binomial-cusps** (`~/git/ordered-binomial-cusps`) | FACTS, RESEARCH_LOG, proofs, analysis scripts, the committed findings in `public/` | uses data, does not produce it (since 2026-10-12) |

OBD and the research repo depend on OBD-core and never on each other.

## Where the data lives

Everything computed is in `~/obd-data` (or `$OBD_DATA`), outside every repository, 14 GB:

| product | what | holds |
|---|---|---|
| `ties` | full tie tables: every tie point of n with p* ≥ ½, all `tie_table` columns | 1,046 n: every n ≤ 1000; 2000, 3000, …, 8000; the 40 animation n (13 GB) |
| `cusps` | the certified cusps of n, same columns | every n from 2 to 5000, plus the larger n above (0.3 GB) |
| `positions` | i, j, p* only (no verdicts), in narrow p-windows | 99,000 n up to 100,000 (0.5 GB) |

All of it comes from `tie_table` (positions from `screen`) through `obd_core.store`. A full tie table
is about n²/4 rows (about 1 GB at n = 8000), so tie tables are kept for chosen n; cusp tables cover
every n.

Not in the store, on purpose: OBD's graph data (`data/graph_data_p01001.h5`, 2.8 GB, a drawing cache
for the older explorers), OBD's `data/tie_cusps.parquet` (a derived cache built from the store), and
the research repo's `analysis/` and `public/` (findings).

## Using it

```python
from obd_core import store
s = store.Store()                                 # ~/obd-data, or $OBD_DATA
t = s.ties(4000)                                  # read it, or compute with tie_table and save it
w = s.ties(20000, p_range=[(0.6, 0.6002), (0.61, 0.6102)])   # only what is missing is computed
c = s.cusps(4321)                                 # cusp rows; the full table is not kept
x = s.positions(90000, p_range=(0.999, 1.0))      # i, j, p* only
s.check("ties", ns=[1000, 4000])                  # recompute, compare bit for bit, log in STORE.json
```

To see what the store holds, on one screen (12 s):

```bash
python ~/git/OBD-core/tools/store_summary.py
```

In OBD the plots do this themselves; in the research repo `cusps_data.py` gives the cusp catalogue
(`read_cusps`: the columns `cusps_all.csv` had) and per-n tables (`load_ties`, `load_cusps`).

## Trusting it

- Every value comes from `tie_table`, released and tested; the store adds no mathematics.
- `NUMERIC_EPOCH` changes only when some output changes. Each stored part records its epoch; a part
  from another epoch is refused until `Store.check` has compared it.
- `Store.check` recomputes and compares bit for bit, with `reference` spot checks, logged in
  `STORE.json`. `tools/regression_check.py` (CLAUDE.md rule 3) runs a fixed set before any kernel
  change is released.
- The migration compared the store with every older copy: 6,014 comparisons, 0 mismatches
  (`~/obd-data/verification/`).

## Caches

The store makes plots quick (tie tables are read, not computed). Drawing is still done each time.
Caches that exist, all in OBD and disposable: animation frames (`plots/pair-map-frames/`, keyed by
the drawing settings and a `STYLE` version), the graph data and `tie_cusps.parquet`.

## Status

1. Done: the store (v0.7.0, v0.8.0); the data moved in and checked; OBD reads only the store and its
   old copies are retired; rule 3 runs against the store.
2. Done: the research scripts read the store. Before/after over all 19: 16 identical (plots pixel
   for pixel); the rest differ only in that 26 cusps are labelled "double" rather than "iv50", and
   the collision check covers 1,046 n instead of 17 (its RESEARCH_LOG, 2026-10-12). Its `cusps/`,
   `data/`, `cusps_fast.py`, `dump_ties.py` and `validate_trigger.py` are retired: the store is
   the only copy.
3. Done: `tools/store_summary.py`.
4. Next: back to the mathematics and the pictures.

## Not planned unless needed

| item | why not now | worth it if |
|---|---|---|
| a render cache for still images | plots take seconds to tens of seconds once data is stored | a plot becomes slow to draw from stored data, or mass restyles become routine |
| graph data into the store | feeds only OBD's older explorers; rebuilds in a minute; builds its masses in OBD (a second copy of that mathematics, for drawing only) | those explorers matter again, or another repo needs E on a grid |
| auditing research scripts for their own mathematics | e.g. `analyze_cusps.py` computes p* with its own formula for neighbour gaps; not checked against OBD-core bit for bit; reading the store instead would need tie tables for every n ≤ 5000 (~500 GB) | a result rests on such a script and must be beyond doubt |
| sorting the research repo's plotting folders, archiving OBD's legacy scripts, renaming the research repo | tidiness only | it gets in the way of finding things |
| a slimmer tie-table format, or evicting old tables | 14 GB used, 1.1 TB free | dense sweeps of large n become routine |
| publishing the data | already documented and versioned | there is a paper or a reader |

## Format details

```
~/obd-data/
  README.md                          the format
  STORE.json                         format, and the log of every Store.check
  ties/n=NNNNN/coverage.json         parts of n: p intervals, epoch, min_pair_mass, rows
  ties/n=NNNNN/part-LO-HI-T.parquet  tie_table rows, all columns, same dtypes
  cusps/n=NNNNN/...                  the same for rows with is_cusp
  positions/n=NNNNN/...              i, j, pstar (int64, int64, float64), no axis row
  verification/sweep-2026-10-11.csv  the migration's comparisons
```

- Coverage: each part records the closed p intervals it holds completely, `min_pair_mass` (0 =
  complete; a thresholded part serves only requests with at least that threshold), `numeric_epoch`,
  `obd_core_version`, rows. An empty search is a part with `"file": null` and the column dtypes.
- Writing: Parquet (zstd, byte-stream-split floats), written under a temporary name and renamed;
  coverage rewritten by rename only after its part is in place. One process per n at a time.
- Reading: rows in the requested windows, duplicates on shared edges dropped, stable p* order.
  `Store.ties(n, both_halves=True)` adds the p* < ½ half with `tie_table`'s own `_with_mirror_half`.
- Positions come from `screen(collect_all=True)`, uncertified; where stored tie tables cover a request
  they are read from those, and nothing is saved.

## History

- Before 2026-10-11: unchecked copies in two repos: OBD's tie tables (every n ≤ 1000, OBD-core
  0.3.2) and window caches; the research repo's tie tables (17 n) and cusp CSVs (every n ≤ 5000,
  from `cusps_fast.py`). Plots computed tie tables afresh each time.
- 2026-10-11..12: the store built, filled, checked against all of them; OBD switched and its old
  copies retired; rule 3 moved onto the store; the research scripts switched on a branch.
- Correction: `cusps_fast.py` was once described as an independent implementation. It drives
  OBD-core's own kernel; the independent check is `obd_core.reference`.
