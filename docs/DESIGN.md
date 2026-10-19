# OBD data store design

How the three repositories divide the work, where every computed number lives, how to check it,
and what is deliberately left undone. As of 2026-10-18.

**This file is the design document.** Change the design here and nowhere else. The page "OBD Data
Store Design" (https://claude.ai/artifact/FgtACNaNpxvsap5rSqFyS7) is a copy made from this file
for easier reading: never edit the page itself; regenerate it from this file after a change. If the
two disagree, this file is right.

A passion project's infrastructure: as much as keeps the data trustworthy and the plots quick, and
no more.

## Goals

- **Trust the data.** Every stored table is right for the OBD-core in use, and you are told plainly
  when one is not.
- **One place, one answer.** All computed tie and cusp data in one store, made by one piece of code,
  and one command to see what it holds.
- **Never compute twice.** Anything computed once is saved and read back, so redoing a plot is
  faster than doing it.
- **No machinery.** No CI-style automation, no scheduled jobs, nothing that republishes itself.
  Things happen when you ask.

## Three repositories

| repo | role | rule |
|---|---|---|
| **OBD-core** (`~/git/OBD-core`) | the mathematics (tie points, cusp verdicts, E and its slopes), the independent rigorous `reference`, and the store: its code, format and tools | code that computes a tie point, a verdict, a slope or E lives here and only here; tagged releases, the others pin one (v0.8.1) |
| **OBD** (`~/git/OBD`) | plots, animations, explorers, the gallery in `docs/plots` | reads data only through the store; keeps only pictures and render-side caches |
| **ordered-binomial-cusps** (`~/git/ordered-binomial-cusps`) | FACTS, RESEARCH_LOG, proofs, analysis scripts, the committed findings in `public/` | uses data, does not produce it (since 2026-10-12) |

OBD and the research repo depend on OBD-core and never on each other. When the research log refers
to a plot, it names the image in OBD's `docs/plots`, whose embedded settings record how it was made.

Locally, both consumers run OBD-core from its checkout (`pip install -e ~/git/OBD-core`), not from
the pinned tag; the pin is what a fresh install gets. So keep the checkout at the pinned release
(or doc-only commits past it) unless deliberately trying a change.

## Where the data lives

Everything computed is in `~/obd-data` (or `$OBD_DATA`), outside every repository, 114 GB
(2026-10-18; `tools/store_summary.py` gives the current figures):

| product | what | holds |
|---|---|---|
| `ties` | full tie tables: every tie point of n with p* ≥ ½, all `tie_table` columns | 1,564 n: every n from 2 to 1000, and 564 larger n up to 10,000 (2000, 3000, …, 8000 and the n the animations use) (114 GB) |
| `cusps` | the certified cusps of n, same columns | every n from 2 to 5000, plus 81 larger n up to 10,000 (0.3 GB) |
| `positions` | i, j, p* only (no verdicts), in narrow p-windows | 99,000 n up to 100,000 (0.1 GB) |

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
Caches that exist, all in OBD and disposable: animation frames (`plots/pair-map-frames/`, 1.4 GB,
keyed by the drawing settings and a `STYLE` version), the graph data and `tie_cusps.parquet`.

## Status

1. Done: the store (v0.7.0, v0.8.0); the data moved in and checked; OBD reads only the store and its
   old copies are retired; rule 3 runs against the store.
2. Done: the research scripts read the store. Before/after over all 19: 16 identical (plots pixel
   for pixel); the rest differ only in that 26 cusps are labelled "double" rather than "iv50", and
   the collision check covers 1,046 n instead of 17 (its RESEARCH_LOG, 2026-10-12). Its `cusps/`,
   `data/`, `cusps_fast.py`, `dump_ties.py` and `validate_trigger.py` are retired: the store is
   the only copy.
3. Done: `tools/store_summary.py`.
4. Done (2026-10-18): this file made the only copy of the design (the page is regenerated from it),
   with the related documents listed below; the research repo's descriptions of its retired data
   marked as superseded, pointing here.
5. Next: back to the mathematics and the pictures.

## Not planned unless needed

| item | why not now | worth it if |
|---|---|---|
| a render cache for still images | plots take seconds to tens of seconds once data is stored | a plot becomes slow to draw from stored data, or mass restyles become routine |
| graph data into the store | feeds only OBD's older explorers; rebuilds in a minute; builds its masses in OBD (a second copy of that mathematics, for drawing only) | those explorers matter again, or another repo needs E on a grid |
| auditing research scripts for their own mathematics | e.g. `analyze_cusps.py` computes p* with its own formula for neighbour gaps; not checked against OBD-core bit for bit; reading the store instead would need tie tables for every n ≤ 5000 (~500 GB) | a result rests on such a script and must be beyond doubt |
| sorting the research repo's plotting folders, archiving OBD's legacy scripts, renaming the research repo | tidiness only | it gets in the way of finding things |
| a slimmer tie-table format, or evicting old tables | 114 GB used, about 1 TB free | dense sweeps of large n become routine |
| publishing the data | already documented and versioned | there is a paper or a reader |

## Related documents

Each holds one part of the design; this file is the overview, and they are not copied here.

| document | holds | where |
|---|---|---|
| OBD-core `CLAUDE.md` | the rules for changing the core: tests, `tools/regression_check.py` (rule 3), `NUMERIC_EPOCH`, tagging a release | this repo |
| OBD-core `README.md` | how to use the API, the store and `reference`, and the pitfalls | this repo |
| `~/obd-data/README.md` | the store's on-disk format, for anyone who finds the folder | the store; written by hand on 2026-10-11, not in any repository |
| ordered-binomial-cusps `CLAUDE.md`, "Data publication policy" | what the public research repo commits (`public/`: per-n results that may grow, samples that are pinned), size limits enforced by `make_public_data.py`, no git-lfs | the research repo |
| ordered-binomial-cusps `RESEARCH_LOG.md` section 7 | the numerical rules for tie-table columns: masses normalised by their sum; the kink taken from `ln_fi`, so `slope_right = slope_left + D`, never `S_plus / (p q)` | the research repo |
| OBD `OBD_data_map.md` | OBD's files and formats: the graph data, `tie_cusps.parquet`, the tie-table columns as OBD's loaders serve them from the store; the retired pathways kept for the record | OBD |
| OBD `README.md` | the plot commands and the gallery | OBD |

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
- 2026-10-17: tie tables up to n = 10,000 for the 30 fps animations (231 new n, about 60 GB, computed
  and saved automatically). `Store.check` on three of them flagged n = 10,000 against the reference;
  the data was right and the reference's E tolerance too tight for large n, recalibrated in v0.8.1.
- 2026-10-17..18: 287 more n of tie tables for the log, square-root and colour-scale animations
  (about 36 GB), saved automatically.
- 2026-10-18: this file made the single source; until then it and the page were two hand-kept copies,
  and the page had drifted (a Goals section the file lacked, an old frame-cache size).
- Correction: `cusps_fast.py` was once described as an independent implementation. It drives
  OBD-core's own kernel; the independent check is `obd_core.reference`.
