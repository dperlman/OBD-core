# Changelog

Each release is a git tag; consumers install a tag (see the README). Only notable changes are
listed. "Numerically identical" means the fast code's output is byte-for-byte unchanged (checked
against ordered-binomial-cusps' tables and dumps).

## Known problems in older releases

**Use v0.3.2 or later.** Every release before it can return wrong numbers:

| releases | problem | fixed in |
|---|---|---|
| v0.1.0 – v0.3.1 | **Buffer bug.** `_one_tie` could copy a never-written buffer entry into the mass window, giving a tie point junk E, S₋ and slopes. It affects rare tie points (16 for n ≤ 1000, about 1300 per n at n = 5000) and only when the reused buffer happens to hold a non-zero value there, so it is silent and memory-dependent. Cusp verdicts are never affected. The bug predates this repository: it was in `binom_core.py` since the TINY window was added (2026-09-21). | v0.3.2 |
| v0.3.0 | `E_slopes_at` wrong at thousands of tie points at n = 1000 | v0.3.1 |
| v0.1.0 – v0.2.0 | n = 2's tie point, where S₋ is exactly 0, is left `UNRESOLVED` (reported as not a cusp, which is correct, but unproved) | v0.2.1 |

The tags are kept because published tags are not rewritten. Nothing pins any of them.

## v0.5.0 (2026-10-07)

- `reference.check_invariants(n, table)`: bounds that follow from the definitions, checked on
  **every** row in milliseconds: 0 ≤ E ≤ n, p increasing, slope_right ≥ slope_left, log₁₀ D finite,
  |E′| ≤ n^1.5/(2√(pq)), and E Lipschitz between neighbouring tie points. Unlike a sampled
  reference comparison, it catches a single corrupt row; it flags the buffer bug's n = 978
  corruption (seen in a v0.2.1 run) at that row.
- Tests: the kernel against the reference on 50 rows at n = 1000; invariants on every row for
  several n, and on an injected copy of the n = 978 corruption.
- CI (GitHub Actions): the test suite on Linux and macOS (Apple Silicon) with the pinned stack,
  on every push to `main`, every tag and every pull request.
- The kernel against `E_slopes_at` comparison test now uses the measured error bound. A
  hand-picked 1e-9 passed on macOS but not on Linux, where the system maths library rounds
  `exp`/`log` differently in the last bits. The docs now say bit-identity under the pins holds
  per platform.
- Numerically identical.

## v0.4.0 (2026-10-07)

- `obd_core` is now a package (`obd_core/__init__.py`); `import obd_core` is unchanged.
- New `obd_core.reference`: an independent rigorous implementation for checking values. Exact
  rationals where p is rational, interval arithmetic otherwise, with precision raised until every
  ranking and sign is proved. Provides `tie`, `axis`, `at`, `compare_tie`, `check_tie_table` and
  `expected_double_error`.
- `recheck()` prints from the reference.
- README rewritten as a usage guide; `CLAUDE.md` holds the rules for changing the core.
- Numerically identical.

## v0.3.3 (2026-10-06)

- `NUMERIC_PINS` and `constraints.txt` pin numpy 2.3.5, numba 0.63.1, llvmlite 0.46.0 and mpmath
  1.4.0 for every environment; `import obd_core` warns on a mismatch.
- Numerically identical.

## v0.3.2 (2026-10-06)

The first release with no known problems.

- **Fix:** `_one_tie` could copy a never-written buffer entry into the mass window. This
  happened when a tie pair is below `TINY` and i falls outside the window while j falls inside.
  It corrupted E and S₋ of that row (e.g. n = 978, pair (1, 978): E = 492 instead of 953.54).
  Cusp verdicts were never affected: such a pair's kink is below 10⁻²⁸⁷. The condition holds at
  16 tie points for n ≤ 1000 and about 1300 per n at n = 5000. ordered-binomial-cusps' existing
  tables and dumps were checked and were unaffected.

## v0.3.1 (2026-10-06): do not use (buffer bug, see above)

- **Fix:** `E_slopes_at` (new in v0.3.0) was wrong at thousands of tie points at n = 1000. A
  fixed 1e-9 tie tolerance merged distinct tie points that sit about 1e-12 apart, and its masses
  were less accurate than the kernel's. It now uses the kernel's masses and a rounding-based
  tolerance.
- `_masses()`: the mass construction, extracted from `_one_tie` so both share it. Numerically
  identical.

## v0.3.0 (2026-10-06): do not use (buffer bug, and `E_slopes_at` wrong)

- Added `E_slopes_at`, E and the exact one-sided slopes at arbitrary p, **with a bug fixed in
  v0.3.1**.

## v0.2.1 (2026-10-05): do not use (buffer bug, see above)

- `certify_exact()`: an exact integer verdict for adjacent pairs, as `tie_table`'s last resort.
  It settles n = 2's tie point, where S₋ is exactly 0, which interval arithmetic cannot.

## v0.2.0 (2026-10-05): do not use (buffer bug, see above)

- `tie_table(n, both_halves=False)`: every tie point of n, certified, with exact slopes and
  log₁₀ D; `both_halves=True` adds the p < ½ mirror half.

## v0.1.0 (2026-10-05): do not use (buffer bug, see above)

- First release: `binom_core.py` from ordered-binomial-cusps, moved here with its history and
  renamed `obd_core.py`.
