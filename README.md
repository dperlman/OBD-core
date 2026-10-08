# OBD-core

[![tests](https://github.com/dperlman/OBD-core/actions/workflows/tests.yml/badge.svg)](https://github.com/dperlman/OBD-core/actions/workflows/tests.yml)

The core computational functions for the ordered binomial distribution, shared by
[ordered-binomial-cusps](https://github.com/dperlman/ordered-binomial-cusps) (the research: every
proved and screened result lives there) and [OBDExplorer](https://github.com/dperlman/OBDExplorer)
(visualization only). This package is the **only** implementation of the mathematics. Neither
repo re-derives any of it, so a fix here reaches both.

It has two parts:

| | what | precision | use it for |
|---|---|---|---|
| `obd_core` | the fast implementation: numba kernel, certified cusp verdicts, exact slopes | double (verdicts certified) | producing tables, plots, anything in bulk |
| `obd_core.reference` | an independent rigorous implementation | exact rationals or interval arithmetic | checking `obd_core`, investigating one value, tests |

Working **on** this package (changing it, releasing it)? Read [CLAUDE.md](CLAUDE.md) first. What
changed in each release: [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
pip install -c https://raw.githubusercontent.com/dperlman/OBD-core/v0.6.0/constraints.txt \
    "obd-core @ git+https://github.com/dperlman/OBD-core.git@v0.6.0"
```

The `-c constraints.txt` pins numpy, numba, llvmlite and mpmath to exact versions (see
[One numerical stack](#one-numerical-stack)). For development, clone the repo and run
`pip install -c constraints.txt -e ".[test]"`, then `pytest`.

## Concepts in one minute

- **Masses.** f(k) = C(n,k) pᵏ (1−p)ⁿ⁻ᵏ for k = 0..n.
- **Ranks.** w_k is the rank of f(k) among the n+1 masses, 0 = smallest. **E(n,p)** = Σ w_k f(k), the
  "ordered binomial" expectation. E is continuous, symmetric (E(p) = E(1−p)), concave between tie
  points, with a convex kink at each.
- **Tie point.** p\*(n,i,j) is where f(i) = f(j) (0 ≤ i < j ≤ n): ρ = p\*/(1−p\*) = (C(n,i)/C(n,j))^(1/(j−i)).
  At p = ½ every mirror pair (k, n−k) ties at once; that is the **axis**.
- **Slopes.** E′₋ = S₋/(p q), with S₋ = Σ w_k f(k)(k − np) using the ranking just *left* of p\* (where
  f(j) < f(i)). E′₊ = E′₋ + D, where **D** = (j−i) f(i)/(p q) > 0 is the slope jump (the kink).
- **Cusp.** A tie point is a cusp (a local minimum of E) iff S₋ < 0 < S₊. No tie point is ever a
  local maximum.
- **Halves.** Most functions work on p\* > ½ (i + j > n). The other half is the mirror image:
  (i, j) at p\* ↦ (n−j, n−i) at 1−p\*, with the same E, D and verdict, and the slopes swapped and
  negated.

## `obd_core`: the fast implementation

```python
import numpy as np
import obd_core as core

t = core.tie_table(1000, both_halves=True)   # every tie point of n = 1000, sorted by p*
t["pstar"], t["slope_left"], t["log10_D"], t["is_cusp"]
w = core.tie_table(20000, p_range=(0.6, 0.600002))   # only the tie points in a p window: 0.08 s
E, sl, sr = core.E_slopes_at(1000, np.linspace(0, 1, 1001))   # E and slopes on any grid
```

| function | returns | notes |
|---|---|---|
| `tie_table(n, both_halves=False, workers=1, pool=None, p_range=None, min_pair_mass=None)` | dict of arrays, one row per tie point, sorted by p\* | **The main entry point.** Columns: `i, j, pstar, ln_fi, E, S_minus, F3, tag, n_tied_pairs, is_cusp, decided_by, slope_left, slope_right, log10_D`. Row 0 of the p\* > ½ half is the axis (sentinel pair (0, n), `decided_by="symmetry"`). `both_halves=True` adds the p\* < ½ mirror rows (F3 = NaN there). Pass a `multiprocessing.Pool` and `workers>1` to parallelise; the result is identical. `p_range=(lo, hi)` computes only the tie points with lo ≤ p\* ≤ hi: exactly the full table's rows there, bit for bit, at a cost of about n²(hi − lo) tie points instead of n²/4. `min_pair_mass` (with `p_range`) also skips pairs whose mass f(i) is below it: faster, but **not proved** to keep every cusp (see Pitfalls). |
| `E_slopes_at(n, p_array)` | `(E, slope_left, slope_right)` arrays | E and the exact one-sided slopes at arbitrary p. Equal except at tie points. At p = 0 and 1: E = n, slopes −n / +n. |
| `E_at(n, p)` | float | E at one p (sort-based). |
| `E_half(n)` | float | E(n, ½). |
| `evaluate(n, i, j)` | `(p*, E, F3, S_-, S_+, slope_left, slope_right)` | One tie point, double precision. |
| `axis_point(n)` | `(S_-, S_+, E, kappa, n_pairs)` | The p = ½ tie point. |
| `screen(n, collect_all=False, p_range=None, min_pair_mass=None)` | dict of arrays | The raw screening kernel (p\* > ½, no axis row, CHECK rows not yet certified). Prefer `tie_table`. |
| `window_count(n, lnC_arr(n), lo, hi)` | int | Number of tie points with lo ≤ p\* ≤ hi (p\* > ½), from binary searches alone. |
| `certify(n, i, j, dps)` / `certify_escalating(n, i, j)` / `certify_exact(n, i, j)` | `'MIN'`/`'NOT'`/`None`, plus the route | Proved cusp verdict for one tie point: interval arithmetic, or exact integers for adjacent pairs. `tie_table` already calls these for every borderline row. |
| `recheck(n, i, j, dps=50)` | prints | Every quantity at one tie point, from `reference.tie`. |
| `n_ties(n)`, `lnC_arr(n)` | int, array | Number of tie points with p\* > ½ (excluding the axis); log C(n,k). |
| `NUMERIC_PINS`, `pin_mismatches()` | dict | The pinned numerical stack, and how the running environment differs from it. |

Constants: `TINY = 1e-290` (masses below it are treated as exactly 0), `MARGIN = 1e-6` (double
screen margin on S₋, S₊), `TAG_MIN`, `TAG_NOT`, `TAG_CHECK`. Names starting with `_` (`_one_tie`,
`_masses`), and `tie_kernel`, are internal.

### What is certified and what is not

- **`is_cusp` is proved.** The double-precision screen decides only when S₋ and S₊ are farther
  than a derived error bound from 0. Everything else goes to interval arithmetic (50, 100, 200
  digits) or exact integers. `decided_by` records which route decided each row.
- **The descriptive numbers are plain double precision:** E, S₋, the slopes, p\*. Their
  accuracy is measured, not proved. See `reference.expected_double_error(n)`: about 1e-9
  relative for the slopes at n = 1000 and 1e-8 at n = 2000; E is near full precision.

### Pitfalls

- **Never compute D as `slope_right - slope_left`.** D spans hundreds of orders of magnitude (the
  median at n = 1000 is below double range) while the slopes are O(1)–O(n), so the difference is 0
  or noise. Use `log10_D`, which comes from `ln_fi` and is exact. `slope_right` itself is fine as
  a slope.
- **The axis row is special:** pair (0, n) is a sentinel, its `ln_fi` is defined so that
  (j−i)·exp(ln_fi) is the whole kink, and `F3` is NaN.
- **Masses below `TINY`** contribute exactly 0. A tie point whose pair mass is below it can never
  be a cusp.
- **`min_pair_mass` is a heuristic.** A cusp needs S₋ < 0 < S₋ + (j−i)·f(i), so a light pair is very
  unlikely to be one, but no lower bound on a cusp's pair mass is proved. Measured: the lightest
  cusp pair for n ≤ 5000 has f(i) = 4.4e-9; with 1e-20, 150 random windows at n = 1001–5000 found
  every one of the 2,958 catalogued cusps in them. Without `min_pair_mass` a window is complete.
- **Floats near a tie point:** `E_slopes_at` treats masses within ~64·eps·(n+1)/(pq) of each other
  as tied. Distinct tie points can sit about 1e-12 apart at n = 1000, so a looser tolerance would
  report kinks that are not there.

## `obd_core.reference`: rigorous values for checking

An independent implementation (a test enforces that it imports nothing from the fast code):
exact integer binomials, every mass computed, the ranking built directly. Every number it
returns is a **`Value`**, a rigorous enclosure `lo ≤ true value ≤ hi`:

- **exact** (Fractions) whenever p is rational: adjacent-pair tie points p\* = (i+1)/(n+1), the
  axis, and any float or Fraction passed to `at()`;
- **interval arithmetic** otherwise (tie points with j − i ≥ 2 are irrational). Precision starts
  at `dps` and doubles until every ranking decision and the cusp verdict are proved, up to
  `max_dps` (then `ReferenceUndecided`).

```python
import obd_core as core
from obd_core import reference as ref

r = ref.tie(1000, 480, 979)           # everything at the tie point of (480, 979)
r.method                               # 'interval@50'  (or 'exact')
float(r.slope_left), r.is_cusp         # (31.85082244300127, False)
r.E                                    # Value(979.105165429875699876449 ± 7.61e-45)
r.E.rel_err(979.1051654298757)         # 1.7e-17: how far off a double is (floor 1: small values absolute)
r.E.contains(979.1051654298757)        # False: a double never lands inside a 1e-44-wide enclosure;
                                       # contains() is for exact or reference values, rel_err() for doubles

ref.compare_tie(1000, 480, 979, slope_left=31.850822443183596, is_cusp=False)
# {'slope_left': 5.7e-12, 'is_cusp': True, 'reference': TieRef(...)}   (True = verdicts agree)

t = core.tie_table(1000, both_halves=True)
ref.check_tie_table(1000, t, sample=50)    # worst errors over 50 random rows + verdict check
# {'max': {...}, 'worst_row': {...}, 'verdicts_ok': True, 'within_expected': True, ...}

a = ref.at(3, 0.75)                    # any p: E and both one-sided slopes
a.tied, float(a.slope_left), float(a.slope_right)    # [(2, 3)], 0.75, 3.0   (a tie point)
ref.axis(1000)                         # p = 1/2, exact
ref.expected_double_error(1000)        # how close obd_core's doubles should be
```

| function | returns |
|---|---|
| `tie(n, i, j, dps=50, max_dps=800)` | `TieRef`: `p, E, S_minus, S_plus, slope_left, slope_right, D, log10_D, F3` (Values), `is_cusp`, `method` |
| `axis(n)` | `TieRef` at p = ½ (pair reported as (0, n); D = the whole kink; F3 = None) |
| `at(n, p, dps=50, max_dps=800)` | `PointRef`: `p, E, slope_left, slope_right, D` (Values), `tied` (groups of indices with equal masses at p), `method` |
| `compare_tie(n, i, j, **values)` | `{name: rel_err}` for numbers, `{"is_cusp": agree?}`, `"reference"`: the TieRef |
| `check_tie_table(n, table, rows=None, sample=25)` | worst errors over rows of `obd_core.tie_table(n)`, verdict agreement, `within_expected` |
| `check_invariants(n, table)` | proved bounds (0 ≤ E ≤ n, slope ordering and bound, E continuity, …) on **every** row, in milliseconds; catches a single corrupt row that sampling would miss |
| `expected_double_error(n)` | `{"p", "E", "slopes", "log10_D"}`: the empirical bounds |

`Value`: `.lo`, `.hi`, `.exact`, `.mid`, `.rad` (mpmath numbers), `float(v)`, `.sign()` (+1/−1/0,
or None if the enclosure straddles 0), `.contains(x)`, `.abs_err(x)`, `.rel_err(x, floor=1)`.

**Cost:** about 0.05 s per tie point at n = 1000 by interval arithmetic, and 0.7 s for `at()` at
an arbitrary float p at n = 1000 (exact, big integers). Use it on single points or samples,
never to build tables.

## Recipes

- **A value looks wrong.** Run `ref.tie(n, i, j)` (or `obd_core.recheck(n, i, j)` to print it)
  and compare with the suspect value. A gap beyond `expected_double_error(n)` is a bug, not
  rounding. That is how the n = 978 buffer bug (E = 492 instead of 953.54) was pinned down.
- **Testing new code that produces tie data.** Run `check_invariants` on every table (cheap) and
  `check_tie_table` on a sample; assert `ok`, `verdicts_ok` and `within_expected`.
- **E or the slope on a grid.** `obd_core.E_slopes_at`. Never difference E numerically: E has
  kinks at every tie point, and a finite difference across one measures the curvature, not the slope.
- **Is this tie point a cusp?** `tie_table(n)["is_cusp"]` (proved), or `ref.tie(n, i, j).is_cusp`
  for one point.

## One numerical stack

Every environment that runs `obd_core` should have exactly the versions in
[`constraints.txt`](constraints.txt) (numpy, numba, llvmlite, mpmath). Then results agree to the
last bit across environments on the same platform, and numba's compiled-code cache, which is per
numba version, stays valid. Different platforms (say Linux x86-64 against macOS arm64) still
differ in the last bits, because their maths libraries round `exp` and `log` differently; cusp
verdicts never depend on that. Install with the constraints file, as above. Importing `obd_core` in an environment that
differs warns (`obd_core.pin_mismatches()` lists the differences). The pins live in
`obd_core.NUMERIC_PINS`; `constraints.txt` is generated from it, and a test keeps them equal.
