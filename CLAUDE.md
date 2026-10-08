# OBD-core: instructions for working on the core

Read [README.md](README.md) first for what the package does and how to call it. This file is the
rules for **changing** it. Anything here reaches two other repositories, one of which holds
research results, so the bar is: prove a change is harmless, or record exactly what it changes.

## What depends on this

| consumer | where | how it installs obd-core | what it calls |
|---|---|---|---|
| ordered-binomial-cusps (research; every result) | `~/git/ordered-binomial-cusps` | **editable**, into its `.venv`: `.venv/bin/pip install -c ~/git/OBD-core/constraints.txt -e ~/git/OBD-core` | `screen`, `certify`, `certify_escalating`, `evaluate`, `tie_table`, `E_at`, `E_half`, `lnC_arr`, `recheck`, the constants `MARGIN GAP TINY TAG_MIN TAG_NOT TAG_CHECK`, and (in `inspect_tie.py`) the internals `_one_tie`, `_err_bounds` |
| OBD / OBDExplorer (visualization only) | `~/git/OBD` | a **pinned release tag**, into the `obd` conda env (`/Users/omgoleus/opt/miniconda3/envs/obd`) via `requirements.txt` | `tie_table`, `E_slopes_at` |

Treat every name in that table as API. Renaming or changing a return shape needs the consumers
updated in the same piece of work.

## Rules

1. **One implementation.** The fast mathematics exists once, here. `_masses` is the only place
   masses are built; `_pstar` the only place p* is computed; `_one_tie` the only place a tie
   point's quantities are computed; `tie_table` is the only bulk table builder (`p_range`
   restricts it to a p window through `window_kernel`, and must stay bit-identical to the full
   table's rows there: `tests/test_window.py`). Do not add a second way to compute any of them in
   this file or in a consumer: extend the existing function.
2. **The reference stays independent.** `obd_core/reference.py` must not import anything from
   `obd_core` (`tests/test_reference.py` enforces this). Its value is that it shares no code with
   the fast kernel, so a bug cannot hide in both. Keep it exact or interval-rigorous: never
   return a plain float as if it were certified.
3. **Kernel changes must be shown harmless, byte for byte.** After any change to `obd_core/__init__.py`
   that the certified pipeline uses (`_masses`, `_pstar`, `_one_tie`, `_err_bounds`, `tie_kernel`, `screen`,
   `certify*`, `axis_point`, `tie_table`), in the cusps repo:
   ```bash
   cd ~/git/ordered-binomial-cusps
   S=$(mktemp -d)
   .venv/bin/python cusps_fast.py --nmin 3 --nmax 600 --workers 8 --out $S
   for n in 1000 1162 2001 3000; do .venv/bin/python cusps_fast.py --nmin $n --nmax $n --workers 8 --out $S; done
   for n in $(seq 3 600) 1000 1162 2001 3000; do f=n$(printf %05d $n).csv; cmp -s $S/$f cusps/$f || echo DIFF $f; done
   ```
   No output means identical. Do the same for `dump_ties.py` (n = 3, 7, 100, 135, 217, 274, 1000,
   1055, 1162, 2001; build references with the old code first, then compare the Parquet bytes).
   n = 1162 and 1055 exercise interval certification. If a change is *meant* to alter output,
   say so in the commit and record it in the cusps repo's `RESEARCH_LOG.md`.
4. **Check new numerics against the reference.** `reference.check_tie_table(n, core.tie_table(n, both_halves=True))`
   must return `verdicts_ok` and `within_expected` (the tests do this for n ≤ 300). If you change
   the error behaviour, re-measure and update `reference.expected_double_error`, whose docstring
   holds the measured table.
5. **One numerical stack.** `NUMERIC_PINS` (in `obd_core/__init__.py`) is the source of truth;
   `constraints.txt` is generated from it (`tests/test_pins.py` checks both, and that the running
   environment matches). To change a pin, change both, reinstall **both** environments with
   `-c constraints.txt`, rerun rule 3, and update OBD's `environment.yml`, which repeats the pins
   for conda.
6. **Never tag a release on failing tests.** `pytest | tail -1 && git tag ...` tags even when tests
   fail, because the pipe returns tail's status. v0.3.0 shipped broken exactly that way. Run
   pytest on its own and gate on its exit code.
7. **CI must be green.** `.github/workflows/tests.yml` runs the whole suite on Linux and macOS
   (Apple Silicon) with the pinned stack on every push to `main`, every tag and every pull
   request. Check it (`gh run list --repo dperlman/OBD-core --limit 3`) before tagging a release.

## Releasing

1. Bump `version` in `pyproject.toml` and the version in the README install line, and add the
   release to `CHANGELOG.md` (say whether it is numerically identical; if not, what changes).
2. Run the tests in both environments, each gated on its own exit code:
   ```bash
   ~/git/ordered-binomial-cusps/.venv/bin/python -m pytest -q -p no:cacheprovider tests
   /Users/omgoleus/opt/miniconda3/envs/obd/bin/python -m pytest -q -p no:cacheprovider tests
   ```
   The `obd` env runs the *installed* release, so to test unreleased code there, install the
   working tree first (`pip install -c constraints.txt .`). Reinstall the tag afterwards.
3. Commit and push `main`; wait for CI on that commit to pass (rule 7); then
   `git tag -a vX.Y.Z -m "..."` and `git push origin vX.Y.Z` (CI runs on the tag too).
4. Consumers: bump the tag in OBD's `requirements.txt`, `pyproject.toml` and `environment.yml`
   (on a branch, with a PR; OBD uses PRs), then reinstall in the `obd` env:
   `pip install -c <constraints URL at the tag> "obd-core @ git+https://github.com/dperlman/OBD-core.git@vX.Y.Z"`.
   In the cusps repo, bump the tag in `README.md` (commits go straight to `main` there); its
   `.venv` is editable and already runs the working tree.
5. Commit messages end with the attribution line the session asks for.

## Numerical facts worth knowing before touching the kernel

- Masses are built by a recurrence outwards from the mode, only inside a window of masses at or
  above `TINY = 1e-290`, then normalised by their own (Neumaier) sum. Outside the window they
  are exactly 0. Buffers (`f`, `w`) are reused across tie points, so **never read a buffer entry
  that was not written for the current tie point**. That was the 2026-10-06 bug: `f[j] = f[i]` with
  i outside the window. The regression test fills the buffer with junk and checks the result does
  not change.
- `S_-` is a sum of large terms that cancel, so descriptive slopes lose digits as n grows (see
  `expected_double_error`). Verdicts are protected by the certification route, not by the doubles.
- Never derive the slope jump as a difference of slopes; it comes from `ln_fi`.
- numba caches compiled code in `obd_core/__pycache__` (`cache=True`), keyed by numba version.
  If you ever import the module under a synthetic name (importlib `spec_from_file_location`),
  delete the `.nbi`/`.nbc` files afterwards.

## History

`binom_core.py` in ordered-binomial-cusps, until 2026-10-05; then `obd_core.py` here, with that
history carried over; then the package `obd_core/` from v0.4.0 (2026-10-07), which added
`reference.py`. `git log --follow obd_core/__init__.py` shows the whole line. Research-relevant
changes are recorded in the cusps repo's `RESEARCH_LOG.md`.
