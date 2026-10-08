"""tie_table(n, p_range=...): exactly the full table's rows in the window, bit for bit."""
import numpy as np
import pytest

import obd_core as core
from obd_core import reference as ref


def _restrict(table, lo, hi):
    keep = (table["pstar"] >= lo) & (table["pstar"] <= hi)
    return {k: v[keep] for k, v in table.items()}


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if a[k].dtype == object:
            assert list(a[k]) == list(b[k]), k
        else:
            assert a[k].dtype == b[k].dtype, k
            assert np.array_equal(a[k], b[k], equal_nan=True), k


def _windows(p, seed):
    rng = np.random.default_rng(seed)
    w = [(0.5, 0.6), (0.0, 1.0), (0.3, 0.55), (0.49, 0.51), (0.5, 0.5), (0.9, 0.95), (0.2, 0.3),
         (0.97, 0.99), (0.66, 0.7)]
    w += [(p[k], p[k]) for k in rng.integers(0, len(p), 4)]               # exactly one tie point
    w += [(p[k], p[k + 1]) for k in rng.integers(0, len(p) - 1, 3)]       # closed at both ends
    w += [tuple(sorted(rng.uniform(0, 1, 2))) for _ in range(6)]
    return w


@pytest.mark.parametrize("n", [2, 3, 4, 7, 50, 135, 300])
@pytest.mark.parametrize("both_halves", [False, True])
def test_window_equals_full_table_restricted(n, both_halves):
    full = core.tie_table(n, both_halves=both_halves)
    for lo, hi in _windows(full["pstar"], n):
        _assert_same(_restrict(full, lo, hi), core.tie_table(n, both_halves=both_halves, p_range=(lo, hi)))


def test_window_with_interval_certification():
    """n = 1162 has rows only interval arithmetic settles; the window must certify them the same way."""
    n = 1162
    full = core.tie_table(n)
    iv = np.flatnonzero(np.char.startswith(full["decided_by"].astype(str), "iv"))
    assert len(iv) > 0
    for t in iv[:3]:
        lo, hi = full["pstar"][t] - 1e-4, full["pstar"][t] + 1e-4
        w = core.tie_table(n, p_range=(lo, hi))
        _assert_same(_restrict(full, lo, hi), w)
        assert any(str(d).startswith("iv") for d in w["decided_by"])


def test_window_count_and_empty_window():
    n = 300
    lnC = core.lnC_arr(n)
    full = core.screen(n, collect_all=True)
    for lo, hi in ((0.5, 1.0), (0.55, 0.6), (0.6, 0.6), (0.999, 1.0)):
        assert core.window_count(n, lnC, lo, hi) == int(np.sum((full["pstar"] >= lo) & (full["pstar"] <= hi)))
    assert core.window_count(n, lnC, 0.5, 1.0) == core.n_ties(n)
    w = core.tie_table(n, p_range=(0.1, 0.2))                              # below 1/2, one half only
    assert len(w["i"]) == 0 and set(w) == set(core.tie_table(3))


def test_window_screen_cusps_only():
    """collect_all=False with a window returns the full screen's MIN/CHECK rows in that window."""
    n = 500
    full = core.screen(n)
    lo, hi = 0.55, 0.6
    w = core.screen(n, p_range=(lo, hi))
    keep = (full["pstar"] >= lo) & (full["pstar"] <= hi)
    assert sorted(zip(full["i"][keep], full["j"][keep])) == sorted(zip(w["i"], w["j"]))


def test_min_pair_mass_skips_only_light_pairs():
    n, lo, hi, tau = 1000, 0.55, 0.62, 1e-20
    full = core.tie_table(n, p_range=(lo, hi))
    w = core.tie_table(n, p_range=(lo, hi), min_pair_mass=tau)
    rows = {(int(i), int(j)): t for t, (i, j) in enumerate(zip(full["i"], full["j"]))}
    for k in w:                                   # every kept row is the full table's row
        t = [rows[(int(i), int(j))] for i, j in zip(w["i"], w["j"])]
        if w[k].dtype == object:
            assert list(w[k]) == list(full[k][t])
        else:
            assert np.array_equal(w[k], full[k][t], equal_nan=True), k
    heavy = full["ln_fi"] > np.log(tau) + 1e-6    # normalisation moves ln f(i) by ~1e-12
    assert set(zip(full["i"][heavy], full["j"][heavy])) <= set(zip(w["i"], w["j"]))
    assert len(w["i"]) < len(full["i"])
    assert set(zip(full["i"][full["is_cusp"]], full["j"][full["is_cusp"]])) <= set(zip(w["i"], w["j"]))
    with pytest.raises(ValueError):
        core.tie_table(n, min_pair_mass=tau)


def test_window_at_large_n_against_the_reference():
    """Far past what a full table can reach quickly: a window at n = 20000, checked rigorously."""
    n = 20000
    w = core.tie_table(n, p_range=(0.6, 0.600002))
    assert len(w["i"]) == core.window_count(n, core.lnC_arr(n), 0.6, 0.600002) > 100
    assert ref.check_invariants(n, w)["ok"]
    c = ref.check_tie_table(n, w, sample=3, seed=1)
    assert c["verdicts_ok"] and c["within_expected"]


@pytest.mark.parametrize("both_halves", [False, True])
def test_several_windows_in_one_call(both_halves):
    """A list of windows (overlapping, unsorted, across 1/2) gives the rows of their union."""
    n = 300
    full = core.tie_table(n, both_halves=both_halves)
    wins = [(0.62, 0.64), (0.55, 0.56), (0.555, 0.57), (0.45, 0.52), (0.9, 0.9001), (0.1, 0.12)]
    keep = np.zeros(len(full["pstar"]), bool)
    for lo, hi in wins:
        keep |= (full["pstar"] >= lo) & (full["pstar"] <= hi)
    _assert_same({k: v[keep] for k, v in full.items()}, core.tie_table(n, both_halves=both_halves, p_range=wins))
    one = [(0.55, 0.6)]
    _assert_same(core.tie_table(n, p_range=(0.55, 0.6)), core.tie_table(n, p_range=one))


def test_tie_table_past_int16():
    """The axis row counts n/2 tied pairs, which overflowed int16 from n = 65,536 (found 2026-10-08)."""
    n = 70000
    t = core.tie_table(n, p_range=[(0.5, 0.5), (1 - 1e-4, 1.0)])
    assert t["n_tied_pairs"][0] == n // 2 and t["n_tied_pairs"].dtype == np.int32
    assert t["pstar"][-1] == pytest.approx(n / (n + 1), abs=1e-12)          # the last tie point, (n-1, n)
