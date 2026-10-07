"""obd_core.reference: rigorous values, and agreement of the fast kernel with them."""
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import obd_core as core
from obd_core import reference as ref
from obd_core.reference import Value


def test_reference_is_independent_of_the_kernel():
    # The reference must share no code with the fast implementation, or a bug could hide in both.
    src = Path(ref.__file__).read_text()
    code = re.sub(r'"""[\s\S]*?"""', "", src)                 # ignore docstrings
    assert not re.search(r"^\s*(from|import)\s+(\.|obd_core)", code, re.M)


def test_exact_zero_at_n2():
    r = ref.tie(2, 1, 2)
    assert r.method == "exact"
    assert r.S_minus.exact and r.S_minus.lo == 0
    assert r.is_cusp is False
    assert r.p.lo == Fraction(2, 3)


def test_axis_is_exact_and_matches_the_kernel():
    for n in (3, 10, 1000):
        r = ref.axis(n)
        assert r.method == "exact" and r.is_cusp
        s_minus, s_plus, E, _, _ = core.axis_point(n)
        assert r.slope_left.rel_err(s_minus / 0.25) < 1e-13
        assert r.E.rel_err(E) < 1e-14
        assert r.D.exact and r.slope_right.lo == -r.slope_left.lo


def test_catches_the_n978_buffer_bug():
    # obd_core before v0.3.2 could give E = 492 here; the true value is 953.5447...
    r = ref.tie(978, 1, 978)
    assert abs(float(r.E) - 953.5446930286613) < 1e-9
    assert not r.E.contains(492.0)
    t = core.tie_table(978)
    k = np.flatnonzero((t["i"] == 1) & (t["j"] == 978))[0]
    res = ref.compare_tie(978, 1, 978, E=t["E"][k], slope_left=t["slope_left"][k], is_cusp=t["is_cusp"][k])
    tol = ref.expected_double_error(978)
    assert res["E"] < tol["E"] and res["slope_left"] < tol["slopes"] and res["is_cusp"]


@pytest.mark.parametrize("n", [3, 10, 57, 100, 300])
def test_kernel_tie_table_within_expected_error(n):
    t = core.tie_table(n, both_halves=True)
    res = ref.check_tie_table(n, t, sample=40, seed=n)
    assert res["verdicts_ok"]
    assert res["within_expected"], res["max"]


def test_certified_cusp_from_interval_arithmetic():
    # n=1162 has a cusp that the double screen could not decide (certified by iv50 in the tables)
    t = core.tie_table(1162)
    rows = np.flatnonzero(np.char.startswith(t["decided_by"].astype(str), "iv"))
    assert rows.size >= 1
    for k in rows:
        r = ref.tie(1162, int(t["i"][k]), int(t["j"][k]))
        assert r.is_cusp == bool(t["is_cusp"][k])


def test_interval_route_agrees_with_exact_route():
    # an adjacent pair has a rational tie point: compute it both ways
    n, i, j = 200, 120, 121
    exact = ref.tie(n, i, j)
    assert exact.method == "exact"
    x = ref._interval_eval(n, 50, ("pair", i, j), equal_pair=(i, j))
    assert ref._from_iv(x["S_left"]).contains(exact.S_minus.lo)
    assert ref._from_iv(x["E"]).contains(exact.E.lo)


def test_precision_escalates_until_the_ranking_is_proved():
    # n=1000, (480,979): pair (723,806) has its own tie point ~2e-12 away, so their masses differ by
    # ~1e-9 relative -- unresolvable at 8 digits, so the reference must raise its precision.
    lo = ref.tie(1000, 480, 979, dps=8)
    hi = ref.tie(1000, 480, 979, dps=50)
    assert int(lo.method.split("@")[1]) > 8
    # a lower-precision answer is wider, but still a rigorous enclosure of the true value
    for f in ("E", "S_minus", "slope_left", "slope_right"):
        assert getattr(lo, f).contains(getattr(hi, f).mid), f
    assert lo.is_cusp == hi.is_cusp


def test_at_exact_ties_and_limits():
    a = ref.at(3, 0.75)                       # tie point of (2,3), and the last one: E = 3p above
    assert a.method == "exact" and a.tied == [(2, 3)]
    assert float(a.slope_left) == 0.75 and float(a.slope_right) == 3.0
    z = ref.at(5, 0)
    assert z.method == "limit" and float(z.E) == 5 and float(z.slope_left) == -5
    h = ref.at(10, "1/2")
    assert h.tied and float(h.slope_right) == -float(h.slope_left)


def test_at_matches_E_slopes_at():
    for n, p in ((50, 0.37), (300, 0.6123), (1000, 0.55)):
        E, sl, sr = core.E_slopes_at(n, [p])
        a = ref.at(n, p)
        tol = ref.expected_double_error(n)
        assert a.E.rel_err(E[0]) < tol["E"]
        assert a.slope_left.rel_err(sl[0]) < tol["slopes"]
        assert a.slope_right.rel_err(sr[0]) < tol["slopes"]


def test_value_helpers():
    v = Value(Fraction(1, 3), Fraction(1, 3))
    assert v.exact and v.contains(Fraction(1, 3)) and not v.contains(0.3333333333333333)
    assert v.rel_err(1 / 3) < 1e-16
    assert Value(Fraction(-1), Fraction(-1)).sign() == -1


@pytest.mark.parametrize("n", [2, 3, 10, 100, 978, 1000])
def test_invariants_hold_on_every_row(n):
    t = core.tie_table(n, both_halves=True)
    res = ref.check_invariants(n, t)
    assert res["ok"], res


def test_invariants_catch_the_n978_corruption():
    # the row obd_core v0.3.1 and earlier could produce from a stale buffer (E = 492, slopes 958972)
    t = core.tie_table(978, both_halves=True)
    k = np.flatnonzero((t["i"] == 1) & (t["j"] == 978))[0]
    for col, val in (("E", 492.0), ("slope_left", 958972.78), ("slope_right", 958972.78)):
        t[col] = t[col].copy()
        t[col][k] = val
    res = ref.check_invariants(978, t)
    assert not res["ok"]
    assert res["first_bad_row"]["slope_bound"] == k and res["first_bad_row"]["E_lipschitz"] == k


def test_kernel_at_n1000_against_the_reference():
    # 50 random rows of both halves at n = 1000 (plus the first and last), checked rigorously
    t = core.tie_table(1000, both_halves=True)
    res = ref.check_tie_table(1000, t, sample=50, seed=2026)
    assert res["verdicts_ok"]
    assert res["within_expected"], res["max"]
