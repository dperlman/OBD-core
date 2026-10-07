"""Smoke tests: cusp counts and the p=1/2 axis against the ordered-binomial-cusps tables."""
import numpy as np
import pytest

import obd_core as core


def cusps(n):
    """(i, j) of every cusp with p* > 1/2: screen in double, certify whatever is tagged CHECK."""
    s = core.screen(n)
    out = []
    for i, j, tag in zip(s["i"], s["j"], s["tag"]):
        if tag == core.TAG_MIN:
            out.append((int(i), int(j)))
        elif tag == core.TAG_CHECK:
            verdict, _ = core.certify_escalating(n, int(i), int(j))
            assert verdict is not None, (n, i, j)
            if verdict == "MIN":
                out.append((int(i), int(j)))
    return out


@pytest.mark.parametrize("n, count", [(3, 1), (10, 3), (50, 18), (100, 34), (1000, 352)])
def test_cusp_counts(n, count):
    assert len(cusps(n)) == count


def test_n3_cusp_is_pair_1_3():
    assert cusps(3) == [(1, 3)]
    p = core.evaluate(3, 1, 3)[0]
    assert abs(p - 0.6339745962155614) < 1e-12


def test_slope_jump_is_positive():
    """D = (j-i) f(i) / (p* q*) > 0 at every tie point (no anti-cusps).

    Taken from ln f(i), never as S_+ - S_-: in double S_+ = S_- + kappa loses kappa whenever it is
    far below |S_-|, so the difference reads exactly 0 at many tie points.
    """
    for n in (50, 100, 300):
        s = core.screen(n, collect_all=True)
        assert len(s["i"]) == core.n_ties(n)
        kappa = (s["j"] - s["i"]) * np.exp(s["ln_fi"])
        D = kappa / (s["pstar"] * (1 - s["pstar"]))
        assert np.all(D > 0)


def test_axis_point():
    s_minus, s_plus, E, _, _ = core.axis_point(1000)
    assert s_minus == -s_plus
    assert abs(s_plus / 0.25 - 25.225018178361323) < 1e-9
    assert abs(E - core.E_half(1000)) < 1e-9


@pytest.mark.parametrize("n", [2, 3, 7, 100])
def test_tie_table_both_halves(n):
    t = core.tie_table(n, both_halves=True)
    c = len(t["i"])
    assert c == 2 * core.n_ties(n) + 1
    assert np.all(np.diff(t["pstar"]) > 0)
    # every pair i<j appears once, except the mirror pairs i+j=n, which share the axis row (0, n)
    pairs = set(zip(t["i"].tolist(), t["j"].tolist()))
    expect = {(i, j) for i in range(n + 1) for j in range(i + 1, n + 1) if i + j != n} | {(0, n)}
    assert pairs == expect
    assert np.all(np.isfinite(t["log10_D"]))
    assert np.all(t["slope_right"] >= t["slope_left"])
    # mirror symmetry: E, D and the verdict are symmetric; the slopes swap and change sign
    for k in ("E", "log10_D", "is_cusp", "ln_fi"):
        assert np.array_equal(t[k], t[k][::-1])
    assert np.array_equal(t["slope_left"], -t["slope_right"][::-1])
    axis = c // 2
    assert t["pstar"][axis] == 0.5 and t["is_cusp"][axis] and t["decided_by"][axis] == "symmetry"


def test_tie_table_matches_evaluate():
    n = 100
    t = core.tie_table(n)
    assert len(t["i"]) == core.n_ties(n) + 1
    assert int(t["is_cusp"][1:].sum()) == 34
    lnC = core.lnC_arr(n)
    for r in range(1, len(t["i"]), 97):
        p, E, F3, s_minus, s_plus, sl, sr = core.evaluate(n, int(t["i"][r]), int(t["j"][r]), lnC)
        assert t["pstar"][r] == p and t["E"][r] == E and t["slope_left"][r] == sl
        assert abs(t["slope_right"][r] - sr) <= 1e-12 * max(1.0, abs(sr))
        assert t["is_cusp"][r] == (s_minus < 0 < s_plus)
    # the axis kink: D = 2 S_+ / (1/4)
    s_minus, s_plus, _, _, _ = core.axis_point(n)
    assert abs(10 ** t["log10_D"][0] - 2 * s_plus / 0.25) < 1e-12


def test_tie_table_parallel_matches_serial():
    from multiprocessing import Pool
    n = 300
    a = core.tie_table(n, both_halves=True)
    with Pool(4) as pool:
        b = core.tie_table(n, both_halves=True, workers=4, pool=pool)
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=(a[k].dtype.kind == "f")), k


def test_certify_exact_settles_exact_zero():
    # n=2, pair (1,2) at p*=2/3: S_- is exactly 0, which interval arithmetic cannot settle
    assert core.certify_escalating(2, 1, 2)[0] is None
    assert core.certify_exact(2, 1, 2) == "NOT"
    t = core.tie_table(2, both_halves=True)
    assert t["decided_by"].tolist() == ["exact", "symmetry", "exact"]
    assert t["is_cusp"].tolist() == [False, True, False]


def test_certify_exact_agrees_with_interval_arithmetic():
    for n in (10, 37, 100):
        for i in range(n // 2, n):
            j = i + 1
            if i + j <= n:
                continue
            v, _ = core.certify_escalating(n, i, j)
            assert v is not None and core.certify_exact(n, i, j) == v, (n, i, j)


def test_E_slopes_at_between_ties_matches_central_difference():
    for n in (5, 50, 300):
        for p in (0.13, 0.4321, 0.61, 0.87):
            E, sl, sr = core.E_slopes_at(n, [p])
            assert sl[0] == sr[0]
            assert abs(E[0] - core.E_at(n, p)) < 1e-12 * n
            h = 1e-7
            fd = (core.E_at(n, p + h) - core.E_at(n, p - h)) / (2 * h)
            assert abs(sl[0] - fd) < 1e-5 * max(1.0, abs(fd))


def test_E_slopes_at_tie_points_match_the_kernel():
    for n in (100, 1000):
        t = core.tie_table(n)
        E, sl, sr = core.E_slopes_at(n, t["pstar"])
        assert np.allclose(sl, t["slope_left"], rtol=1e-9, atol=1e-9)
        assert np.allclose(sr, t["slope_right"], rtol=1e-9, atol=1e-9)
        assert np.allclose(E, t["E"], rtol=1e-13)


def test_E_slopes_at_ends_and_last_tie():
    E, sl, sr = core.E_slopes_at(3, [0.0, 0.75, 1.0])
    assert E.tolist() == [3.0, 2.25, 3.0]
    assert sl.tolist() == [-3.0, 0.75, 3.0]   # (2,3) at 3/4: left 0.75, right n (E = n p above it)
    assert sr.tolist() == [-3.0, 3.0, 3.0]
