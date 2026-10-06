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
