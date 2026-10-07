"""obd_core.reference -- rigorous reference values, for CHECKING obd_core, never for building tables.

What it is
    A second implementation of the same mathematics that shares NO code with the fast kernel in
    obd_core: exact integer binomials (math.comb), every mass computed (no TINY window), the
    ranking built directly, no normalisation needed.  Every number it returns is a rigorous
    enclosure (a ``Value`` with ``lo <= true value <= hi``):

    * exact rational arithmetic whenever p is rational -- an adjacent pair's tie point
      p* = (i+1)/(n+1), the axis p = 1/2, and any float, int or Fraction you pass to ``at()``;
    * otherwise interval arithmetic (mpmath.iv) -- the tie point of a pair with j - i >= 2 is
      irrational -- with the working precision doubled automatically until every ranking decision
      and the cusp verdict are proved, up to ``max_dps``.

    So these are certified values, not merely high-precision ones.  Independence is the point: a
    bug in the fast kernel cannot hide in here (the 2026-10-06 buffer bug, which gave E = 492
    instead of 953.54 at n = 978, pair (1,978), shows up as a mismatch against ``tie(978, 1, 978)``).

When to use it
    * Tests of obd_core: compare its double-precision output with ``compare_tie`` /
      ``check_tie_table``; ``expected_double_error(n)`` says how close it should be.
    * Investigating one suspicious row: ``tie(n, i, j)`` gives every quantity at that tie point.
    * Settling what double precision and certify() cannot: S_- = 0 exactly (n = 2, pair (1,2)).
    Cost: ~0.05-0.5 s per point at n = 1000 (exact or interval at 50 digits), growing roughly as
    n^2 for exact arithmetic at arbitrary float p.  Never loop it over a whole table.

Definitions (same as obd_core; see its module docstring)
    f(k) = C(n,k) p^k q^(n-k);  w_k = rank of f(k) among the n+1 masses, 0 = smallest;
    E = sum_k w_k f(k).  E'(p) = S / (p q) with S = sum_k w_k f(k) (k - n p), the ranking taken
    just LEFT of p for E'_- and just RIGHT for E'_+ (they differ only where p is a tie point).
    Masses that are exactly equal at p are ranked by the one-sided limit: f_l / f_k grows with p
    for l > k, so just left of a tie the larger index has the smaller mass.

API
    tie(n, i, j, dps=50, max_dps=800) -> TieRef     everything at the tie point of pair (i, j)
    axis(n) -> TieRef                               the tie point p = 1/2 (all mirror pairs tie)
    at(n, p, dps=50, max_dps=800) -> PointRef       E and both one-sided slopes at any p
    compare_tie(n, i, j, **values) -> dict          relative errors of your values vs the reference
    check_tie_table(n, table, rows=None) -> dict    the same over rows of obd_core.tie_table(n)
    check_invariants(n, table) -> dict              proved bounds checked on EVERY row (fast)
    expected_double_error(n) -> dict                how close obd_core's double values should be
    Value, TieRef, PointRef, ReferenceUndecided

Example
    >>> from obd_core import reference as ref
    >>> r = ref.tie(1000, 480, 979)
    >>> float(r.slope_left), r.is_cusp, r.method
    (31.85082244300127, False, 'interval@50')
    >>> ref.compare_tie(1000, 480, 979, slope_left=31.850822443183596)["slope_left"]  # obd_core's value
    5.72...e-12
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, log2

from mpmath import iv, mp

__all__ = [
    "Value", "TieRef", "PointRef", "ReferenceUndecided",
    "tie", "axis", "at", "compare_tie", "check_tie_table", "expected_double_error", "check_invariants",
]

# Exact arithmetic at a rational p = a/b builds n+1 integers of ~n log2(max(a, b-a)) bits each.
# Above this many bits in total the interval route is used instead (it handles ties at rational p
# with an exact pairwise test, so nothing is lost but speed).
EXACT_BITS_LIMIT = 4e8


# Precision for Value's midpoint, radius and comparisons; interval endpoints carry their own.
_WORK_DPS = 120


class ReferenceUndecided(RuntimeError):
    """The reference could not prove a ranking or a sign even at ``max_dps`` digits."""


class _NeedMorePrecision(Exception):
    pass


# ------------------------------------------------------------------------------------------------
# Value: a rigorous enclosure


@dataclass(frozen=True)
class Value:
    """A rigorous enclosure ``lo <= x <= hi`` of a real number.

    ``lo``/``hi`` are ``Fraction`` when the value is known exactly (then ``lo == hi``), otherwise
    mpmath ``mpf`` endpoints from interval arithmetic.  ``float(v)`` gives the midpoint rounded to
    double; ``v.rel_err(x)`` is how far a double result ``x`` is from it.
    """

    lo: object
    hi: object

    @property
    def exact(self) -> bool:
        return isinstance(self.lo, Fraction) and self.lo == self.hi

    @property
    def mid(self):
        """Midpoint as an mpf (at the precision active when called; 50 digits by default)."""
        with mp.workdps(max(mp.dps, _WORK_DPS)):
            return (_to_mpf(self.lo) + _to_mpf(self.hi)) / 2

    @property
    def rad(self):
        """Half-width (0 for an exact value)."""
        if self.exact:
            return mp.mpf(0)
        with mp.workdps(max(mp.dps, _WORK_DPS)):
            return (_to_mpf(self.hi) - _to_mpf(self.lo)) / 2

    def __float__(self) -> float:
        if self.exact:
            return float(self.lo)
        return float(self.mid)

    def sign(self) -> int | None:
        """+1 or -1 if the whole enclosure is on one side of 0, 0 if exactly 0, None if undecided."""
        if self.exact:
            return (self.lo > 0) - (self.lo < 0)
        if _to_mpf(self.lo) > 0:
            return 1
        if _to_mpf(self.hi) < 0:
            return -1
        return None

    def contains(self, x) -> bool:
        """True if ``x`` (float, int, Fraction or mpf) lies inside the enclosure."""
        xf = _as_fraction(x)
        if self.exact:
            return xf == self.lo
        with mp.workdps(max(mp.dps, _WORK_DPS)):
            xm = _to_mpf(xf)
            return _to_mpf(self.lo) <= xm <= _to_mpf(self.hi)

    def abs_err(self, x) -> float:
        """|x - value| (using the midpoint; the enclosure is far narrower than any double error)."""
        with mp.workdps(max(mp.dps, _WORK_DPS)):
            return float(abs(_to_mpf(_as_fraction(x)) - self.mid))

    def rel_err(self, x, floor: float = 1.0) -> float:
        """|x - value| / max(|value|, floor).  ``floor=1`` measures small values absolutely."""
        with mp.workdps(max(mp.dps, _WORK_DPS)):
            m = self.mid
            return float(abs(_to_mpf(_as_fraction(x)) - m) / max(abs(m), mp.mpf(floor)))

    def __repr__(self) -> str:
        with mp.workdps(_WORK_DPS):
            if self.exact:
                return f"Value({mp.nstr(_to_mpf(self.lo), 25)}, exact)"
            return f"Value({mp.nstr(self.mid, 25)} ± {mp.nstr(self.rad, 3)})"


def _as_fraction(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, float)):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    man, exp = mp.mpf(x).man_exp                     # mpf -> its exact binary fraction
    return Fraction(int(man)) * (Fraction(2) ** int(exp))


def _to_mpf(x):
    if isinstance(x, Fraction):
        return mp.mpf(x.numerator) / x.denominator
    return mp.mpf(x)


def _from_iv(x) -> Value:
    return Value(mp.make_mpf(x._mpi_[0]), mp.make_mpf(x._mpi_[1]))


# ------------------------------------------------------------------------------------------------
# Result records


@dataclass(frozen=True)
class TieRef:
    """Everything at the tie point of pair (i, j) of n.  Every number is a ``Value``.

    ``S_minus``/``S_plus`` are the slope numerators (E'_± = S_± / (p q)); ``D`` is the slope jump
    E'_+ - E'_- (at the axis p = 1/2 it is the sum of every mirror pair's kink); ``F3`` is
    (n+i-j)(i+j-2np) + (j-np) (None at the axis).  ``is_cusp`` is the proved verdict
    S_- < 0 < S_+ (None only if undecided at max_dps).  ``method`` is "exact" or
    "interval@<dps>" (the precision at which everything was decided).
    """

    n: int
    i: int
    j: int
    p: Value
    E: Value
    S_minus: Value
    S_plus: Value
    slope_left: Value
    slope_right: Value
    D: Value
    log10_D: Value
    F3: Value | None
    is_cusp: bool | None
    method: str


@dataclass(frozen=True)
class PointRef:
    """E and the one-sided slopes at an arbitrary p.  ``tied`` lists the groups of indices whose
    masses are exactly equal at p (empty unless p is a tie point); ``D = slope_right - slope_left``
    is nonzero only then."""

    n: int
    p: Value
    E: Value
    slope_left: Value
    slope_right: Value
    D: Value
    tied: list = field(default_factory=list)
    method: str = ""


# ------------------------------------------------------------------------------------------------
# Ranking with exact ties resolved by the one-sided limit


def _ranks(order_groups, n):
    """order_groups: list of index groups in increasing mass order, each group of equal masses.
    Returns (w_left, w_right): ranks just left / just right of p."""
    wl = [0] * (n + 1)
    wr = [0] * (n + 1)
    r = 0
    for g in order_groups:
        g = sorted(g)
        m = len(g)
        for t in range(m):
            wr[g[t]] = r + t                 # right of p: larger index, larger mass
            wl[g[m - 1 - t]] = r + t         # left of p:  larger index, smaller mass
        r += m
    return wl, wr


# ------------------------------------------------------------------------------------------------
# Exact route: p = a/b rational


def _exact_bits(n, a, b):
    return (n + 1) * (n * log2(max(a, b - a, 2)) + n)


def _exact_eval(n, a, b):
    """E, S_left, S_right, slopes and tie groups at p = a/b, exactly.  Masses are the integers
    M_k = C(n,k) a^k (b-a)^(n-k); f(k) = M_k / b^n."""
    c = b - a
    M = [comb(n, k) * a ** k * c ** (n - k) for k in range(n + 1)]
    idx = sorted(range(n + 1), key=lambda k: M[k])
    groups, cur = [], [idx[0]]
    for k in idx[1:]:
        if M[k] == M[cur[-1]]:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    wl, wr = _ranks(groups, n)
    den = b ** n
    E = Fraction(sum(wl[k] * M[k] for k in range(n + 1)), den)
    Sl = Fraction(sum(wl[k] * M[k] * (k * b - n * a) for k in range(n + 1)), b * den)
    Sr = Fraction(sum(wr[k] * M[k] * (k * b - n * a) for k in range(n + 1)), b * den)
    pq = Fraction(a * c, b * b)
    tied = [tuple(sorted(g)) for g in groups if len(g) > 1 and M[g[0]] != 0]
    return {
        "p": Fraction(a, b), "E": E, "S_left": Sl, "S_right": Sr,
        "slope_left": Sl / pq, "slope_right": Sr / pq, "tied": tied,
        "f": lambda k: Fraction(M[k], den), "pq": pq,
    }


# ------------------------------------------------------------------------------------------------
# Interval route


def _interval_eval(n, dps, p_spec, equal_pair=None, rational=None):
    """E, S_left, S_right at p, in interval arithmetic at ``dps`` digits.

    p_spec: ("pair", i, j) for the irrational tie point of (i, j), or ("ratio", a, b) for p = a/b.
    Masses whose enclosures overlap must be provably equal -- the tie pair itself (``equal_pair``)
    or, at a rational p, a pair passing the exact integer test -- else _NeedMorePrecision.
    """
    saved = iv.prec
    try:
        iv.dps = dps
        if p_spec[0] == "pair":
            _, i, j = p_spec
            rho = (iv.mpf(comb(n, i)) / comb(n, j)) ** (iv.mpf(1) / (j - i))     # p/q at p*
        else:
            _, a, b = p_spec
            rho = iv.mpf(a) / (b - a)
        p = 1 / (1 + 1 / rho)
        q = 1 / (1 + rho)
        f = [None] * (n + 1)
        f[0] = q ** n
        for k in range(n):
            f[k + 1] = f[k] * (n - k) / (k + 1) * rho
        if equal_pair is not None:
            i, j = equal_pair
            f[j] = f[i]                                  # equal by definition at p*
        lo = [mp.make_mpf(x._mpi_[0]) for x in f]
        hi = [mp.make_mpf(x._mpi_[1]) for x in f]
        order = sorted(range(n + 1), key=lambda k: (lo[k] + hi[k]) / 2)
        # overlapping enclosures form groups; each must be provably all-equal
        groups, cur, cur_hi = [], [order[0]], hi[order[0]]
        for k in order[1:]:
            if lo[k] <= cur_hi:
                cur.append(k)
                cur_hi = max(cur_hi, hi[k])
            else:
                groups.append(cur)
                cur, cur_hi = [k], hi[k]
        groups.append(cur)
        tied = []
        for g in groups:
            if len(g) == 1:
                continue
            if not _provably_equal(n, g, equal_pair, rational):
                raise _NeedMorePrecision
            tied.append(tuple(sorted(g)))
        wl, wr = _ranks(groups, n)
        E = sum((wl[k] * f[k] for k in range(n + 1)), iv.mpf(0))
        Sl = sum((wl[k] * f[k] * (k - n * p) for k in range(n + 1)), iv.mpf(0))
        Sr = sum((wr[k] * f[k] * (k - n * p) for k in range(n + 1)), iv.mpf(0))
        pq = p * q
        return {"p": p, "q": q, "E": E, "S_left": Sl, "S_right": Sr,
                "slope_left": Sl / pq, "slope_right": Sr / pq, "tied": tied, "f": f, "pq": pq}
    finally:
        iv.prec = saved


def _provably_equal(n, group, equal_pair, rational):
    if equal_pair is not None and sorted(group) == sorted(equal_pair):
        return True
    if rational is None:
        return False                                     # irrational p: no other exact ties
    a, b = rational
    c = b - a
    g = sorted(group)
    k = g[0]
    # f(k) == f(l)  <=>  C(n,k) c^(l-k) == C(n,l) a^(l-k)
    return all(comb(n, k) * c ** (l - k) == comb(n, l) * a ** (l - k) for l in g[1:])


def _log10(v):
    """log10 of a positive Value, as a Value (interval)."""
    saved = iv.prec
    try:
        iv.dps = 60
        x = iv.mpf([_to_mpf(v.lo), _to_mpf(v.hi)]) if not v.exact else iv.mpf(v.lo.numerator) / v.lo.denominator
        return _from_iv(iv.log(x) / iv.log(iv.mpf(10)))
    finally:
        iv.prec = saved


# ------------------------------------------------------------------------------------------------
# Public functions


def tie(n: int, i: int, j: int, dps: int = 50, max_dps: int = 800) -> TieRef:
    """Every quantity at the tie point p*(n, i, j) where f(i) = f(j), 0 <= i < j <= n.

    Adjacent pairs (j = i+1) have a rational tie point and are computed exactly; i + j = n is the
    axis p = 1/2 (see ``axis``); otherwise interval arithmetic from ``dps`` digits, doubled until
    the ranking and the sign of S_-, S_+ are proved (``ReferenceUndecided`` past ``max_dps``).
    Pairs with i + j < n are fine too (p* < 1/2).
    """
    n, i, j = int(n), int(i), int(j)
    if not (0 <= i < j <= n):
        raise ValueError(f"need 0 <= i < j <= n, got n={n}, i={i}, j={j}")
    if i + j == n:
        r = axis(n)
        return TieRef(n, i, j, r.p, r.E, r.S_minus, r.S_plus, r.slope_left, r.slope_right,
                      r.D, r.log10_D, None, r.is_cusp, r.method)
    m = j - i
    if m == 1 and _exact_bits(n, i + 1, n + 1) <= EXACT_BITS_LIMIT:
        x = _exact_eval(n, i + 1, n + 1)
        p = x["p"]
        Sm, Sp = x["S_left"], x["S_right"]
        D = x["slope_right"] - x["slope_left"]
        F3 = (n + i - j) * (i + j - 2 * n * p) + (j - n * p)
        verdict = Sm < 0 < Sp
        return TieRef(n, i, j, Value(p, p), Value(x["E"], x["E"]), Value(Sm, Sm), Value(Sp, Sp),
                      Value(x["slope_left"], x["slope_left"]), Value(x["slope_right"], x["slope_right"]),
                      Value(D, D), _log10(Value(D, D)), Value(F3, F3), verdict, "exact")
    d = int(dps)
    while d <= max_dps:
        try:
            x = _interval_eval(n, d, ("pair", i, j), equal_pair=(i, j))
        except _NeedMorePrecision:
            d *= 2
            continue
        Sm, Sp = _from_iv(x["S_left"]), _from_iv(x["S_right"])
        sm, sp = Sm.sign(), Sp.sign()
        if sm is None or sp is None:
            d *= 2
            continue
        saved = iv.prec
        try:
            iv.dps = d
            kink = m * x["f"][i] / x["pq"]
            pm = x["p"]
            F3 = (n + i - j) * (i + j - 2 * n * pm) + (j - n * pm)
        finally:
            iv.prec = saved
        D = _from_iv(kink)
        return TieRef(n, i, j, _from_iv(x["p"]), _from_iv(x["E"]), Sm, Sp,
                      _from_iv(x["slope_left"]), _from_iv(x["slope_right"]), D, _log10(D),
                      _from_iv(F3), (sm < 0 < sp), f"interval@{d}")
    raise ReferenceUndecided(f"tie({n}, {i}, {j}): ranking or sign of S_-/S_+ unproved at {max_dps} digits")


def axis(n: int) -> TieRef:
    """The tie point p = 1/2, where every mirror pair (k, n-k) ties at once.  Exact.

    Reported with the sentinel pair (i, j) = (0, n) (obd_core's convention); D is the whole kink
    E'_+ - E'_- there, the sum over all mirror pairs; F3 is None (it is defined per pair)."""
    n = int(n)
    x = _exact_eval(n, 1, 2)
    D = x["slope_right"] - x["slope_left"]
    return TieRef(n, 0, n, Value(x["p"], x["p"]), Value(x["E"], x["E"]),
                  Value(x["S_left"], x["S_left"]), Value(x["S_right"], x["S_right"]),
                  Value(x["slope_left"], x["slope_left"]), Value(x["slope_right"], x["slope_right"]),
                  Value(D, D), _log10(Value(D, D)), None, x["S_left"] < 0 < x["S_right"], "exact")


def at(n: int, p, dps: int = 50, max_dps: int = 800) -> PointRef:
    """E and the one-sided slopes at an arbitrary p in [0, 1].

    ``p`` may be a float (taken as its exact binary value), int, Fraction or a string like "3/5".
    Rational p is computed exactly when the integers stay small enough (``EXACT_BITS_LIMIT``),
    otherwise by interval arithmetic with ties proved by an exact integer test.  At p = 0 and 1,
    E = n and both slopes are the one-sided limits -n and +n (method "limit"), as in
    obd_core.E_slopes_at.  For a tie point given by its pair, use ``tie`` instead: irrational tie
    points cannot be passed as a number.
    """
    n = int(n)
    pf = _as_fraction(p)
    if not (0 <= pf <= 1):
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if pf in (0, 1):
        s = Fraction(-n if pf == 0 else n)
        return PointRef(n, Value(pf, pf), Value(Fraction(n), Fraction(n)), Value(s, s), Value(s, s),
                        Value(Fraction(0), Fraction(0)), [], "limit")
    a, b = pf.numerator, pf.denominator
    if _exact_bits(n, a, b) <= EXACT_BITS_LIMIT:
        x = _exact_eval(n, a, b)
        D = x["slope_right"] - x["slope_left"]
        return PointRef(n, Value(pf, pf), Value(x["E"], x["E"]),
                        Value(x["slope_left"], x["slope_left"]), Value(x["slope_right"], x["slope_right"]),
                        Value(D, D), x["tied"], "exact")
    d = int(dps)
    while d <= max_dps:
        try:
            x = _interval_eval(n, d, ("ratio", a, b), rational=(a, b))
        except _NeedMorePrecision:
            d *= 2
            continue
        sl, sr = _from_iv(x["slope_left"]), _from_iv(x["slope_right"])
        saved = iv.prec
        try:
            iv.dps = d
            D = _from_iv(x["slope_right"] - x["slope_left"]) if x["tied"] else Value(Fraction(0), Fraction(0))
        finally:
            iv.prec = saved
        return PointRef(n, Value(pf, pf), _from_iv(x["E"]), sl, sr, D, x["tied"], f"interval@{d}")
    raise ReferenceUndecided(f"at({n}, {p!r}): ranking unproved at {max_dps} digits")


# ------------------------------------------------------------------------------------------------
# Checking double-precision results


def expected_double_error(n: int) -> dict[str, float]:
    """How close obd_core's double-precision values should be to the reference.

    Relative errors in the sense of ``Value.rel_err`` (floor 1, so small values are measured
    absolutely).  EMPIRICAL bounds, not proofs: the worst errors over ~200 sampled rows of
    tie_table(n, both_halves=True) at n = 10, 100, 500, 1000, 2000 (2026-10-07), times a margin of
    ~4-6.  Measured worst cases:

        n      p*        E         slopes    log10_D
        100    4.1e-15   5.5e-16   8.6e-12   3.2e-14
        1000   1.4e-14   6.7e-16   5.3e-10   2.3e-13
        2000   3.7e-13   6.5e-16   1.7e-08   4.8e-13

    The slopes are the weak spot: S_- is a sum of terms ~ n f(k)(k - np) that cancel to a small
    number, so its error grows faster than n^2 eps.  p* comes from a difference of log-binomials of
    size ~n, so its error grows like n log(n) eps.  E is essentially full precision.  An error
    above these bounds means something is wrong, not just rounding.
    """
    from math import log

    n = max(int(n), 1)
    return {
        "p": 1e-16 * n * log(n + 1) + 1e-15,
        "E": 4e-15,
        "slopes": 4 * (2.5e-15 * n * n + 2e-18 * n ** 3) + 1e-13,
        "log10_D": 1e-15 * n + 1e-14,
    }


def compare_tie(n: int, i: int, j: int, dps: int = 50, **values) -> dict:
    """Relative errors of double-precision values at the tie point of (i, j) against ``tie()``.

    Pass any of: p (or pstar), E, S_minus, S_plus, slope_left, slope_right, log10_D, F3,
    is_cusp.  Returns {name: rel_err} for the numbers (``Value.rel_err``, floor 1) and
    {"is_cusp": True/False} for whether the verdict matches, plus "reference": the TieRef.
    """
    r = tie(n, i, j, dps=dps)
    out: dict = {"reference": r}
    alias = {"pstar": "p"}
    for name, val in values.items():
        key = alias.get(name, name)
        if key == "is_cusp":
            out["is_cusp"] = (bool(val) == r.is_cusp)
            continue
        ref = getattr(r, key, None)
        if ref is None:
            raise KeyError(f"no reference value for {name!r}")
        out[name] = ref.rel_err(val)
    return out


def check_tie_table(n: int, table: dict, rows=None, sample: int = 25, seed: int = 0) -> dict:
    """Check rows of ``obd_core.tie_table(n)`` (either half) against the reference.

    ``rows``: explicit row indices, else ``sample`` random rows (plus row 0 and the last row).
    Returns {"max": {field: worst rel_err}, "worst_row": {field: row}, "verdicts_ok": bool,
    "rows": [...], "within_expected": bool} where "within_expected" compares the maxima with
    ``expected_double_error(n)``.
    """
    import random

    m = len(table["i"])
    if rows is None:
        rnd = random.Random(seed)
        rows = sorted({0, m - 1} | set(rnd.sample(range(m), min(sample, m))))
    fields = [f for f in ("pstar", "E", "slope_left", "slope_right", "log10_D") if f in table]
    worst = {f: 0.0 for f in fields}
    worst_row = {f: None for f in fields}
    verdicts_ok = True
    for r_ in rows:
        i, j = int(table["i"][r_]), int(table["j"][r_])
        res = compare_tie(n, i, j, **{f: float(table[f][r_]) for f in fields},
                          **({"is_cusp": bool(table["is_cusp"][r_])} if "is_cusp" in table else {}))
        verdicts_ok &= res.get("is_cusp", True)
        for f in fields:
            if res[f] > worst[f]:
                worst[f], worst_row[f] = res[f], r_
    tol = expected_double_error(n)
    limit = {"pstar": tol["p"], "E": tol["E"], "slope_left": tol["slopes"],
             "slope_right": tol["slopes"], "log10_D": tol["log10_D"]}
    return {"max": worst, "worst_row": worst_row, "verdicts_ok": verdicts_ok, "rows": rows,
            "within_expected": all(worst[f] <= limit[f] for f in fields)}


def check_invariants(n: int, table: dict) -> dict:
    """Check every row of a tie table against bounds that follow from the definitions.  Fast.

    ``table``: dict of arrays sorted by p, as ``obd_core.tie_table(n, ...)`` returns (or OBD's tie
    tables): ``pstar`` (or ``p``), ``E``, ``slope_left``, ``slope_right``, ``log10_D``.  These are
    necessary conditions, not a reference comparison, so they cost microseconds per row and catch a
    single corrupt row among millions -- which sampling ``check_tie_table`` would almost surely miss.

    The bounds (q = 1 - p, K ~ Bin(n, p)):
      * ``p_order``    0 < p < 1, strictly increasing
      * ``E_range``    0 <= E <= n                       (ranks are 0..n, masses sum to 1)
      * ``slope_order`` slope_right >= slope_left       (the kink D > 0)
      * ``log10_D``    finite
      * ``slope_bound`` |E'| <= n^1.5 / (2 sqrt(pq)): S = sum (w_k - n/2) f(k)(k - np) since
                       sum f(k)(k - np) = 0, so |S| <= (n/2) E|K - np| <= (n/2) sqrt(npq), and
                       E' = S / (pq)
      * ``E_lipschitz`` |E(p_b) - E(p_a)| <= max |E'| (p_b - p_a) between neighbouring tie points
                       (E is continuous), with max |E'| taken at the end where pq is smaller
    Each bound gets a relative slack of 1e-9 plus a few units of double rounding.  Returns
    {"ok": bool, "violations": {check: count}, "first_bad_row": {check: row}}.
    """
    import numpy as np

    n = int(n)
    p = np.asarray(table["pstar"] if "pstar" in table else table["p"], dtype=float)
    E = np.asarray(table["E"], dtype=float)
    sl = np.asarray(table["slope_left"], dtype=float)
    sr = np.asarray(table["slope_right"], dtype=float)
    ld = np.asarray(table["log10_D"], dtype=float)
    eps = 1e-9
    pq = p * (1.0 - p)
    bound = n ** 1.5 / (2.0 * np.sqrt(np.where(pq > 0, pq, np.nan)))
    bad = {
        "p_order": (p <= 0) | (p >= 1) | np.r_[False, np.diff(p) <= 0],
        "E_range": (E < -eps * n) | (E > n * (1 + eps)) | ~np.isfinite(E),
        "slope_order": ~(sr >= sl),
        "log10_D": ~np.isfinite(ld),
        "slope_bound": ~((np.abs(sl) <= bound * (1 + eps)) & (np.abs(sr) <= bound * (1 + eps))),
    }
    lip = np.zeros(p.size, dtype=bool)
    if p.size > 1:
        steep = np.maximum(bound[:-1], bound[1:])
        jump = np.abs(np.diff(E))
        lip[1:] = jump > steep * np.diff(p) * (1 + eps) + 1e-12 * n
    bad["E_lipschitz"] = lip
    violations = {k: int(v.sum()) for k, v in bad.items()}
    first = {k: int(np.flatnonzero(v)[0]) for k, v in bad.items() if v.any()}
    return {"ok": not any(violations.values()), "violations": violations, "first_bad_row": first}
