"""
obd_core -- the shared mathematics for the ordered-binomial projects (formerly binom_core.py).

The only implementation of it: ordered-binomial-cusps (research) and OBD (visualization) both
import this, and nothing below is duplicated elsewhere.  Full guide with examples and pitfalls:
README.md at https://github.com/dperlman/OBD-core (rules for changing the package: CLAUDE.md).

Start here
    tie_table(n, both_halves=False)  every tie point of n: p*, E, exact slopes, log10_D, proved
                                     cusp verdict (dict of numpy arrays; row 0 = the axis p=1/2)
    tie_table(n, p_range=(lo, hi))   only the tie points with lo <= p* <= hi, same rows bit for bit
    E_slopes_at(n, p_array)          E and the exact one-sided slopes at any p (grids)
    E_at(n, p), E_half(n)            E at one p / at 1/2
    evaluate(n, i, j)                one tie point in double precision
    certify*(n, i, j)                proved verdict for one tie point (tie_table already does this)
    recheck(n, i, j)                 print every quantity at one tie point, rigorously
    obd_core.reference               independent exact / interval-arithmetic values for CHECKING
                                     this module: reference.tie, .at, .axis, .compare_tie,
                                     .check_tie_table, .expected_double_error
    NUMERIC_PINS, pin_mismatches()   the pinned numpy/numba/llvmlite/mpmath versions

Precision: cusp verdicts are proved; E, S_-, slopes and p* are plain double precision, accurate
to reference.expected_double_error(n) (slopes ~1e-9 relative at n=1000).  Never take the slope
jump D as slope_right - slope_left; use log10_D.

Definitions:

    f_p(k) = C(n,k) p^k (1-p)^(n-k),  k = 0..n
    p*     = tie point of masses i<j (0<=i<j<=n, i+j>n so p*>1/2): f(i)=f(j);
             The range was widened from 0<i<j<n on 2026-09-21: the pairs (i,n) are genuine order
             changes with p*>1/2 -- their mirrors (0,j) sit below 1/2, so the symmetry restriction
             does not remove them -- and the last of them, (n-1,n), is at p*=n/(n+1), above which
             every mass is in natural order and E = n p exactly.  They hold no cusps for n>=4.
             rho = p*/(1-p*) = (C(n,i)/C(n,j))^(1/(j-i))
    w_k    = rank of f(k) increasing, 0 = smallest.  Left of p*: w_j = w_i - 1.
    E(n,p) = sum_k w_k f_p(k)
    S_-    = sum_k w_k f(k)(k - n p*)       left slope numerator; E'_- = S_-/(p* q*)
    S_+    = S_- + (j-i) f(i)               right slope numerator
    cusp  <=>  S_- < 0 < S_+
    F3     = (n+i-j)(i+j-2np*) + (j-np*)    right-hand slope of the pair's own contribution T+V,
                                            in units of f(i)/(p* q*).  NOT the slope of T alone.

Screening rule (tie_kernel): a tie point is decided in double precision when
S_- < -MARGIN and S_+ > MARGIN (MIN), or S_- > MARGIN or S_+ < -MARGIN (NOT).  Anything else is
tagged CHECK and must go to certify().  Masses below TINY are treated as zero: without that the
CHECK count explodes with n (182,542 at n=2000 instead of ~75).

On top of the margin test the ranking itself may be uncertain, and there are two rules for that.
The ORIGINAL one (sharp=False) is a proxy: flag whenever two adjacent masses in the ranking, both
>= TINY, are within relative GAP = 1e-8.  It never asks whether a wrong ranking would MATTER, and
almost always it does not -- swapping adjacent ranks of masses a,b moves S_- by exactly
f_a(a-np*) - f_b(b-np*) ~ f*(a-b), so the damage scales with the SIZE of the masses, and a near-tie
between two masses at 1e-287 moves S_- by ~1e-284 against an S_- of order 1e-2.
The SHARPENED one (sharp=True, the DEFAULT since 2026-09-21) bounds that movement instead of
guessing at it: masses whose order
double precision cannot resolve are grouped into maximal clusters, within a cluster of size c any
rank moves by at most c-1, and the resulting bound (c-1)*sum_{k in C} f_k|k - n p*| is added to
MARGIN on both sides.  The resolution threshold comes from _err_bounds, which is derived from the
algorithm's own operation count -- nothing in it is fitted.  Validated by validate_trigger.py:
over all 195,243 tie points that the GAP rule escalated for n<=3000 it decides 195,062 in double
with ZERO disagreements against the mpmath verdict, and over every tie point of
n = 135,400,800,1000,1100,1200,1500,2000,2500,3000,4000,5000 it flags nothing the GAP rule decided.
At n=5000 it takes the interval-arithmetic workload from 1411 tie points to 1.  GAP is now used
by nothing but that comparison path.

The masses are ALWAYS normalised by their own sum before E and S_- are accumulated.  As computed
from exp(lnC + k ln p + (n-k) ln q) they carry a shared relative error ~6e-13 (they sum to
0.999999999999363 at n=2000), which leaves only ~3.6 digits on E - E(1/2); normalised, ~6.3.
This cannot change a certified verdict: it scales S_- and S_+ by a common 1+6e-13, twelve orders
below MARGIN.  It DOES change the descriptive columns in the 16th significant digit, so files
written before 2026-09-20 differ from freshly generated ones -- regenerate rather than mix them.
"""
import numpy as np
from math import comb, lgamma, log as _log
from numba import njit

# The numerical stack every environment that runs this module should have, exactly.  Different
# numba/llvmlite/numpy builds can change the last bit of exp/log and so of every descriptive column,
# and numba's on-disk cache is per numba version.  Bit-identity holds on one platform: Linux x86-64
# and macOS arm64 differ in the last bits even with identical pins (their libm), as CI shows.  Both repos install with
#     pip install -c constraints.txt ...
# (constraints.txt is generated from this dict; tests/test_pins.py keeps them equal).  Importing
# in an environment that differs only warns: verdicts do not depend on the last bit, but byte-for-
# byte agreement between environments is no longer guaranteed.
NUMERIC_PINS = {"numpy": "2.3.5", "numba": "0.63.1", "llvmlite": "0.46.0", "mpmath": "1.4.0"}

def pin_mismatches():
    """{package: installed version} for every NUMERIC_PINS entry the running environment differs on."""
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for pkg, want in NUMERIC_PINS.items():
        try:
            have = version(pkg)
        except PackageNotFoundError:
            have = None
        if have != want:
            out[pkg] = have
    return out

def _warn_on_pin_mismatch():
    bad = pin_mismatches()
    if bad:
        import warnings
        warnings.warn(
            "obd_core: numerical stack differs from NUMERIC_PINS ("
            + ", ".join(f"{k} {v} != {NUMERIC_PINS[k]}" for k, v in sorted(bad.items()))
            + "); results may differ in the last bit from other environments.  "
            "Install with: pip install -c constraints.txt (from the OBD-core repo).",
            stacklevel=3,
        )

_warn_on_pin_mismatch()

MARGIN, GAP, TINY = 1e-6, 1e-8, 1e-290
LN_TINY = _log(TINY)                 # window test in _one_tie; do not hardcode this
EPS = 2.0**-53                       # unit roundoff, for the sharpened trigger's error bounds
TAG_NOT, TAG_MIN, TAG_CHECK = 0, 1, 2

def lnC_arr(n):
    """log C(n,k) for k=0..n."""
    return np.array([lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1) for k in range(n+1)])

def n_ties(n):
    """number of tie points with 0<=i<j<=n and i+j>n.  (i=0 contributes none: i+j>n needs j>n.)"""
    return sum(max(0, n - max(i+1, n-i+1) + 1) for i in range(1, n))

def E_half(n):
    """E(n,1/2), masses normalised by their own sum -- same convention as _one_tie."""
    import math
    f = np.exp(lnC_arr(n) - n*math.log(2.0))
    f = f/math.fsum(f.tolist())
    return math.fsum((np.arange(n+1)*np.sort(f)).tolist())

def E_at(n, p):
    """E(n,p) at an arbitrary p, masses normalised by their own sum -- same convention as E_half
    and _one_tie.  Ranks come from a sort, which is exact away from tie points; AT a tie point E is
    still right (E is continuous there), only the one-sided slopes need the kernel's bookkeeping.
    Descriptive only, not certified."""
    import math
    k = np.arange(n+1)
    f = np.exp(lnC_arr(n) + k*math.log(p) + (n-k)*math.log1p(-p))
    f = f/math.fsum(f.tolist())
    return math.fsum((np.arange(n+1)*np.sort(f)).tolist())

@njit(cache=True)
def _err_bounds(n, lnC, i, j, md, q, lnp, lnq):
    """Error bounds for the sharpened re-ranking trigger.  Returns (dp, df0, dstep).

    Derived from the algorithm in _one_tie, term by term; eps = 2^-53 throughout.  Every constant
    below counts floating-point operations, so nothing here is fitted.

    dp -- relative error of the computed p*.  lnrho = (lnC[i]-lnC[j])/m subtracts two numbers of
      size ~n ln2 whose difference is only O(m), so the cancellation costs a factor n/m: with
      lgamma at <=2 ulp per term the absolute error of lnrho is <= 3 eps max(lnC[i],lnC[j])/m, and
      p = 1/(1+exp(-lnrho)) has dp/dlnrho = p q, so the RELATIVE error of p is q times that.
      (Measured: 5.8e-12 worst case at n=8000, m=1, against 8e-12 from this form.)
    df0, dstep -- relative error of the computed mass f[k], as df0 + dstep*|k-md|.
      f[md] = exp(lnC[md] + md lnp + (n-md) lnq): exp turns an ABSOLUTE argument error into a
      relative result error, and the argument's three terms are each ~n-sized while their sum is
      O(1), so their roundings do not cancel -- 4 eps L with L the sum of their magnitudes, plus
      1 eps for exp itself.  Each recurrence step is 3 flops on positive quantities, so the
      relative error accumulates additively at 3 eps per step away from the mode.
      Normalisation by the mass sum is NOT included: it is a factor common to every mass, so it
      cancels in every ratio the trigger tests and cannot change the sign of S_-.
    """
    m = j - i
    Lc = lnC[i] if lnC[i] > lnC[j] else lnC[j]
    dp = q * 3.0*EPS*Lc / m
    L = abs(lnC[md]) + abs(md*lnp) + abs((n-md)*lnq)
    df0 = 4.0*EPS*L + EPS
    return dp, df0, 3.0*EPS

@njit(cache=True)
def _masses(n, lnC, md, lnp, lnq, rho, lo_req, hi_req, f):
    """Binomial masses by recurrence outwards from the mode md, into f[lo..hi].  The ONLY place the
    masses are built; _one_tie (tie points) and _E_slopes_one (arbitrary p) both call it.

    Only the masses at or above TINY can affect anything: the rest are zeroed, contribute exactly
    0.0 to every sum, and sit as an equal block at the bottom of the ranking.  The masses fall away
    monotonically from the mode, so once the recurrence drops below TINY it stays below and we can
    stop -- giving a window [lo,hi] of width O(sqrt(n)) instead of n+1.  The recurrence is not
    allowed to stop before lo_req on the left or hi_req on the right.  Unnormalised; returns
    (lo, hi).  Entries of f outside [lo,hi] are not written.
    """
    f[md] = np.exp(lnC[md] + md*lnp + (n-md)*lnq)
    k = md
    while k > 0:                                 # leftwards: f_{k-1} = f_k * k/((n-k+1) rho)
        v = f[k] * k / ((n-k+1.0)*rho)
        f[k-1] = v; k -= 1
        if v < TINY and k <= lo_req: break
    lo = k
    k = md
    while k < n:                                 # rightwards
        v = f[k] * rho*(n-k) / (k+1.0)
        f[k+1] = v; k += 1
        if v < TINY and k >= hi_req: break
    hi = k
    for k in range(lo, hi+1):                    # masses below TINY are numerically zero
        if f[k] < TINY: f[k] = 0.0
    return lo, hi

@njit(cache=True)
def _pstar(lnC, i, j):
    """The tie point p* of the pair (i, j): (p*/q*)^(j-i) = C(n,i)/C(n,j).  Returns (p*, ln rho).

    The ONLY place p* is computed: _one_tie and the windowed search (window_kernel) both call it,
    so a window test and the table it selects agree to the last bit.
    """
    lnrho = (lnC[i] - lnC[j]) / (j - i)
    return 1.0/(1.0 + np.exp(-lnrho)), lnrho

@njit(cache=True)
def _one_tie(n, lnC, i, j, f, w):
    """All quantities for a single tie point.  The masses come from _masses, the one place they are built.

    Returns (p, ln_fi, E, S_minus, kappa, F3, tag).  Only the masses at or above TINY are touched
    (see the window comment in the body); the rest are exactly zero and change no result.
    Masses are always normalised by their own sum:
    unnormalised they carry a shared relative error ~6e-13 from exp/lgamma (they sum to
    0.999999999999363 at n=2000), which leaves only ~3.6 digits on E - E(1/2) instead of ~6.3.
    f and w are scratch buffers of length n+1, passed in so a loop can reuse them.
    """
    m = j - i
    p, lnrho = _pstar(lnC, i, j)
    q = 1.0 - p; rho = p/q
    lnp = np.log(p); lnq = np.log(q)
    md = int(np.floor((n+1)*p))                  # mode of Bin(n,p)
    if md > n: md = n
    # Masses below TINY are zeroed (see _masses).  The one trap: f[j] = f[i] below is applied AFTER
    # zeroing, so it can rescue a j that fell just under TINY.  f(i) is known in closed form, so
    # when the pair is above TINY we refuse to stop before reaching i on the left and j on the
    # right, and the rescue still happens.
    pair_in = (lnC[i] + i*lnp + (n-i)*lnq) >= LN_TINY
    lo_req = i if pair_in else md
    hi_req = j if pair_in else md
    lo, hi = _masses(n, lnC, md, lnp, lnq, rho, lo_req, hi_req, f)
    if pair_in or i >= lo:
        f[j] = f[i]                              # exact tie
    else:
        # The pair is below TINY and i fell outside the window, so f[i] was never written for this
        # tie point: it holds whatever the reused buffer last held.  Copying it into a j inside the
        # window corrupted E and S_- (n=978, pair (1,978), window [2,978]).  Both masses are below
        # TINY, i.e. numerically zero, which is what f[j] must be.
        f[j] = 0.0
    nz = lo + (n - hi)                           # masses outside the window: all exactly zero,
                                                 # so they take ranks 0..nz-1 as an equal block
    s = 0.0; comp = 0.0                          # Neumaier sum, then rescale
    for k in range(lo, hi+1):
        t = s + f[k]
        if abs(s) >= abs(f[k]): comp += (s - t) + f[k]
        else:                   comp += (f[k] - t) + s
        s = t
    s = s + comp
    lns = 0.0
    if s > 0.0:
        inv = 1.0/s
        for k in range(lo, hi+1): f[k] *= inv
        lns = np.log(s)
    ln_fi = lnC[i] + i*lnp + (n-i)*lnq - lns
    kappa = np.exp(np.log(m) + ln_fi)            # kink (j-i)f(i), via logs; underflows to 0 naturally
    sp_ = md                                     # split: lo..sp_ increasing, sp_+1..hi decreasing
    if sp_ >= j: sp_ = j - 1
    if sp_ < i: sp_ = i
    if sp_ < lo: sp_ = lo
    if sp_ > hi: sp_ = hi
    dp, df0, dstep = _err_bounds(n, lnC, i, j, md, q, lnp, lnq)
    a = lo; b = hi; r = nz                       # two-pointer merge; ties: j before i
    neartie = False; prev = 0.0 if nz > 0 else -1.0
    # Sharpened trigger.  Masses whose order double precision cannot resolve are grouped into
    # maximal clusters; within a cluster of size c any rank can move by at most c-1, so the total
    # possible perturbation of S_- is bounded by (c-1)*sum_{k in C} f_k |k - n p*|, summed over
    # clusters.  Adjacent masses are unresolvable when their relative gap is within the two masses'
    # own error bounds plus the differential effect of dp: d(ln f_k - ln f_l)/dp = (k-l)/(pq), so an
    # error p*dp in p* moves the ratio by dp*|k-l|/q.  The cluster {i,j} alone contributes NOTHING:
    # f(i)=f(j) exactly and w_j = w_i - 1 is the left-limit ranking by definition, not a numerical
    # guess.  A third mass joining them makes the cluster count in full, which is conservative.
    rbnd = 0.0; csum = 0.0; csize = 0; conly_ij = True; previdx = -1
    while a <= sp_ or b > sp_:
        if a > sp_: take_left = False
        elif b <= sp_: take_left = True
        elif f[a] < f[b]: take_left = True
        elif f[a] > f[b]: take_left = False
        else: take_left = not (a == i and b == j)
        if take_left: k = a; a += 1
        else:         k = b; b -= 1
        w[k] = r
        if r > 0 and f[k] > 0.0 and not ((k == i and prev == f[j]) or (k == j and prev == f[i])):
            if (f[k] - prev)/f[k] < GAP and not (k == i or k == j): neartie = True
            if (k == i or k == j) and (f[k]-prev)/f[k] < GAP and prev != f[k]: neartie = True
        amb = False
        if previdx >= 0 and f[k] > 0.0:
            thr = (2.0*df0 + dstep*(abs(k-md) + abs(previdx-md))
                   + dp*abs(k-previdx)/q)
            if (f[k] - prev)/f[k] <= thr: amb = True
        t_ka = f[k]*abs(k - n*p)
        if amb:
            csize += 1; csum += t_ka
            if not (k == i or k == j): conly_ij = False
        else:
            if csize > 1 and not conly_ij: rbnd += (csize-1)*csum
            csize = 1; csum = t_ka; conly_ij = (k == i or k == j)
        previdx = k
        prev = f[k]; r += 1
    if csize > 1 and not conly_ij: rbnd += (csize-1)*csum
    Sm = 0.0; E = 0.0; ce = 0.0
    for k in range(lo, hi+1):
        Sm += w[k]*f[k]*(k - n*p)
        t = w[k]*f[k]
        u = E + t                                # Neumaier for E
        if abs(E) >= abs(t): ce += (E - u) + t
        else:                ce += (t - u) + E
        E = u
    E = E + ce
    Sp = Sm + kappa
    ismin = (Sm < -MARGIN) and (Sp > MARGIN)
    isnot = (Sm > MARGIN) or (Sp < -MARGIN)
    if neartie or not (ismin or isnot): tag_old = TAG_CHECK
    elif ismin:                          tag_old = TAG_MIN
    else:                                tag_old = TAG_NOT
    # Sharpened tag: the same margin test, widened by the re-ranking bound.  rbnd perturbs S_- and
    # S_+ equally (kappa is exact), so it enters both sides.  No new constant is introduced.
    mg = MARGIN + rbnd
    smin = (Sm < -mg) and (Sp > mg)
    snot = (Sm > mg) or (Sp < -mg)
    if not (smin or snot): tag_new = TAG_CHECK
    elif smin:             tag_new = TAG_MIN
    else:                  tag_new = TAG_NOT
    F3 = (n+i-j)*(i+j-2*n*p) + (j-n*p)
    return p, ln_fi, E, Sm, kappa, F3, tag_old, tag_new, rbnd

@njit(cache=True)
def tie_kernel(n, lnC, collect_all, i_lo, i_hi, sharp,
               out_i, out_j, out_p, out_lnf, out_E, out_Sm, out_F3, out_tag,
               out_tag2, out_rbnd):
    """Screen every tie point of n (i<j<=n, i+j>n).  Returns the number of rows written.

    collect_all=False writes only MIN/CHECK rows (the certified generator's path);
    True writes every tie point (the Parquet export's path).  A count larger than the array
    length means the buffers were too small -- retry with bigger ones.
    i_lo/i_hi restrict the outer loop, so one n can be split across processes; each tie point is
    computed identically regardless of how the range is cut.
    sharp selects which trigger drives out_tag and the collect_all=False filter: True (the
    DEFAULT since 2026-09-21) is the re-ranking cluster bound, False the historical near-tie proxy
    (relative GAP between adjacent masses), kept only so validate_trigger.py can compare them.  BOTH tags are always written -- out_tag is the selected one, out_tag2 the other -- along
    with the bound itself in out_rbnd, so a single pass can compare the two rules.
    """
    cap = out_i.shape[0]
    f = np.empty(n+1); w = np.empty(n+1, np.int64); cnt = 0
    for i in range(i_lo, i_hi):
        for j in range(max(i+1, n-i+1), n+1):        # j <= n: the pairs (i,n) are real tie points
            p, ln_fi, E, Sm, kappa, F3, tag_old, tag_new, rbnd = _one_tie(n, lnC, i, j, f, w)
            tag = tag_new if sharp else tag_old
            if collect_all or tag != TAG_NOT:
                if cnt < cap:
                    out_i[cnt] = i; out_j[cnt] = j; out_p[cnt] = p; out_lnf[cnt] = ln_fi
                    out_E[cnt] = E; out_Sm[cnt] = Sm; out_F3[cnt] = F3; out_tag[cnt] = tag
                    out_tag2[cnt] = tag_old if sharp else tag_new
                    out_rbnd[cnt] = rbnd
                cnt += 1
    return cnt

@njit(cache=True)
def _first_i(lnC, m, a, b, p_lo, strict):
    """Smallest i in [a, b] with p*(i, i+m) >= p_lo (> p_lo if strict), or b+1.  p*(i, i+m) is
    strictly increasing in i: (p*/q*)^m = prod_{t=i+1}^{i+m} t/(n+1-t), every factor increasing."""
    while a <= b:
        c = (a + b) // 2
        p = _pstar(lnC, c, c + m)[0]
        if p > p_lo or (p == p_lo and not strict): b = c - 1
        else: a = c + 1
    return a

@njit(cache=True)
def window_kernel(n, lnC, collect_all, p_lo, p_hi, min_ln_fi, sharp,
                  out_i, out_j, out_p, out_lnf, out_E, out_Sm, out_F3, out_tag,
                  out_tag2, out_rbnd):
    """tie_kernel restricted to the tie points with p_lo <= p* <= p_hi (p* > 1/2 as always).

    For each width m = j - i, p* is increasing in i, so a binary search on the kernel's own p*
    (_pstar) finds exactly the pairs in the window; each goes through _one_tie unchanged, so every
    row is bit-identical to tie_kernel's.  Cost: O(n log n) for the search plus _one_tie on the
    pairs found, about n^2 (p_hi - p_lo) of them, instead of all n^2/4.
    min_ln_fi > -inf also skips pairs whose (unnormalised) ln f(i) at p* is below it, without
    computing them: see screen().  Rows come out by width, then i.
    """
    cap = out_i.shape[0]
    f = np.empty(n+1); w = np.empty(n+1, np.int64); cnt = 0
    for m in range(1, n):
        a = (n - m)//2 + 1                           # i + j > n, i.e. 2i + m > n
        b = n - m                                    # j <= n
        if a < 1: a = 1
        if a > b: continue
        lo = _first_i(lnC, m, a, b, p_lo, False)
        hi = _first_i(lnC, m, a, b, p_hi, True) - 1
        for i in range(lo, hi + 1):
            j = i + m
            if min_ln_fi > -np.inf:
                p = _pstar(lnC, i, j)[0]
                if lnC[i] + i*np.log(p) + (n-i)*np.log(1.0 - p) < min_ln_fi: continue
            p, ln_fi, E, Sm, kappa, F3, tag_old, tag_new, rbnd = _one_tie(n, lnC, i, j, f, w)
            tag = tag_new if sharp else tag_old
            if collect_all or tag != TAG_NOT:
                if cnt < cap:
                    out_i[cnt] = i; out_j[cnt] = j; out_p[cnt] = p; out_lnf[cnt] = ln_fi
                    out_E[cnt] = E; out_Sm[cnt] = Sm; out_F3[cnt] = F3; out_tag[cnt] = tag
                    out_tag2[cnt] = tag_old if sharp else tag_new
                    out_rbnd[cnt] = rbnd
                cnt += 1
    return cnt

@njit(cache=True)
def window_count(n, lnC, p_lo, p_hi):
    """Number of tie points with p_lo <= p* <= p_hi (p* > 1/2), from the binary searches alone."""
    c = 0
    for m in range(1, n):
        a = (n - m)//2 + 1
        b = n - m
        if a < 1: a = 1
        if a > b: continue
        c += _first_i(lnC, m, a, b, p_hi, True) - _first_i(lnC, m, a, b, p_lo, False)
    return c

def ties_in_range(n, i_lo, i_hi):
    return sum(max(0, n - max(i+1, n-i+1) + 1) for i in range(i_lo, i_hi))

def work_chunks(n, parts):
    """Split i in [1,n) into `parts` ranges of roughly equal work (work per i ~ number of valid j)."""
    r = np.array([max(0, n - max(i+1, n-i+1) + 1) for i in range(1, n)])
    c = np.cumsum(r); total = c[-1]
    out = []; lo = 1
    for t in range(1, parts+1):
        hi = int(np.searchsorted(c, total*t/parts)) + 2
        hi = min(hi, n)
        if hi > lo: out.append((lo, hi)); lo = hi
    if lo < n: out.append((lo, n))
    return [(a, b) for a, b in out if ties_in_range(n, a, b) > 0]

def screen(n, collect_all=False, lnC=None, i_lo=1, i_hi=None, sharp=True, p_range=None,
           min_pair_mass=None):
    """tie_kernel with buffer management.  Returns a dict of numpy arrays.

    p_range=(p_lo, p_hi) screens only the tie points with p_lo <= p* <= p_hi, through window_kernel
    (i_lo/i_hi do not apply).  The rows are exactly tie_kernel's for those pairs, bit for bit, in
    another order (by width, then i).
    min_pair_mass (with p_range only) also skips every pair whose mass f(i) at p* is below it.
    That is a SPEED-UP, NOT A PROOF: a cusp needs S_- < 0 < S_- + (j-i) f(i), so a tiny pair mass
    makes a cusp very unlikely but not impossible, and no lower bound on the pair mass of a cusp is
    proved.  Measured: the smallest pair mass at any cusp of n <= 5000 is 4.4e-9 (n = 4000..5000),
    so 1e-20 leaves a wide margin there; it skips ~80% of pairs at n = 5000 and ~90% at 20,000.
    Tables built with it may miss cusps in principle.  The test uses the unnormalised mass
    (normalisation shifts ln f(i) by ~1e-12).

    sharp picks the CHECK trigger (see tie_kernel); it DEFAULTS TO THE SHARPENED ONE.  The
    returned dict always carries both verdicts: 'tag' from the selected rule and 'tag_alt' from the
    other, plus 'rbnd'.  Anything comparing the two rules must pass sharp explicitly rather than
    relying on which one 'tag' happens to hold.
    """
    lnC = lnC_arr(n) if lnC is None else lnC
    i_hi = n if i_hi is None else i_hi
    if min_pair_mass is not None and p_range is None:
        raise ValueError("min_pair_mass applies only with p_range")
    if p_range is not None:
        p_lo, p_hi = float(p_range[0]), float(p_range[1])
        min_ln_fi = -np.inf if min_pair_mass is None else float(np.log(min_pair_mass))
        cap = window_count(n, lnC, p_lo, p_hi) if collect_all else max(64, 4*n)
    else:
        cap = ties_in_range(n, i_lo, i_hi) if collect_all else max(64, 4*n)
    while True:
        a = dict(i=np.empty(cap, np.int64), j=np.empty(cap, np.int64), pstar=np.empty(cap),
                 ln_fi=np.empty(cap), E=np.empty(cap), S_minus=np.empty(cap),
                 F3=np.empty(cap), tag=np.empty(cap, np.int64),
                 tag_alt=np.empty(cap, np.int64), rbnd=np.empty(cap))
        outs = (a['i'], a['j'], a['pstar'], a['ln_fi'], a['E'], a['S_minus'], a['F3'],
                a['tag'], a['tag_alt'], a['rbnd'])
        if p_range is not None:
            c = window_kernel(n, lnC, collect_all, p_lo, p_hi, min_ln_fi, sharp, *outs)
        else:
            c = tie_kernel(n, lnC, collect_all, i_lo, i_hi, sharp, *outs)
        if c <= cap: break
        cap = 2*c
    return {k: v[:c] for k, v in a.items()}

def axis_point(n):
    """The tie point at p=1/2 itself, where ALL mirror pairs (i,n-i) tie simultaneously.

    Returns (S_minus, S_plus, E, kappa, n_tied_pairs).  This is not an ordinary tie point: the
    i+j>n filter excludes p=1/2 precisely because it IS the symmetry axis, and the single-pair
    bookkeeping S_+ = S_- + (j-i)f(i) does not apply -- every mirror pair contributes to the kink
    at once.  What does apply is the symmetry E(p) = E(1-p), which gives E'(1/2-) = -E'(1/2+)
    exactly, hence S_- = -S_+ and u = S_-/kappa = -1/2 exactly: the zero sits dead centre in the
    slope jump, making p=1/2 the most robust cusp there is.

    The ranking must be built, not sorted for: f(k) and f(n-k) are equal in exact arithmetic but
    differ in the last ulp via lgamma, so a stable sort orders them by numerical noise and gets the
    sign of S_+ wrong (it did, for 26 values of n, until the masses were symmetrised first).  Just
    to the right of 1/2 the larger index carries the larger mass, so ties break to smaller index
    first.
    """
    import math
    k = np.arange(n+1)
    f = np.exp(lnC_arr(n) - n*math.log(2.0))
    f = 0.5*(f + f[::-1])                       # enforce f(k) = f(n-k) exactly
    f = f/math.fsum(f.tolist())
    o = np.lexsort((k, f))                      # by mass, ties to the smaller index first
    w = np.empty(n+1, np.int64); w[o] = np.arange(n+1)
    Sp = float(math.fsum((w*f*(k - n*0.5)).tolist()))
    E = float(math.fsum((w*f).tolist()))
    n_pairs = (n+1)//2 if n % 2 else n//2       # pairs (i,n-i) with 0<i<n-i<n, plus i=0 with j=n
    return -Sp, Sp, E, 2*Sp, n_pairs

def certify(n, i, j, dps, order=None):
    """Interval-arithmetic verdict for one tie point: 'MIN', 'NOT', or None if undecided.

    Scale-free: every decision here is invariant under a common positive scale (S_- < 0 < S_+, and
    the relative separations), so the masses are taken relative to f(i) = 1 and built by the same
    recurrence the double-precision kernel uses.  That avoids the binomial coefficients entirely --
    they were ~n-digit integers costing 72% of this routine at n=4000, and computing all n+1 of them
    is O(n^2) in bit complexity.  Cost drops from ~n^1.53 to ~n^1.05: 2.8x faster at n=2000,
    7.9x at n=5000.  S_+ = S_- + (j-i) exactly, since f(i) = 1 in these units.

    order: the ranking from the double-precision screen, if known.  It is VERIFIED here either way
    (each adjacent pair must be separated in interval arithmetic), so passing it only skips a sort.
    """
    from mpmath import iv
    iv.dps = dps; m = j - i
    r = iv.mpf(1)
    for t in range(i+1, j+1):                  # C(n,i)/C(n,j) = prod_{t=i+1}^{j} t/(n+1-t)
        r = r * iv.mpf(t) / iv.mpf(n+1-t)
    rho = r**(iv.mpf(1)/m)
    p = rho/(1+rho); q = 1-p
    g = [iv.mpf(0)]*(n+1)
    g[i] = iv.mpf(1)
    for k in range(i, 0, -1):                  # leftwards
        g[k-1] = g[k] * iv.mpf(k) / (iv.mpf(n-k+1)*rho)
    for k in range(i, n):                      # rightwards
        g[k+1] = g[k] * rho * iv.mpf(n-k) / iv.mpf(k+1)
    g[j] = g[i]                                # exact tie
    if order is None:
        order = sorted(range(n+1),
                       key=lambda k: (g[i].mid if k in (i, j) else g[k].mid, 0 if k == j else 1))
    for a, b in zip(order, order[1:]):
        if {a, b} == {i, j}: continue
        if not (g[a].b < g[b].a): return None
    w = [0]*(n+1)
    for t, k in enumerate(order): w[k] = t
    Sm = iv.mpf(0)
    for k in range(n+1):
        if g[k].b == 0: continue               # underflowed to exactly zero: contributes nothing
        Sm = Sm + w[k]*g[k]*(k - n*p)
    Sp = Sm + m                                # (j-i)*g_i with g_i = 1
    if Sm.b < 0 and Sp.a > 0: return 'MIN'
    if Sm.a >= 0 or Sp.b <= 0: return 'NOT'
    return None

def certify_escalating(n, i, j, dps_seq=(50, 100, 200)):
    """certify() at increasing precision.  Returns (verdict or None, 'iv50'/... or 'double')."""
    for dps in dps_seq:
        v = certify(n, i, j, dps)
        if v: return v, f'iv{dps}'
    return None, 'double'

def evaluate(n, i, j, lnC=None):
    """Descriptive values at one tie point: (p*, E, F3, S_-, S_+, slope_left, slope_right).

    Served by _one_tie, so there is exactly one mass computation in the project.  It used to
    recompute the masses itself, unnormalised and ranked with lexsort, which cost ~3 digits on
    E - E(1/2) and was a second implementation of the same mathematics.
    """
    lnC = lnC_arr(n) if lnC is None else lnC
    f = np.empty(n+1); w = np.empty(n+1, np.int64)
    p, ln_fi, E, Sm, kappa, F3, tag_old, tag_new, rbnd = _one_tie(n, lnC, i, j, f, w)
    q = 1 - p; Sp = Sm + kappa
    return p, E, F3, Sm, Sp, Sm/(p*q), Sp/(p*q)

def recheck(n, i, j, dps=50):
    """Print every quantity at one tie point from the rigorous reference (``reference.tie``).

    For the ordered-binomial-cusps CLI (``cusps_fast.py --recheck n i j``).  In code, call
    ``obd_core.reference.tie`` and use its ``Value`` fields instead of parsing this output.
    """
    from mpmath import mp, nstr
    from . import reference
    r = reference.tie(n, i, j, dps=dps)
    print(f"n={n} i={i} j={j}  ({r.method}; every value is a rigorous enclosure, shown as mid +/- rad)")
    with mp.workdps(dps):
        for name, v in (("p*", r.p), ("E", r.E), ("F3", r.F3), ("S_-", r.S_minus), ("S_+", r.S_plus),
                        ("slope_left", r.slope_left), ("slope_right", r.slope_right),
                        ("log10_D", r.log10_D)):
            if v is not None:
                print(f"  {name:12s} {nstr(v.mid, dps - 10)}   +/- {nstr(v.rad, 2)}")
    print("  cusp:", r.is_cusp)

@njit(cache=True)
def _E_slopes_one(n, lnC, p, tol, f, k_lo):
    """E, E'_- and E'_+ at one p in (0,1).  See E_slopes_at."""
    q = 1.0 - p; rho = p/q
    lnp = np.log(p); lnq = np.log(q)
    md = int(np.floor((n+1)*p))                  # mode of Bin(n,p)
    if md > n: md = n
    lo, hi = _masses(n, lnC, md, lnp, lnq, rho, md, md, f)
    s = 0.0; comp = 0.0                          # Neumaier sum, then rescale (as _one_tie)
    for k in range(lo, hi+1):
        t = s + f[k]
        if abs(s) >= abs(f[k]): comp += (s - t) + f[k]
        else:                   comp += (f[k] - t) + s
        s = t
    s = s + comp
    inv = 1.0/s
    for k in range(lo, hi+1): f[k] *= inv
    nz = lo + (n - hi)                           # masses outside the window are exactly zero: they
    w = hi - lo + 1                              # take ranks 0..nz-1 as an equal block, add nothing
    order = np.argsort(f[lo:hi+1], kind='mergesort') + lo
    # Group masses equal to within tol (relative): at a tie point they are equal in exact
    # arithmetic and their computed order is rounding noise.  Just LEFT of a tie the larger index
    # has the smaller mass (f_j/f_i grows with p for j > i), so the left ranking puts the larger
    # index lower inside a group; the right ranking does the opposite.
    El = 0.0; cE = 0.0; Sl = 0.0; cl = 0.0; Sr = 0.0; cr = 0.0
    a = 0
    while a < w:
        b = a
        while b < w - 1 and f[order[b+1]] - f[order[b]] <= tol*f[order[b+1]]:
            b += 1
        m = b - a + 1
        for t in range(m):
            k_lo[t] = order[a+t]
        k_lo[:m].sort()                          # indices ascending
        for t in range(m):
            kr = k_lo[t]                         # right ranking: ascending index
            kl = k_lo[m-1-t]                     # left ranking: descending index
            r = nz + a + t
            vl = r*f[kl]*(kl - n*p); vr = r*f[kr]*(kr - n*p); ve = r*f[kl]
            tt = Sl + vl
            if abs(Sl) >= abs(vl): cl += (Sl - tt) + vl
            else: cl += (vl - tt) + Sl
            Sl = tt
            tt = Sr + vr
            if abs(Sr) >= abs(vr): cr += (Sr - tt) + vr
            else: cr += (vr - tt) + Sr
            Sr = tt
            tt = El + ve
            if abs(El) >= abs(ve): cE += (El - tt) + ve
            else: cE += (ve - tt) + El
            El = tt
        a = b + 1
    pq = p*q
    return El + cE, (Sl + cl)/pq, (Sr + cr)/pq

def E_slopes_at(n, p):
    """E(n,p) and the exact one-sided slopes E'_-(p), E'_+(p) at every p of an array.

    Returns three float64 arrays (E, slope_left, slope_right).  The slopes come from the ranking,
    not from differences of E: E' = S/(p q) with S = sum_k w_k f(k)(k - n p).  Between tie points
    slope_left == slope_right == E'(p).  AT a tie point -- a grid often hits one exactly, e.g.
    p = 1/2 for every n, or (i+1)/(n+1) at a dyadic p -- E has a kink and the two differ by D.
    Masses within relative tol = 64 EPS (n+1)/(p q) of each other are treated as tied: that bounds
    both their own rounding error (~n EPS) and the spread a correctly rounded tie point leaves
    between two tied masses ((j-i) ulp(p)/(p q)).  It must stay that tight: tie points of
    different pairs can sit ~1e-12 apart (n=1000 has thousands), and a looser tolerance merges
    them and reports a kink that is not there.  So p is reported as AT a tie only within
    ~tol p q/(j-i) of it.
    Same mass convention as E_at (normalised by their own sum); E agrees with E_at to ~1e-15
    relative.  At p = 0 and p = 1, E = n and both slopes are the one-sided limits -n and +n.
    Descriptive, not certified.
    """
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    lnC = lnC_arr(n)
    E = np.empty(p.size); sl = np.empty(p.size); sr = np.empty(p.size)
    f = np.empty(n+1); k_lo = np.empty(n+1, np.int64)
    for t in range(p.size):
        pt = float(p[t])
        if pt <= 0.0:
            E[t], sl[t], sr[t] = n, -n, -n
        elif pt >= 1.0:
            E[t], sl[t], sr[t] = n, n, n
        else:
            tol = 64.0*EPS*(n + 1)/(pt*(1.0 - pt))
            E[t], sl[t], sr[t] = _E_slopes_one(n, lnC, pt, tol, f, k_lo)
    return E, sl, sr

# ---------------------------------------------------------------------------------------------
# Every tie point of one n, certified: the bulk table that dump_ties.py and the OBD repo build on.

def _screen_chunk(a):
    n, lo, hi = a
    return screen(n, collect_all=True, i_lo=lo, i_hi=hi)

def certify_exact(n, i, j):
    """Exact verdict for an ADJACENT pair j = i+1, whose tie point p* = (i+1)/(n+1) is rational.

    Interval arithmetic cannot settle S_- = 0 exactly, which happens (n=2, pair (1,2), p*=2/3).
    At a rational p* every mass is C(n,k)(i+1)^k (n-i)^(n-k) / (n+1)^n, so the signs of S_- and
    S_+ follow from integers: rank the numerators F_k (left of p*: w_j = w_i - 1) and evaluate
    (n+1) S_- (n+1)^n = sum_k w_k F_k ((n+1)k - n(i+1)), S_+ adding (n+1)(j-i) F_i.
    Returns 'MIN', 'NOT', or None (not adjacent, or another exact tie makes the ranking ambiguous).
    """
    if j != i + 1:
        return None
    a, b = i + 1, n - i                      # p* = a/(n+1), q* = b/(n+1)
    F = [comb(n, k) * a**k * b**(n - k) for k in range(n + 1)]
    others = [F[k] for k in range(n + 1) if k not in (i, j)]
    if len(set(others)) != len(others) or F[i] in others:
        return None
    order = sorted(range(n + 1), key=lambda k: (F[k], 0 if k == j else 1))
    w = [0]*(n + 1)
    for r, k in enumerate(order): w[k] = r
    Sm = sum(w[k]*F[k]*((n + 1)*k - n*a) for k in range(n + 1))
    Sp = Sm + (n + 1)*(j - i)*F[i]
    return 'MIN' if Sm < 0 < Sp else 'NOT'

def _certify_one(a):
    n, i, j = a
    v, how = certify_escalating(n, i, j)
    if v is None:
        v = certify_exact(n, i, j)
        how = 'exact' if v else 'UNRESOLVED'
    return (v == 'MIN'), how

def _as_windows(p_range):
    """p_range as a list of (lo, hi): one pair, or a sequence of pairs."""
    if len(p_range) == 2 and np.ndim(p_range[0]) == 0:
        return [(float(p_range[0]), float(p_range[1]))]
    return [(float(a), float(b)) for a, b in p_range]

def tie_table(n, both_halves=False, workers=1, pool=None, p_range=None, min_pair_mass=None):
    """Every tie point of n, sorted by p*, with certified cusp verdicts and exact slopes.

    Returns a dict of equal-length numpy arrays:
        i, j, pstar, ln_fi, E, S_minus, F3, tag     as from screen(n, collect_all=True)
        n_tied_pairs    1 for an ordinary tie point
        is_cusp         bool, certified: CHECK-tagged rows go through certify_escalating
        decided_by      'double', 'iv50'/'iv100'/'iv200', 'exact' (certify_exact, adjacent pairs),
                        'UNRESOLVED', or 'symmetry' (axis)
        slope_left, slope_right    E'_- and E'_+ at p*
        log10_D         log10 of the slope jump D = E'_+ - E'_- = (j-i) f(i) / (p* q*), from ln_fi
                        and so exact far below double range.  NEVER take D as slope_right -
                        slope_left: for most tie points D is many orders below the slopes and the
                        difference is 0 or noise.  slope_right = slope_left + D carries the same
                        loss, so it is right as a slope but useless for recovering D.

    Row 0 of the p* > 1/2 half is the symmetry axis p = 1/2, where all mirror pairs (i, n-i) tie at
    once.  It carries the sentinel (i, j) = (0, n), n_tied_pairs = the number of pairs, F3 = NaN,
    decided_by = 'symmetry', and ln_fi = ln(kappa/n) so that (j-i) exp(ln_fi) is its full kink
    kappa = 2 S_+ (see axis_point).

    both_halves=True adds the p* < 1/2 tie points by the symmetry E(p) = E(1-p): the mirror of
    (i, j) at p* is (n-j, n-i) at 1-p*, with the same E, ln_fi, D and verdict, and
    E'_-(1-p*) = -E'_+(p*), E'_+(1-p*) = -E'_-(p*).  F3 is NaN there (it is defined for p* > 1/2).
    The axis appears once.

    pool (a multiprocessing.Pool) with workers > 1 splits the screen into work-balanced i-chunks and
    distributes the certifications; the result is identical however it is cut.

    p_range=(p_lo, p_hi), or a list of such windows: only the rows with p* in a window (with
    both_halves, on either side of 1/2; the axis row only if 1/2 is in one).  Many windows in one
    call share the per-n setup (log binomials, the axis), which dominates for narrow windows.  Exactly the full table's rows there, bit for
    bit and in the same order, but only those tie points are computed (screen's p_range), so a
    narrow window costs a small fraction of the full table: about n^2 (p_hi - p_lo) tie points
    instead of n^2/4.  The pool then serves only the certifications.
    min_pair_mass (with p_range): skip pairs with f(i) below it, uncomputed.  A speed-up that is
    NOT proved safe for cusps: see screen().
    """
    if min_pair_mass is not None and p_range is None:
        raise ValueError("min_pair_mass applies only with p_range")
    if p_range is not None:
        wins = _as_windows(p_range)
        # The p* > 1/2 tie points needed: each window above 1/2, and the mirror of its part below;
        # merged so that no pair is screened twice.
        need = []
        for lo, hi in wins:
            if hi >= 0.5: need.append((max(lo, 0.5), hi))
            if both_halves and lo <= 0.5: need.append((1.0 - min(hi, 0.5), 1.0 - lo))
        merged = []
        for a, b in sorted(need):
            if merged and a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], b)
            else: merged.append([a, b])
        lnC = lnC_arr(n)                                # once for every window
        parts = [screen(n, collect_all=True, lnC=lnC, p_range=(a, b), min_pair_mass=min_pair_mass)
                 for a, b in (merged or [[2.0, 2.0]])]
        r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for k in ('tag_alt', 'rbnd'): r.pop(k, None)
        # tie_kernel emits i-major, j-minor, and the full table sorts that stably by p*; same order:
        o = np.lexsort((r['j'], r['i'], r['pstar']))
    else:
        if pool is not None and workers > 1:
            parts = pool.map(_screen_chunk, [(n, a, b) for a, b in work_chunks(n, workers)])
            r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        else:
            r = screen(n, collect_all=True)
        for k in ('tag_alt', 'rbnd'): r.pop(k, None)   # trigger diagnostics
        o = np.argsort(r['pstar'], kind='stable')
    r = {k: v[o] for k, v in r.items()}
    aSm, aSp, aE, akap, apairs = axis_point(n)
    axis = dict(i=0, j=n, pstar=0.5, ln_fi=np.log(akap/n), E=aE, S_minus=aSm,
                F3=np.nan, tag=(TAG_MIN if aSm < 0 < aSp else 0))
    for k in r: r[k] = np.concatenate([np.array([axis[k]], dtype=r[k].dtype), r[k]])
    c = len(r['i'])
    n_pairs = np.ones(c, np.int16); n_pairs[0] = apairs
    decided = np.array(['double']*c, dtype=object)
    decided[0] = 'symmetry'                # settled exactly by E(p) = E(1-p)
    is_cusp = r['tag'] == TAG_MIN
    checks = np.flatnonzero(r['tag'] == TAG_CHECK)
    if len(checks):
        args = [(n, int(r['i'][t]), int(r['j'][t])) for t in checks]
        res = pool.map(_certify_one, args, chunksize=1) if pool is not None else [_certify_one(a) for a in args]
        for t, (cusp, how) in zip(checks, res):
            is_cusp[t] = cusp; decided[t] = how
    r['n_tied_pairs'] = n_pairs; r['is_cusp'] = is_cusp; r['decided_by'] = decided
    pq = r['pstar']*(1 - r['pstar'])
    r['slope_left'] = r['S_minus']/pq
    ln_D = np.log((r['j'] - r['i']).astype(float)) + r['ln_fi'] - np.log(pq)
    r['log10_D'] = ln_D/np.log(10.0)
    r['slope_right'] = r['slope_left'] + np.exp(ln_D)
    r['slope_right'][0] = aSp/pq[0]                    # exact at the axis: E'_+ = -E'_-
    if both_halves:
        m = {k: v[:0:-1] for k, v in r.items()}       # rows 1..c-1, reversed: p* descending
        m['i'], m['j'] = n - r['j'][:0:-1], n - r['i'][:0:-1]
        m['pstar'] = 1 - r['pstar'][:0:-1]
        m['slope_left'], m['slope_right'] = -r['slope_right'][:0:-1], -r['slope_left'][:0:-1]
        m['S_minus'] = m['slope_left']*(m['pstar']*(1 - m['pstar']))
        m['F3'] = np.full(c - 1, np.nan)
        r = {k: np.concatenate([m[k], r[k]]) for k in r}
    if p_range is not None:
        keep = np.zeros(len(r['pstar']), bool)
        for lo, hi in wins:
            keep |= (r['pstar'] >= lo) & (r['pstar'] <= hi)
        r = {k: v[keep] for k, v in r.items()}
    return r
