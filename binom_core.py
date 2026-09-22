"""
binom_core.py -- the shared mathematics for the ordered-binomial cusp project.

Every other script imports from here; nothing below is duplicated elsewhere.  Definitions:

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
S_- < -MARGIN and S_+ > MARGIN (MIN), or S_- > MARGIN or S_+ < -MARGIN (NOT).  Anything else, or
any tie point where two adjacent masses in the ranking (both >= TINY) are within relative GAP,
is tagged CHECK and must go to certify().  Masses below TINY are treated as zero: without that the
CHECK count explodes with n (182,542 at n=2000 instead of ~75).

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

MARGIN, GAP, TINY = 1e-6, 1e-8, 1e-290
LN_TINY = _log(TINY)                 # window test in _one_tie; do not hardcode this
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

@njit(cache=True)
def _one_tie(n, lnC, i, j, f, w):
    """All quantities for a single tie point.  The ONLY place the masses and ranks are computed.

    Returns (p, ln_fi, E, S_minus, kappa, F3, tag).  Only the masses at or above TINY are touched
    (see the window comment in the body); the rest are exactly zero and change no result.
    Masses are always normalised by their own sum:
    unnormalised they carry a shared relative error ~6e-13 from exp/lgamma (they sum to
    0.999999999999363 at n=2000), which leaves only ~3.6 digits on E - E(1/2) instead of ~6.3.
    f and w are scratch buffers of length n+1, passed in so a loop can reuse them.
    """
    m = j - i
    lnrho = (lnC[i] - lnC[j]) / m
    p = 1.0/(1.0 + np.exp(-lnrho)); q = 1.0 - p; rho = p/q
    lnp = np.log(p); lnq = np.log(q)
    md = int(np.floor((n+1)*p))                  # mode of Bin(n,p)
    if md > n: md = n
    # Only the masses at or above TINY can affect anything: the rest are zeroed below, contribute
    # exactly 0.0 to every sum, and sit as an equal block at the bottom of the ranking.  The masses
    # fall away monotonically from the mode, so once the recurrence drops below TINY it stays below
    # and we can stop -- giving a window [lo,hi] of width O(sqrt(n)) instead of n+1.
    # The one trap: f[j] = f[i] below is applied AFTER zeroing, so it can rescue a j that fell just
    # under TINY.  f(i) is known in closed form, so when the pair is above TINY we refuse to stop
    # before reaching i on the left and j on the right, and the rescue still happens.
    pair_in = (lnC[i] + i*lnp + (n-i)*lnq) >= LN_TINY
    lo_req = i if pair_in else md
    hi_req = j if pair_in else md
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
    f[j] = f[i]                                  # exact tie
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
    a = lo; b = hi; r = nz                       # two-pointer merge; ties: j before i
    neartie = False; prev = 0.0 if nz > 0 else -1.0
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
        prev = f[k]; r += 1
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
    if neartie or not (ismin or isnot): tag = TAG_CHECK
    elif ismin:                          tag = TAG_MIN
    else:                                tag = TAG_NOT
    F3 = (n+i-j)*(i+j-2*n*p) + (j-n*p)
    return p, ln_fi, E, Sm, kappa, F3, tag

@njit(cache=True)
def tie_kernel(n, lnC, collect_all, i_lo, i_hi,
               out_i, out_j, out_p, out_lnf, out_E, out_Sm, out_F3, out_tag):
    """Screen every tie point of n (i<j<n, i+j>n).  Returns the number of rows written.

    collect_all=False writes only MIN/CHECK rows (the certified generator's path);
    True writes every tie point (the Parquet export's path).  A count larger than the array
    length means the buffers were too small -- retry with bigger ones.
    i_lo/i_hi restrict the outer loop, so one n can be split across processes; each tie point is
    computed identically regardless of how the range is cut.
    """
    cap = out_i.shape[0]
    f = np.empty(n+1); w = np.empty(n+1, np.int64); cnt = 0
    for i in range(i_lo, i_hi):
        for j in range(max(i+1, n-i+1), n+1):        # j <= n: the pairs (i,n) are real tie points
            p, ln_fi, E, Sm, kappa, F3, tag = _one_tie(n, lnC, i, j, f, w)
            if collect_all or tag != TAG_NOT:
                if cnt < cap:
                    out_i[cnt] = i; out_j[cnt] = j; out_p[cnt] = p; out_lnf[cnt] = ln_fi
                    out_E[cnt] = E; out_Sm[cnt] = Sm; out_F3[cnt] = F3; out_tag[cnt] = tag
                cnt += 1
    return cnt

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

def screen(n, collect_all=False, lnC=None, i_lo=1, i_hi=None):
    """tie_kernel with buffer management.  Returns a dict of numpy arrays."""
    lnC = lnC_arr(n) if lnC is None else lnC
    i_hi = n if i_hi is None else i_hi
    cap = ties_in_range(n, i_lo, i_hi) if collect_all else max(64, 4*n)
    while True:
        a = dict(i=np.empty(cap, np.int64), j=np.empty(cap, np.int64), pstar=np.empty(cap),
                 ln_fi=np.empty(cap), E=np.empty(cap), S_minus=np.empty(cap),
                 F3=np.empty(cap), tag=np.empty(cap, np.int64))
        c = tie_kernel(n, lnC, collect_all, i_lo, i_hi,
                       a['i'], a['j'], a['pstar'], a['ln_fi'], a['E'], a['S_minus'], a['F3'], a['tag'])
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
    p, ln_fi, E, Sm, kappa, F3, tag = _one_tie(n, lnC, i, j, f, w)
    q = 1 - p; Sp = Sm + kappa
    return p, E, F3, Sm, Sp, Sm/(p*q), Sp/(p*q)

def recheck(n, i, j, dps=50):
    """High-precision values for one tie point, printed."""
    from mpmath import mp, mpf, nstr
    mp.dps = dps; m = j-i
    rho = (mpf(comb(n,i))/comb(n,j))**(mpf(1)/m); p = rho/(1+rho); q = 1-p
    f = [comb(n,k)*p**k*q**(n-k) for k in range(n+1)]
    order = sorted(range(n+1), key=lambda k: (f[i] if k in (i,j) else f[k], 0 if k==j else 1))
    w = [0]*(n+1)
    for r, k in enumerate(order): w[k] = r
    Sm = sum(w[k]*f[k]*(k-n*p) for k in range(n+1)); Sp = Sm + m*f[i]
    E = sum(w[k]*f[k] for k in range(n+1)); F3 = (n+i-j)*(i+j-2*n*p) + (j-n*p)
    print(f"n={n} i={i} j={j}  ({dps} digits)")
    for name, v in (("p*",p),("E",E),("F3",F3),("S_-",Sm),("S_+",Sp),
                    ("slope_left",Sm/(p*q)),("slope_right",Sp/(p*q))):
        print(f"  {name:12s} {nstr(v, dps-10)}")
    print("  cusp:", Sm < 0 < Sp)
