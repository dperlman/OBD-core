"""obd_core.store returns exactly what tie_table returns, computes only what is missing, and refuses stale data."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("pyarrow")

import obd_core as core
from obd_core import store as st


def _same(a, b):
    assert list(a) == list(b), "columns differ"
    for k in a:
        if b[k].dtype == object:
            assert list(a[k]) == list(b[k]), k
        else:
            assert a[k].dtype == b[k].dtype, k
            assert np.array_equal(a[k], b[k], equal_nan=True), k


def _restrict(t, lo, hi):
    keep = (t["pstar"] >= lo) & (t["pstar"] <= hi)
    return {k: v[keep] for k, v in t.items()}


@pytest.fixture
def s(tmp_path):
    return st.Store(tmp_path / "store")


@pytest.mark.parametrize("n", [3, 7, 50, 135, 300])
def test_full_table_round_trip(s, n):
    full = core.tie_table(n)
    _same(s.ties(n), full)                      # computed and saved
    _same(s.ties(n), full)                      # read back
    _same(s.cusps(n), {k: v[full["is_cusp"]] for k, v in full.items()})


def test_second_request_computes_nothing(s, monkeypatch):
    first = s.ties(200)

    def boom(*a, **k):
        raise AssertionError("tie_table called for stored data")
    monkeypatch.setattr(core, "tie_table", boom)
    _same(s.ties(200), first)
    _same(s.ties(200, p_range=(0.55, 0.6)), _restrict(first, 0.55, 0.6))
    _same(s.cusps(200), {k: v[first["is_cusp"]] for k, v in first.items()})


def test_windows_fill_only_the_gaps(s, monkeypatch):
    n = 300
    full = core.tie_table(n)
    _same(s.ties(n, p_range=(0.55, 0.6)), _restrict(full, 0.55, 0.6))
    calls = []
    real = core.tie_table

    def spy(n_, **k):
        calls.append(k.get("p_range"))
        return real(n_, **k)
    monkeypatch.setattr(core, "tie_table", spy)
    _same(s.ties(n, p_range=(0.58, 0.65)), _restrict(full, 0.58, 0.65))
    assert calls == [[(0.6, 0.65)]]                 # only the uncovered part was computed
    _same(s.ties(n, p_range=(0.5, 1.0)), full)      # edges shared by parts are not duplicated
    assert s.covered("ties", n) == [[0.5, 1.0]]


def test_cusps_use_stored_ties_and_large_tables_stay_optional(s, monkeypatch):
    n = 250
    full = core.tie_table(n)
    _same(s.cusps(n), {k: v[full["is_cusp"]] for k, v in full.items()})
    assert s.ns("ties") == [] and s.ns("cusps") == [n]          # cusps alone: no tie table kept
    s.ties(n)
    monkeypatch.setattr(core, "tie_table", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    _same(s.cusps(n, p_range=(0.6, 0.7)), _restrict({k: v[full["is_cusp"]] for k, v in full.items()}, 0.6, 0.7))


def test_columns_subset(s):
    t = s.ties(120, columns=["pstar", "S_minus", "is_cusp"])
    assert list(t) == ["pstar", "S_minus", "is_cusp"]
    full = core.tie_table(120)
    assert np.array_equal(t["S_minus"], full["S_minus"])


def test_stale_epoch_is_refused_and_check_restamps_good_data(s, monkeypatch):
    n = 100
    s.ties(n)
    path = os.path.join(s._dir("ties", n), "coverage.json")
    cov = json.load(open(path))
    cov["parts"][0]["numeric_epoch"] = core.NUMERIC_EPOCH - 1      # pretend it is from an older epoch
    json.dump(cov, open(path, "w"))
    with pytest.raises(st.StaleData):
        s.ties(n)
    r = s.check("ties", ns=[n])                     # the data still agrees, so it is re-stamped
    assert r["failures"] == [] and r["restamped_parts"] == 1
    _same(s.ties(n), core.tie_table(n))
    assert json.load(open(os.path.join(s.root, "STORE.json")))["checks"][-1]["n_checked"] == 1


def test_check_reports_corrupt_data(s):
    import pyarrow.parquet as pq

    n = 80
    s.ties(n)
    part = os.path.join(s._dir("ties", n), s.coverage("ties", n)["parts"][0]["file"])
    t = pq.read_table(part)
    e = t["E"].to_numpy().copy()
    e[5] += 1e-9
    import pyarrow as pa
    pq.write_table(t.set_column(t.schema.get_field_index("E"), "E", pa.array(e)), part)
    r = s.check("ties", ns=[n])
    assert r["failures"] and r["failures"][0]["n"] == n


def test_mass_threshold_parts_only_serve_thresholded_requests(s):
    n, lo, hi, tau = 1000, 0.6, 0.6005, 1e-20
    light = s.ties(n, p_range=(lo, hi), min_pair_mass=tau)
    assert s.covered("ties", n) == []                         # not complete for an exact request
    assert s.covered("ties", n, min_pair_mass=tau) == [[lo, hi]]
    exact = s.ties(n, p_range=(lo, hi))
    _same(exact, core.tie_table(n, p_range=(lo, hi)))
    assert len(light["i"]) <= len(exact["i"])


def test_interrupted_write_leaves_nothing_used(s):
    n = 60
    d = s._dir("ties", n)
    os.makedirs(d)
    open(os.path.join(d, "part-0.500000-1.000000-1.parquet.tmp"), "w").write("half a file")
    _same(s.ties(n), core.tie_table(n))             # the stray temp file is ignored, the table recomputed


def test_gaps():
    assert st._gaps(0.5, 1.0, []) == [[0.5, 1.0]]
    assert st._gaps(0.5, 1.0, [[0.6, 0.7]]) == [[0.5, 0.6], [0.7, 1.0]]
    assert st._gaps(0.6, 0.65, [[0.5, 0.7]]) == []
    assert st._gaps(0.55, 0.65, [[0.5, 0.6], [0.6, 0.62]]) == [[0.62, 0.65]]
    assert st._gaps(0.6, 0.6, []) == [[0.6, 0.6]]


def test_several_windows_in_one_request(s, monkeypatch):
    n = 300
    full = core.tie_table(n)
    wins = [(0.62, 0.64), (0.55, 0.56), (0.555, 0.57)]
    keep = np.zeros(len(full["pstar"]), bool)
    for lo, hi in wins:
        keep |= (full["pstar"] >= lo) & (full["pstar"] <= hi)
    calls = []
    real = core.tie_table
    monkeypatch.setattr(core, "tie_table", lambda n_, **k: (calls.append(k.get("p_range")), real(n_, **k))[1])
    _same(s.ties(n, p_range=wins), {k: v[keep] for k, v in full.items()})
    assert len(calls) == 1                                  # one tie_table call for all the windows
    _same(s.cusps(n, p_range=wins), {k: v[keep & full["is_cusp"]] for k, v in full.items()})


def _screen_positions(n, lo, hi):
    t = core.screen(n, collect_all=True, p_range=(lo, hi))
    o = np.argsort(t["pstar"], kind="stable")
    return {k: t[k][o] for k in ("i", "j", "pstar")}


@pytest.mark.parametrize("n", [40, 300])
def test_positions_are_screens_tie_points(s, n):
    got = s.positions(n, p_range=(0.55, 0.6))
    want = _screen_positions(n, 0.55, 0.6)
    assert sorted(zip(got["i"].tolist(), got["j"].tolist())) == sorted(zip(want["i"].tolist(), want["j"].tolist()))
    assert np.array_equal(np.sort(got["pstar"]), np.sort(want["pstar"]))
    assert np.all(np.diff(got["pstar"]) >= 0)


def test_positions_from_stored_ties_equal_screened_ones(s, tmp_path, monkeypatch):
    n = 300
    s.ties(n)
    monkeypatch.setattr(core, "screen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("screened")))
    derived = s.positions(n, p_range=(0.5, 1.0))
    monkeypatch.undo()
    other = st.Store(tmp_path / "other").positions(n, p_range=(0.5, 1.0))
    assert sorted(zip(derived["i"].tolist(), derived["j"].tolist(), derived["pstar"].tolist())) == \
        sorted(zip(other["i"].tolist(), other["j"].tolist(), other["pstar"].tolist()))
    assert 0 not in derived["i"]                            # no axis row, as screen gives none
    assert s.ns("positions") == []                          # read from the ties, not duplicated
    o = st.Store(tmp_path / "other")
    assert o.check("positions", ns=[n])["failures"] == []
