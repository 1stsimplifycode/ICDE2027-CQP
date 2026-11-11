"""Numerical-soundness tests for exact (integer) certification. Run: python -m pytest -q cqp/test_numeric.py"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import CQP

BIG = 10**12          # large financial magnitudes (sums of order 1e13)


def one_key(vals, theta, w=None, exact=True):
    """vals: list of (lo,hi) for src 0 over days; src 1 exact zero."""
    T = len(vals); w = w or T
    rows = [(0, 0, d, lo, hi) for d, (lo, hi) in enumerate(vals)] + [(0, 1, d, 0, 0) for d in range(T)]
    e = CQP(exact=exact); e.load([(0, 0)], rows); e.register_query(1, "stock", w, 1.0, {0: theta})
    return e


def state(e, end):
    return e.con.execute("SELECT state FROM decisions WHERE end_day=?", [end]).fetchone()[0]


def test_lower_bound_equality_is_not_true():
    e = one_key([(BIG, BIG), (BIG + 1, BIG + 1), (0, 5)], theta=2 * BIG + 1)   # L == theta exactly
    assert state(e, 2) == -1                                                    # L > theta is false; U > theta


def test_upper_bound_equality_is_false():
    e = one_key([(BIG, BIG), (BIG + 1, BIG + 1), (0, 5)], theta=2 * BIG + 6)   # U == theta exactly
    assert state(e, 2) == 0


def test_one_integer_step_each_side():
    v = [(BIG, BIG), (BIG + 1, BIG + 1), (0, 5)]
    assert state(one_key(v, theta=2 * BIG), 2) == 1            # L = theta + 1
    assert state(one_key(v, theta=2 * BIG + 5), 2) == -1       # U = theta + 1
    assert state(one_key(v, theta=2 * BIG + 0.5), 2) == 1       # representable non-integer threshold below L
    # a threshold literal is taken at its exact binary value: 2e12+0.999999 IS 2e12+1.0 as a double, so L == theta
    assert float(2 * BIG + 0.999999) == 2 * BIG + 1 and state(one_key(v, theta=2 * BIG + 0.999999), 2) == -1


def test_fractional_threshold_and_outward_rounding():
    # approximate reading [10.2, 12.7] -> integer hull [11, 12]; theta 11.5: L=11 !> 11.5, U=12 > 11.5 -> insufficient
    assert state(one_key([(10.2, 12.7)], theta=11.5), 0) == -1
    assert state(one_key([(10.2, 12.7)], theta=10.99), 0) == 1   # 11 > 10.99
    assert state(one_key([(10.2, 12.7)], theta=12.0), 0) == 0    # 12 <= 12


def test_near_zero():
    assert state(one_key([(0, 0), (0, 0)], theta=0), 1) == 0       # sum 0 <= 0
    assert state(one_key([(0, 1), (0, 0)], theta=0), 1) == -1
    assert state(one_key([(1, 1), (0, 0)], theta=0), 1) == 1


def test_exact_agreement_after_many_updates_and_contradictions():
    rng = np.random.default_rng(1); K, T = 8, 40
    truth = rng.integers(10**9, 10**11, (K, 2, T))
    rows = [(k, s, d, 0, 2 * 10**11) for k in range(K) for s in range(2) for d in range(T)]
    for mode in ("inc", "inc_scan", "full"):
        e = CQP(mode=mode); e.load([(k, k % 3) for k in range(K)], rows)
        for q, (lvl, w) in enumerate([("stock", 5), ("stock", 20), ("group", 7)], 1):
            ents = range(K) if lvl == "stock" else range(3)
            e.register_query(q, lvl, w, 1.0, {x: float(rng.integers(1, 4)) * w * 5 * 10**10 + 0.5 for x in ents})
        e.emit_initial()
        for step in range(30):
            b = {(int(rng.integers(0, K)), int(rng.integers(0, 2)), int(rng.integers(0, T))) for _ in range(25)}
            out = []
            for k, s, d in b:
                v = int(truth[k, s, d])
                if step % 7 == 6: v = v // 3                                   # contradictory restatement
                lo, hi = (v, v) if rng.random() < 0.6 else (math.floor(v * 0.9), math.ceil(v * 1.1))
                out.append((k, s, d, lo, hi))
            e.ingest(out)
        assert e.max_incremental_error() == 0                                    # exact, not approximate
        # independent Python-integer reference of every state
        ev = {(k, s, d): (lo, hi) for k, s, d, lo, hi in e.con.execute("SELECT key,src,day,lo,hi FROM evidence").fetchall()}
        for qid, lvl, w, ent, end, st, th in e.con.execute(
                "SELECT q.qid,q.level,q.w,d.ent,d.end_day,d.state,t.theta FROM decisions d JOIN queries q USING(qid) JOIN thresholds t USING(qid,ent)").fetchall():
            ks = [ent] if lvl == "stock" else [k for k in range(K) if k % 3 == ent]
            L = sum(ev[(k, s, d)][0] for k in ks for s in (0, 1) for d in range(end - w + 1, end + 1))
            U = sum(ev[(k, s, d)][1] for k in ks for s in (0, 1) for d in range(end - w + 1, end + 1))
            ref = 1 if L > th else (0 if U <= th else -1)
            assert ref == st
        assert e.con.execute("SELECT COUNT(*) FROM retraction_log").fetchone()[0] > 0


def test_non_integer_exact_evidence_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        one_key([(10.5, 10.5)], theta=3)
