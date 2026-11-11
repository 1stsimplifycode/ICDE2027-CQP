"""Tests for the CQP prototype. Run: python -m pytest -q cqp/test_cqp.py"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from engine import CQP
from acquire_sql import seaq_scores_sql, gr_plan
from cert import DecisionSet, UNK


def make(seed=0, K=6, T=30, G=2, p_miss=0.4, mode="inc"):
    rng = np.random.default_rng(seed)
    truth = rng.gamma(2, 50, (K, 2, T)).round(0)          # integer-valued domain (volumes)
    rows = []
    for k in range(K):
        for s in range(2):
            for d in range(T):
                v = truth[k, s, d]; u = rng.random()
                if u < p_miss: rows.append((k, s, d, 0.0, 500.0))
                elif u < p_miss + 0.3: rows.append((k, s, d, v * 0.9, v * 1.1))
                else: rows.append((k, s, d, v, v))
    e = CQP(mode=mode); e.load([(k, k % G) for k in range(K)], rows)
    med = np.median(truth.sum(1), axis=1)
    e.register_query(1, "stock", 5, 1.0, {k: 5 * med[k] for k in range(K)})
    e.register_query(2, "stock", 3, 1.5, {k: 3 * 1.5 * med[k] for k in range(K)})
    e.register_query(3, "group", 5, 1.0, {g: 5 * med[[k for k in range(K) if k % G == g]].sum() for g in range(G)})
    return e, truth, rows


def test_incremental_equals_full_recompute():
    e, truth, _ = make(1)
    rng = np.random.default_rng(5)
    for _ in range(10):
        b = [(int(rng.integers(0, 6)), int(rng.integers(0, 2)), int(rng.integers(0, 30))) for _ in range(15)]
        b = list({(k, s, d) for k, s, d in b})
        e.ingest([(k, s, d, truth[k, s, d], truth[k, s, d]) for k, s, d in b])
    assert e.max_incremental_error() < 1e-6


def test_states_match_reference_semantics():
    e, truth, rows = make(2)
    d = e.con.execute("""SELECT d.L, d.U, d.state, t.theta FROM decisions d JOIN thresholds t USING(qid,ent)""").fetchnumpy()
    ref = np.where(d["L"] > d["theta"], 1, np.where(d["U"] <= d["theta"], 0, -1))
    assert (ref == d["state"]).all()


def test_soundness_certified_never_wrong_with_sound_bounds():
    e, truth, _ = make(3)
    for (qid, level, w) in e.con.execute("SELECT qid, level, w FROM queries").fetchall():
        dec = e.con.execute("SELECT ent, end_day, state, t.theta FROM decisions JOIN thresholds t USING(qid,ent) WHERE qid=?", [qid]).fetchall()
        for ent, end, st, th in dec:
            ks = [ent] if level == "stock" else [k for k in range(6) if k % 2 == ent]
            tv = truth[ks, :, end - w + 1:end + 1].sum() > th
            if st == 1: assert tv
            if st == 0: assert not tv


def test_contradiction_detected_and_retraction_logged():
    e, truth, _ = make(4)
    e.emit_initial()
    row = e.con.execute("""SELECT d.qid, d.ent, d.end_day, d.state FROM decisions d JOIN queries q USING(qid)
                           WHERE q.level='stock' AND d.state=1 LIMIT 1""").fetchone()
    assert row is not None
    qid, k, end, _ = row
    # restate every item of that window to 0 (disjoint from any positive exact evidence) -> must retract
    items = e.con.execute("SELECT key,src,day,lo FROM evidence WHERE key=? AND day BETWEEN ? AND ?", [k, end - 4, end]).fetchall()
    r = e.ingest([(kk, s, d, 0.0, 0.0) for kk, s, d, lo in items])
    assert r["contradictions"] >= 1
    ret = e.con.execute("SELECT caused_by_contradiction FROM retraction_log WHERE qid=? AND ent=? AND end_day=?", [qid, k, end]).fetchall()
    assert ret and all(x[0] for x in ret)


def test_narrowing_never_flips_certified():
    e, truth, _ = make(5)
    e.emit_initial()
    rows = e.con.execute("SELECT key,src,day FROM evidence WHERE hi>lo").fetchall()
    e.ingest([(k, s, d, truth[k, s, d], truth[k, s, d]) for k, s, d in rows])     # sound narrowing
    assert e.con.execute("SELECT COUNT(*) FROM retraction_log").fetchone()[0] == 0


def test_seaq_sql_matches_numpy_reference():
    e, truth, _ = make(6)
    top = seaq_scores_sql(e, 10000)
    sql = {(k, s, d): sc for k, s, d, sc in top}
    ev = e.con.execute("SELECT key,src,day,lo,hi FROM evidence ORDER BY key,src,day").fetchall()
    idx = {(k, s, d): i for i, (k, s, d, _, _) in enumerate(ev)}
    lo = np.array([x[3] for x in ev]); hi = np.array([x[4] for x in ev])
    dec = e.con.execute("""SELECT q.level, q.w, d.ent, d.end_day, t.theta FROM decisions d JOIN queries q USING(qid)
                           JOIN thresholds t USING(qid,ent)""").fetchall()
    import scipy.sparse as sp
    r, c = [], []
    for j, (lvl, w, ent, end, th) in enumerate(dec):
        ks = [ent] if lvl == "stock" else [k for k in range(6) if k % 2 == ent]
        for k in ks:
            for s in range(2):
                for d in range(end - w + 1, end + 1): r.append(j); c.append(idx[(k, s, d)])
    M = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(dec), len(ev)))
    ds = DecisionSet(M, np.array([x[4] for x in dec]), lo, hi); st = ds.state()
    m = (st[ds.nz_d] == UNK) & ((hi - lo)[ds.nz_i] > 0)
    ref = np.bincount(ds.nz_i[m], weights=((hi - lo)[ds.nz_i] / (ds.U - ds.L)[ds.nz_d])[m], minlength=len(ev))
    for key, sc in sql.items():
        assert abs(ref[idx[key]] - sc) < 1e-9
    assert len(sql) == int((ref > 0).sum())


def test_gr_plan_is_guaranteed_under_adversarial_outcomes():
    e, truth, _ = make(7)
    budget = 12
    v, plan, cost = gr_plan(e, budget)
    assert cost <= budget
    planned = set(plan)
    und = e.con.execute("""SELECT d.qid, d.ent, d.end_day, q.w, t.theta, d.L, d.U FROM decisions d JOIN queries q USING(qid)
                           JOIN thresholds t USING(qid,ent) WHERE q.level='stock' AND d.state=-1""").fetchall()
    ev = {(k, s, d): (lo, hi) for k, s, d, lo, hi in e.con.execute("SELECT key,src,day,lo,hi FROM evidence").fetchall()}
    covered = [x for x in und if all((x[1], d) in planned or all(ev[(x[1], s, d)][0] == ev[(x[1], s, d)][1] for s in (0, 1))
                                     for d in range(x[2] - x[3] + 1, x[2] + 1))]
    assert len(covered) == v
    rng = np.random.default_rng(0)
    for _ in range(50):                               # any outcome of the planned probes resolves every covered decision
        for (qid, k, end, w, th, L, U) in covered:
            Lp, Up = 0.0, 0.0
            for d in range(end - w + 1, end + 1):
                for s in (0, 1):
                    lo, hi = ev[(k, s, d)]
                    x = int(rng.integers(lo, hi + 1)) if (k, d) in planned else None   # integer domain
                    Lp += x if x is not None else lo; Up += x if x is not None else hi
            assert Lp > th or Up <= th


def test_gr_plan_optimal_vs_bruteforce_small():
    e, truth, _ = make(8, K=2, T=10)
    slots = e.con.execute("SELECT key, day, COUNT(*) FROM evidence WHERE hi>lo GROUP BY 1,2").fetchall()
    und = e.con.execute("""SELECT d.ent, d.end_day, q.w FROM decisions d JOIN queries q USING(qid)
                           WHERE q.level='stock' AND d.state=-1""").fetchall()
    unc = {(k, d): c for k, d, c in slots}
    budget = 6
    best = 0
    keys = list(unc)
    for r in range(len(keys) + 1):
        for P in itertools.combinations(keys, r):
            if sum(unc[x] for x in P) > budget: continue
            Ps = set(P)
            best = max(best, sum(all((k, d) in Ps or (k, d) not in unc for d in range(end - w + 1, end + 1)) for k, end, w in und))
        if r > 12: break
    v, plan, cost = gr_plan(e, budget)
    assert v == best


def test_three_engine_modes_have_identical_observable_semantics():
    """inc (affected-only), inc_scan (full transition scan) and full (recompute bounds) must emit identical logs/states."""
    outs = []
    for mode in ("inc", "inc_scan", "full"):
        e, truth, _ = make(9, mode=mode); e.emit_initial()
        rng = np.random.default_rng(3)
        for step in range(8):
            b = list({(int(rng.integers(0, 6)), int(rng.integers(0, 2)), int(rng.integers(0, 30))) for _ in range(12)})
            rows = []
            for k, s, d in b:
                v = float(round(truth[k, s, d] * (0.3 if step == 5 else 1.0)))   # step 5: contradictory restatements
                rows.append((k, s, d, v, v))
            e.ingest(rows)
        q = lambda sql: e.con.execute(sql).fetchall()
        outs.append((q("SELECT qid,ent,end_day,state,emitted_state,emitted_at FROM decisions ORDER BY 1,2,3"),
                     q("SELECT * FROM decision_log ORDER BY 1,2,3,4"), q("SELECT * FROM retraction_log ORDER BY 1,2,3,4"),
                     q("SELECT * FROM contradiction_log ORDER BY 1,2,3,4")))
    assert outs[0] == outs[1] == outs[2]
    assert len(outs[0][2]) > 0
