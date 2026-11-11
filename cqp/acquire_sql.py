"""Acquisition on top of the CQP engine.
seaq_scores_sql: SEAQ score  sum_{undetermined d containing i} w_i / (U_d - L_d)  computed as one SQL query.
gr_plan: exact guaranteed-resolution plan for STOCK-level windows (Theorem 2), using the value-indexed DP
          from src/wcdp_sp.py on uncertain slots extracted with SQL; returns slots to probe.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from wcdp_sp import key_dp_value


SEAQ_SQL = """
WITH und AS (SELECT d.qid, d.ent, d.end_day, d.U - d.L AS wd, q.lv, q.w FROM decisions d JOIN queries q USING(qid)
             WHERE d.state = -1 AND d.U > d.L)
SELECT e.key, e.src, e.day, SUM((e.hi - e.lo) / u.wd) AS score
FROM evidence e JOIN keys k ON e.key = k.key
JOIN und u ON u.ent = (CASE WHEN u.lv=0 THEN e.key ELSE k.grp END)
          AND e.day BETWEEN u.end_day - u.w + 1 AND u.end_day
WHERE e.hi > e.lo
GROUP BY 1,2,3 ORDER BY score DESC, e.key, e.src, e.day LIMIT {limit}"""


def seaq_scores_sql(engine, limit=8):
    return engine.con.execute(SEAQ_SQL.format(limit=limit)).fetchall()


def acquire_seaq(engine, oracle, budget, batch=8):
    """Probe items in SEAQ order; oracle(key,src,day) -> exact value. Logs to acquisition_log."""
    spent = 0
    while spent < budget:
        top = seaq_scores_sql(engine, min(batch, budget - spent))
        if not top:
            break
        rows = []
        for (k, s, d, sc) in top:
            v = oracle(k, s, d); rows.append((k, s, d, v, v))
            engine.con.execute("INSERT INTO acquisition_log VALUES (?,?,?,?,?,?)", [engine.ts + 1, k, s, d, "seaq", sc])
        engine.ingest(rows); spent += len(rows)
    return spent


def _key_plan(c, ranges, v_target):
    """Reconstruct a min-cost plan guaranteeing >= v_target decisions for one key (value-indexed DP with back-pointers)."""
    m = len(c); nd = len(ranges)
    cnt = np.zeros((m + 2, m + 2), int)
    for p, q in ranges: cnt[p, q] += 1
    G = np.zeros((m + 2, m + 2), int)
    for i in range(m + 1):
        acc = np.cumsum(cnt[i + 1:, :].sum(0))
        for j in range(i + 1, m + 2): G[i, j] = acc[j - 1]
    pc = np.concatenate([[0.0], np.cumsum(np.asarray(c, float))])
    best = np.full((m + 2, nd + 1), np.inf); best[0, 0] = 0; bp = {}
    for j in range(1, m + 2):
        for i in range(j):
            cost = pc[j - 1] - pc[i]; g = G[i, j]
            for v in range(g, nd + 1):
                val = best[i, v - g] + cost
                if val < best[j, v]: best[j, v] = val; bp[(j, v)] = (i, v - g)
    v = min(range(v_target, nd + 1), key=lambda x: best[m + 1, x])
    j = m + 1; unprobed = []
    while j != 0:
        i, v = bp[(j, v)]; unprobed.append(i); j = i
    return set(range(1, m + 1)) - set(unprobed)


def gr_plan(engine, budget):
    """Exact GR plan over stock-level undetermined decisions. Returns (guaranteed_count, list of (key,day) slots, cost)."""
    con = engine.con
    slots = con.execute("SELECT key, day, COUNT(*) c FROM evidence WHERE hi>lo GROUP BY 1,2 ORDER BY 1,2").fetchall()
    decs = con.execute("""SELECT d.ent, d.end_day - q.w + 1, d.end_day FROM decisions d JOIN queries q USING(qid)
                          WHERE q.lv=0 AND d.state=-1""").fetchall()
    from collections import defaultdict
    S = defaultdict(list); D = defaultdict(list)
    for k, d, c in slots: S[k].append((d, c))
    for k, a, b in decs: D[k].append((a, b))
    per = []
    for k in sorted(D):
        days = [d for d, _ in S[k]]; cost = [c for _, c in S[k]]; pos = {d: i + 1 for i, d in enumerate(days)}
        R = []
        for a, b in D[k]:
            ins = [d for d in days if a <= d <= b]
            if ins: R.append((pos[ins[0]], pos[ins[-1]]))
        per.append((k, days, cost, R, key_dp_value(cost, R)))
    # min-plus combine with back-tracking
    tots = [np.array([0.0])]
    for (_, _, _, _, g) in per:
        prev = tots[-1]; new = np.full(len(prev) + len(g) - 1, np.inf)
        for v, cv in enumerate(g):
            if np.isfinite(cv): new[v:v + len(prev)] = np.minimum(new[v:v + len(prev)], prev + cv)
        tots.append(new)
    ok = np.flatnonzero(tots[-1] <= budget + 1e-9); vstar = int(ok.max()) if len(ok) else 0
    rem = vstar; plan = []; cost_total = 0.0
    for idx in range(len(per) - 1, -1, -1):
        k, days, cost, R, g = per[idx]; prev = tots[idx]
        for v in range(min(rem, len(g) - 1), -1, -1):
            if np.isfinite(g[v]) and rem - v < len(prev) and abs(prev[rem - v] + g[v] - tots[idx + 1][rem]) < 1e-9:
                if v > 0:
                    for x in _key_plan(cost, R, v): plan.append((k, days[x - 1])); cost_total += cost[x - 1]
                rem -= v; break
    return vstar, sorted(plan), cost_total
