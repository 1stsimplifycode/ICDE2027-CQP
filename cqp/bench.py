"""DuckDB benchmark on the real NSE+BSE panel: incremental certified maintenance vs full SQL recomputation.
Daily batches over N_DAYS days: on-time exact reports (40%), approximate reports (30%, exact 1-3 days later),
late exact reports (30%, 1-3 days later). 8 registered queries (W in {5,20} x mult in {1.0,1.5} x stock/industry).
Per batch: latency of ingest() (contradiction check + delta + state transitions + logs) vs recompute_full().
Usage: python cqp/bench.py [N_DAYS] -> results/cqp_bench.json
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from engine import CQP
from workload import load_real, Calib, CAL_DAYS

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MODE = sys.argv[2] if len(sys.argv) > 2 else "inc"
V, industry, keys, dates = load_real(); cal = Calib(V, industry)
K, S, T = V.shape
t0d = CAL_DAYS; t1 = min(T, t0d + N_DAYS); V = V[:, :, t0d:t1]; T2 = t1 - t0d
gnames = sorted(cal.groups); gid = np.zeros(K, int)
for j, g in enumerate(gnames): gid[cal.groups[g]] = j
rng = np.random.default_rng(11)
rows = [(k, s, d, 0.0, float(cal.cap[k, s])) for k in range(K) for s in range(S) for d in range(T2)]
e = CQP(mode=MODE)
t = time.perf_counter(); e.load([(k, int(gid[k])) for k in range(K)], rows); t_load = time.perf_counter() - t
gmed = np.array([cal.med[cal.groups[g]].sum() for g in gnames])
qid = 0; t = time.perf_counter()
for W in (5, 20):
    for c in (1.0, 1.5):
        qid += 1; e.register_query(qid, "stock", W, c, {k: c * W * cal.med[k] for k in range(K)})
        qid += 1; e.register_query(qid, "group", W, c, {g: c * W * gmed[g] for g in range(len(gnames))})
t_reg = time.perf_counter() - t
n_dec = e.con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
e.emit_initial()
import tracemalloc
pending = {}                                      # arrival day -> rows
for k in range(K):
    for s in range(S):
        for d in range(T2):
            v = float(V[k, s, d]); u = rng.random()
            if u < 0.4: pending.setdefault(d, []).append((k, s, d, v, v))
            elif u < 0.7:
                pending.setdefault(d, []).append((k, s, d, v / 1.1 * 1.0, v / 0.9 * 1.0))
                pending.setdefault(d + int(rng.integers(1, 4)), []).append((k, s, d, v, v))
            else: pending.setdefault(d + int(rng.integers(1, 4)), []).append((k, s, d, v, v))
lat_inc, lat_full, nrows = [], [], []
for day in range(T2 + 3):
    b = pending.get(day, [])
    if not b: continue
    t = time.perf_counter(); e.ingest(b); lat_inc.append(time.perf_counter() - t); nrows.append(len(b))
    if MODE == "inc" and day % 5 == 0:             # bounds-only full recomputation (sampled every 5th batch)
        t = time.perf_counter(); e.recompute_full(); lat_full.append(time.perf_counter() - t)
err = e.max_incremental_error()
pct = lambda a, q: float(np.percentile(a, q))
st = e.con.execute("SELECT state, COUNT(*) FROM decisions GROUP BY 1 ORDER BY 1").fetchall()
out = {"K": K, "days": T2, "queries": qid, "decisions": n_dec, "evidence_rows": len(rows), "load_s": t_load, "register_s": t_reg,
       "batches": len(lat_inc), "rows_per_batch_mean": float(np.mean(nrows)),
       "inc_p50_ms": 1e3 * pct(lat_inc, 50), "inc_p95_ms": 1e3 * pct(lat_inc, 95), "inc_p99_ms": 1e3 * pct(lat_inc, 99),
       "boundsonly_p50_ms": 1e3 * pct(lat_full, 50) if lat_full else None, "boundsonly_p95_ms": 1e3 * pct(lat_full, 95) if lat_full else None, "boundsonly_p99_ms": 1e3 * pct(lat_full, 99) if lat_full else None, "mode": MODE, "duckdb": __import__("duckdb").__version__,
       "inc_rows_per_s": float(np.sum(nrows) / np.sum(lat_inc)),
       "max_incremental_error": err, "final_states": st, "state_digest": e.con.execute("SELECT SUM(hash(qid,ent,end_day,state,emitted_state)) FROM decisions").fetchone()[0], "log_counts": e.con.execute("SELECT (SELECT COUNT(*) FROM decision_log),(SELECT COUNT(*) FROM retraction_log)").fetchone(),
       "retractions": e.con.execute("SELECT COUNT(*) FROM retraction_log").fetchone()[0],
       "contradictions": e.con.execute("SELECT COUNT(*) FROM contradiction_log").fetchone()[0]}
print(json.dumps(out, default=str))
out["raw_batch_ms"] = [1e3 * x for x in lat_inc]; out["total_wall_s"] = float(np.sum(lat_inc)); out["rep"] = os.environ.get("REP", "0")
out["exact"] = e.exact
json.dump(out, open(os.path.join(HERE, "..", "results", f"cqp_bench_{T2}d_{MODE}" + (f"_rep{os.environ['REP']}" if "REP" in os.environ else "") + ".json"), "w"), indent=1, default=str)
