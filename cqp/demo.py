"""Deterministic end-to-end demonstration (no external data). Run: python cqp/demo.py
Writes cqp/demo_out/: decisions.csv, decision_log.csv, retraction_log.csv, contradiction_log.csv,
acquisition_log.csv, gr_plan.json, verification.json, and demo_db.duckdb (SQL state snapshot)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from engine import CQP
from acquire_sql import acquire_seaq, gr_plan

OUT = os.path.join(HERE, "demo_out"); os.makedirs(OUT, exist_ok=True)
db = os.path.join(OUT, "demo_db.duckdb")
if os.path.exists(db): os.remove(db)
# 3 stocks in one industry, 2 sources (0=NSE, 1=BSE), 10 days; deterministic truth
truth = np.array([[[100, 110, 90, 400, 120, 100, 105, 95, 100, 98], [10, 11, 9, 40, 12, 10, 10, 9, 10, 9]],
                  [[50, 55, 60, 52, 48, 51, 49, 300, 50, 52], [5, 6, 6, 5, 5, 5, 5, 30, 5, 5]],
                  [[80, 82, 79, 81, 80, 78, 85, 83, 80, 81], [8, 8, 8, 8, 8, 8, 9, 8, 8, 8]]], float)
K, S, T = truth.shape
rows = []
for k in range(K):
    for s in range(S):
        for d in range(T):
            v = truth[k, s, d]
            if (k, s, d) in [(0, 0, 3), (1, 0, 7), (1, 1, 7), (2, 0, 9)]:      # missing: declared bound [0, 5*median]
                rows.append((k, s, d, 0.0, 5 * np.median(truth[k, s])))
            elif d >= 8:                                                          # recent: approximate +-10%
                r = v * 1.03; rows.append((k, s, d, r / 1.1, r / 0.9))
            else:
                rows.append((k, s, d, v, v))
e = CQP(db)
e.load([(k, 0) for k in range(K)], rows)
med = np.median(truth.sum(1), axis=1)
e.register_query(1, "stock", 5, 1.3, {k: 5 * 1.3 * med[k] for k in range(K)})          # 5-day volume spike alert
e.register_query(2, "group", 5, 1.3, {0: 5 * 1.3 * med.sum()})                          # industry roll-up
e.emit_initial()
log = []
def snap(tag):
    st = e.con.execute("SELECT qid, ent, end_day, ROUND(L,1), ROUND(U,1), state FROM decisions WHERE end_day>=7 ORDER BY 1,2,3").fetchall()
    log.append({"step": tag, "decisions_from_day7": st})
snap("initial")
unk = e.con.execute("SELECT qid, ent, end_day FROM decisions WHERE state=-1 ORDER BY 1,2,3").fetchall()
expl = {str(u): e.explain(*u) for u in unk[:3]}
# guaranteed-resolution plan with budget 3 item probes
v, plan, cost = gr_plan(e, 3)
oracle = lambda k, s, d: float(truth[k, s, d])
# acquisition by SEAQ (budget 6)
spent = acquire_seaq(e, oracle, budget=6, batch=2)
snap("after_seaq_acquisition")
# late contradictory restatement: stock 0 day 3 NSE volume restated from 400 to 100 (trade bust)
r = e.ingest([(0, 0, 3, 100.0, 100.0)])
snap("after_restatement")
# verification: incremental state equals full SQL recomputation; certified decisions agree with final truth
truth_final = truth.copy(); truth_final[0, 0, 3] = 100.0
wrong = 0; cert = 0
for qid, lvl, w in e.con.execute("SELECT qid, level, w FROM queries").fetchall():
    for ent, end, st, th in e.con.execute("SELECT ent,end_day,state,t.theta FROM decisions JOIN thresholds t USING(qid,ent) WHERE qid=?", [qid]).fetchall():
        ks = [ent] if lvl == "stock" else list(range(K))
        tv = truth_final[ks, :, end - w + 1:end + 1].sum() > th
        if st in (0, 1):
            cert += 1; wrong += int((st == 1) != tv)
ver = {"incremental_vs_full_max_abs_error": e.max_incremental_error(), "certified_decisions": cert,
       "certified_wrong_vs_final_truth": wrong, "seaq_probes": spent, "restatement_result": r,
       "gr_plan": {"guaranteed": v, "slots": plan, "cost": cost},
       "retractions": e.con.execute("SELECT COUNT(*) FROM retraction_log").fetchone()[0],
       "retractions_flagged_contradiction": e.con.execute("SELECT COUNT(*) FROM retraction_log WHERE caused_by_contradiction").fetchone()[0]}
for t in ["decisions", "decision_log", "retraction_log", "contradiction_log", "acquisition_log"]:
    e.con.execute(f"COPY {t} TO '{os.path.join(OUT, t + '.csv').replace(chr(92), '/')}' (HEADER)")
json.dump({"steps": log, "explanations_for_insufficient": expl}, open(os.path.join(OUT, "trace.json"), "w"), indent=1, default=str)
json.dump(ver, open(os.path.join(OUT, "verification.json"), "w"), indent=1, default=str)
print(json.dumps(ver, default=str))
assert ver["incremental_vs_full_max_abs_error"] < 1e-6 and wrong == 0 and ver["retractions_flagged_contradiction"] == ver["retractions"]
print("DEMO OK")
