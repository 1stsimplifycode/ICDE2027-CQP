"""CQP: a DuckDB-backed certified decision-query processor.

State lives in SQL tables; every operation is SQL executed by DuckDB:
  evidence(key, src, day, lo, hi, ver)                interval evidence per item (declared bound if unreported)
  keys(key, grp)                                      key -> roll-up group
  queries(qid, level, w, mult)                        registered threshold queries (level 'stock' | 'group')
  thresholds(qid, ent, theta)                         per-entity thresholds
  decisions(qid, ent, end_day, L, U, state, emitted_state, emitted_at)
  decision_log, retraction_log, contradiction_log, acquisition_log
Incremental maintenance: a batch of evidence changes becomes a delta table; one UPDATE ... FROM
applies (dlo, dhi) to every decision window containing the changed slot (O(deg) rows per item).
Baseline: full recomputation with SQL window functions (recompute_full).
"""
import duckdb
import numpy as np

STATE_SQL = "CASE WHEN d.L > t.theta THEN 1 WHEN d.U <= t.theta THEN 0 ELSE -1 END"


class CQP:
    def __init__(self, path=":memory:", mode="inc", exact=True):
        """mode: 'inc' (delta + transitions on affected decisions only), 'inc_scan' (delta + transitions over all
        decisions; the v5 implementation), 'full' (same semantics, but bounds recomputed from scratch each batch)."""
        assert mode in ("inc", "inc_scan", "full")
        self.mode = mode
        # exact=True: integer-valued domain (traded volumes). Evidence is rounded OUTWARD-SAFE to the integer hull
        # [ceil(lo), floor(hi)] (sound because true values are integers), thresholds to floor(theta) (exact for integer
        # sums: L > theta <=> L > floor(theta); U <= theta <=> U <= floor(theta)). All bound arithmetic is then exact BIGINT.
        self.exact = exact
        N = "BIGINT" if exact else "DOUBLE"
        self.con = duckdb.connect(path)
        self.ts = 0
        c = self.con
        c.execute("CREATE TABLE evidence(key INT, src INT, day INT, lo {N}, hi {N}, ver INT, PRIMARY KEY(key,src,day))".format(N=N))
        c.execute("CREATE TABLE keys(key INT PRIMARY KEY, grp INT)")
        c.execute("CREATE TABLE queries(qid INT PRIMARY KEY, level VARCHAR, w INT, mult DOUBLE, lv INT)")
        c.execute("CREATE TABLE thresholds(qid INT, ent INT, theta {N}, PRIMARY KEY(qid,ent))".format(N=N))
        c.execute(("CREATE TABLE decisions(qid INT, ent INT, end_day INT, L {N}, U {N}, state INT, "
                   "emitted_state INT, emitted_at INT, PRIMARY KEY(qid,ent,end_day))").format(N=N))
        c.execute("CREATE TABLE decision_log(ts INT, qid INT, ent INT, end_day INT, old_state INT, new_state INT)")
        c.execute("CREATE TABLE retraction_log(ts INT, qid INT, ent INT, end_day INT, emitted_state INT, new_state INT, "
                  "caused_by_contradiction BOOLEAN)")
        c.execute("CREATE TABLE contradiction_log(ts INT, key INT, src INT, day INT, old_lo {N}, old_hi {N}, new_lo {N}, new_hi {N})".format(N=N))
        c.execute("CREATE TABLE acquisition_log(ts INT, key INT, src INT, day INT, policy VARCHAR, score DOUBLE)")

    # ------------------------------------------------------------------ setup
    def load(self, keys_grp, evidence_rows):
        """keys_grp: list of (key, grp); evidence_rows: list/array of (key, src, day, lo, hi)."""
        self.con.executemany("INSERT INTO keys VALUES (?,?)", [(int(k), int(g)) for k, g in keys_grp])
        import pandas as pd
        df = self._frame(evidence_rows); df["ver"] = 0
        self.con.execute("SELECT 1").fetchall(); self.con.register("ev_in", df); self.con.execute("INSERT INTO evidence SELECT key,src,day,lo,hi,ver FROM ev_in")
        self.con.execute("SELECT 1").fetchall(); self.con.unregister("ev_in")
        self.T = int(df.day.max()) + 1

    def _frame(self, rows):
        import pandas as pd
        df = pd.DataFrame(np.asarray(rows, float), columns=["key", "src", "day", "lo", "hi"])
        df[["key", "src", "day"]] = df[["key", "src", "day"]].astype(np.int64)
        if self.exact:
            df["lo"] = np.ceil(df["lo"].values).astype(np.int64); df["hi"] = np.floor(df["hi"].values).astype(np.int64)
            if (df["lo"] > df["hi"]).any():
                raise ValueError("evidence interval contains no integer: violates the integer-valued domain assumption")
        return df

    def _theta(self, th):
        import math
        return int(math.floor(th)) if self.exact else float(th)

    def register_query(self, qid, level, w, mult, theta_by_ent):
        c = self.con
        c.execute("INSERT INTO queries VALUES (?,?,?,?,?)", [qid, level, w, mult, 0 if level == "stock" else 1]).fetchall()
        c.executemany("INSERT INTO thresholds VALUES (?,?,?)", [(qid, int(e), self._theta(th)) for e, th in theta_by_ent.items()])
        c.execute(f"INSERT INTO decisions SELECT qid, ent, end_day, L, U, -1, NULL, NULL FROM ({self._full_sql(qid)})")
        c.execute(f"""UPDATE decisions d SET state = {STATE_SQL} FROM thresholds t
                      WHERE d.qid=t.qid AND d.ent=t.ent AND d.qid={qid}""")

    def _full_sql(self, qid):
        level, w = self.con.execute("SELECT level, w FROM queries WHERE qid=?", [qid]).fetchone()
        ent = "s.key" if level == "stock" else "k.grp"
        return f"""
          WITH slot AS (SELECT {ent} AS ent, s.day, SUM(s.lo) lo, SUM(s.hi) hi
                        FROM evidence s JOIN keys k ON s.key=k.key GROUP BY 1,2)
          SELECT {qid} AS qid, ent, day AS end_day,
                 SUM(lo) OVER (PARTITION BY ent ORDER BY day ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS L,
                 SUM(hi) OVER (PARTITION BY ent ORDER BY day ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS U,
                 COUNT(*) OVER (PARTITION BY ent ORDER BY day ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS n
          FROM slot QUALIFY n = {w}"""

    # ------------------------------------------------------------------ ingestion
    def ingest(self, rows):
        """rows: (key, src, day, lo, hi). Detects contradictions, applies deltas incrementally,
        logs state changes, emissions and retractions. Returns dict of counts."""
        import pandas as pd
        self.ts += 1; c = self.con
        df = self._frame(rows)
        c.execute("SELECT 1").fetchall(); c.register("batch", df)
        c.execute(f"""INSERT INTO contradiction_log SELECT {self.ts}, e.key, e.src, e.day, e.lo, e.hi, b.lo, b.hi
                      FROM batch b JOIN evidence e USING(key,src,day) WHERE b.hi < e.lo OR b.lo > e.hi""")
        c.execute("""CREATE OR REPLACE TEMP TABLE delta AS
                     SELECT b.key, b.day, SUM(b.lo - e.lo) dlo, SUM(b.hi - e.hi) dhi
                     FROM batch b JOIN evidence e USING(key,src,day) GROUP BY 1,2""")
        c.execute("""UPDATE evidence e SET lo=b.lo, hi=b.hi, ver=e.ver+1 FROM batch b
                     WHERE e.key=b.key AND e.src=b.src AND e.day=b.day""")
        c.execute("SELECT 1").fetchall(); c.unregister("batch")
        if self.mode == "full":
            self._recompute_into_decisions()
        else:
            self._apply_delta()
        return self._transitions()

    def _recompute_into_decisions(self):
        parts = " UNION ALL ".join(f"SELECT qid, ent, end_day, L, U FROM ({self._full_sql(q)})"
                                   for (q,) in self.con.execute("SELECT qid FROM queries ORDER BY qid").fetchall())
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE fullb AS {parts}").fetchall()
        self.con.execute("""UPDATE decisions d SET L=f.L, U=f.U FROM fullb f
                            WHERE d.qid=f.qid AND d.ent=f.ent AND d.end_day=f.end_day""").fetchall()

    def _apply_delta(self):
        # stock-level and group-level windows touched by each changed slot
        self.con.execute("""
          CREATE OR REPLACE TEMP TABLE upd AS
            SELECT q.qid, CASE WHEN q.lv=0 THEN dl.key ELSE k.grp END AS ent, de.end_day,
                   SUM(dl.dlo) dlo, SUM(dl.dhi) dhi
            FROM delta dl JOIN keys k ON dl.key=k.key CROSS JOIN queries q
            JOIN decisions de ON de.qid=q.qid AND de.ent = (CASE WHEN q.lv=0 THEN dl.key ELSE k.grp END)
                             AND de.end_day BETWEEN dl.day AND dl.day + q.w - 1
            GROUP BY 1,2,3""").fetchall()
        self.con.execute("""UPDATE decisions d SET L = d.L + x.dlo, U = d.U + x.dhi FROM upd x
          WHERE d.qid=x.qid AND d.ent=x.ent AND d.end_day=x.end_day""").fetchall()

    def _transitions(self):
        c = self.con; ts = self.ts
        scope = "JOIN upd u ON d.qid=u.qid AND d.ent=u.ent AND d.end_day=u.end_day" if self.mode == "inc" else ""
        c.execute(f"""CREATE OR REPLACE TEMP TABLE newst AS SELECT d.qid,d.ent,d.end_day,d.state old_state,{STATE_SQL} new_state,
                      d.emitted_state FROM decisions d {scope} JOIN thresholds t ON d.qid=t.qid AND d.ent=t.ent""").fetchall()
        if self.mode != "inc":   # only rows whose state changes need writing
            c.execute("DELETE FROM newst WHERE old_state = new_state AND ((new_state IN (0,1) AND emitted_state = new_state) OR (new_state = -1 AND emitted_state IS NULL))").fetchall()
        c.execute(f"INSERT INTO decision_log SELECT {ts},qid,ent,end_day,old_state,new_state FROM newst WHERE old_state<>new_state")
        c.execute(f"""INSERT INTO retraction_log
            SELECT {ts}, n.qid, n.ent, n.end_day, n.emitted_state, n.new_state,
                   EXISTS (SELECT 1 FROM contradiction_log cl JOIN keys k ON cl.key=k.key JOIN queries q ON q.qid=n.qid
                           WHERE cl.ts={ts} AND (CASE WHEN q.lv=0 THEN cl.key ELSE k.grp END)=n.ent
                             AND n.end_day BETWEEN cl.day AND cl.day+q.w-1)
            FROM newst n WHERE n.emitted_state IS NOT NULL AND n.new_state <> n.emitted_state""")
        c.execute("""UPDATE decisions d SET state=n.new_state,
                       emitted_state = CASE WHEN n.new_state IN (0,1) THEN n.new_state ELSE NULL END,
                       emitted_at = CASE WHEN n.new_state IN (0,1) AND (d.emitted_state IS NULL OR d.emitted_state<>n.new_state)
                                         THEN %d ELSE d.emitted_at END
                     FROM newst n WHERE d.qid=n.qid AND d.ent=n.ent AND d.end_day=n.end_day""" % ts)
        r = c.execute(f"""SELECT (SELECT COUNT(*) FROM decision_log WHERE ts={ts}),
                                 (SELECT COUNT(*) FROM retraction_log WHERE ts={ts}),
                                 (SELECT COUNT(*) FROM contradiction_log WHERE ts={ts})""").fetchone()
        return {"transitions": r[0], "retractions": r[1], "contradictions": r[2]}

    def emit_initial(self):
        self.con.execute(f"UPDATE decisions SET emitted_state=state, emitted_at=0 WHERE state IN (0,1)")

    # ------------------------------------------------------------------ baselines / checks
    def recompute_full(self):
        """Conventional SQL baseline: recompute every decision from scratch with window functions."""
        out = []
        for (qid,) in self.con.execute("SELECT qid FROM queries").fetchall():
            out.append(self.con.execute(f"SELECT qid, ent, end_day, L, U FROM ({self._full_sql(qid)})").fetchnumpy())
        return out

    def max_incremental_error(self):
        err = 0.0
        for (qid,) in self.con.execute("SELECT qid FROM queries").fetchall():
            r = self.con.execute(f"""SELECT MAX(ABS(d.L-f.L)) , MAX(ABS(d.U-f.U)) FROM decisions d
                    JOIN ({self._full_sql(qid)}) f ON d.qid=f.qid AND d.ent=f.ent AND d.end_day=f.end_day""").fetchone()
            err = max(err, r[0] or 0, r[1] or 0)
        return err

    # ------------------------------------------------------------------ explanation
    def explain(self, qid, ent, end_day, top=3):
        """Why is a decision insufficient? Gap to the threshold and the widest contributing items."""
        c = self.con
        L, U, st, th, level, w = c.execute("""SELECT d.L,d.U,d.state,t.theta,q.level,q.w FROM decisions d
              JOIN thresholds t USING(qid,ent) JOIN queries q USING(qid) WHERE qid=? AND ent=? AND end_day=?""",
                                           [qid, ent, end_day]).fetchone()
        cond = "e.key=?" if level == "stock" else "k.grp=?"
        items = c.execute(f"""SELECT e.key,e.src,e.day,e.lo,e.hi,e.hi-e.lo w FROM evidence e JOIN keys k ON e.key=k.key
              WHERE {cond} AND e.day BETWEEN ? AND ? AND e.hi>e.lo ORDER BY w DESC LIMIT {top}""",
                          [ent, end_day - w + 1, end_day]).fetchall()
        return {"state": st, "L": L, "U": U, "theta": th, "gap_to_true": th - L, "gap_to_false": U - th, "widest_items": items}
