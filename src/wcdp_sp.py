"""Strongly polynomial exact GR for interval workloads (value-indexed DP) + exact MILP for arbitrary sharing.

key_dp_value(c, ranges): c = arbitrary nonnegative real costs of the uncertain slots of one key,
ranges = (p,q) index ranges of its undetermined decisions. Returns g[v] = minimum cost to GUARANTEE
at least v decisions (v = 0..|ranges|), +inf if impossible. Same gap decomposition as wcdp.key_dp, but
indexed by value (bounded by #decisions) instead of budget, so the running time does not depend on the
numeric magnitude of costs or budget:  O(m^2 * |D_k|) additions/comparisons per key.
combine_value(gs): min-plus convolution over keys, O(|D|^2) total.
answer(B) = max{v : G[v] <= B}.

milp_gr(...): exact 0-1 ILP for arbitrary incidence (e.g. industry roll-ups), solved by HiGHS (scipy.milp).
"""
import itertools
import numpy as np


def key_dp_value(c, ranges):
    m = len(c); nd = len(ranges)
    cnt = np.zeros((m + 2, m + 2), int)
    for p, q in ranges:
        cnt[p, q] += 1
    G = np.zeros((m + 2, m + 2), int)
    for i in range(m + 1):
        acc = np.cumsum(cnt[i + 1:, :].sum(axis=0))
        for j in range(i + 1, m + 2):
            G[i, j] = acc[j - 1]
    pc = np.concatenate([[0.0], np.cumsum(np.asarray(c, float))])
    INF = np.inf
    best = np.full((m + 2, nd + 1), INF); best[0, 0] = 0.0      # best[j][v] = min cost, last unprobed = j, value v
    for j in range(1, m + 2):
        for i in range(j):
            if not np.isfinite(best[i]).any():
                continue
            cost = pc[j - 1] - pc[i]; g = G[i, j]
            cand = np.full(nd + 1, INF)
            cand[g:] = best[i, :nd + 1 - g] + cost
            best[j] = np.minimum(best[j], cand)
    out = best[m + 1].copy()
    for v in range(nd - 1, -1, -1):                            # at least v
        out[v] = min(out[v], out[v + 1])
    return out


def combine_value(gs):
    tot = np.array([0.0])
    for g in gs:
        new = np.full(len(tot) + len(g) - 1, np.inf)
        for v, cv in enumerate(g):
            if np.isfinite(cv):
                new[v:v + len(tot)] = np.minimum(new[v:v + len(tot)], tot + cv)
        tot = new
    for v in range(len(tot) - 2, -1, -1):
        tot[v] = min(tot[v], tot[v + 1])
    return tot


def answer(tot, B):
    ok = np.flatnonzero(tot <= B + 1e-9)
    return int(ok.max()) if len(ok) else 0


def brute_real(c, ranges, B):
    m = len(c); best = 0
    for r in range(m + 1):
        for P in itertools.combinations(range(1, m + 1), r):
            if sum(c[x - 1] for x in P) > B + 1e-12: continue
            Ps = set(P)
            best = max(best, sum(all(x in Ps for x in range(p, q + 1)) for p, q in ranges))
    return best


def milp_gr(slot_cost, dec_slots, B, time_limit=60):
    """Exact GR for arbitrary incidence. slot_cost: (n,) costs; dec_slots: list of arrays of slot ids
    (uncertain slots of each undetermined decision). max sum y_d s.t. y_d <= x_s (s in d), c.x <= B."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp
    n = len(slot_cost); D = len(dec_slots)
    rows, cols, vals = [], [], []; r = 0
    for d, S in enumerate(dec_slots):
        for s in S:
            rows += [r, r]; cols += [n + d, s]; vals += [1.0, -1.0]; r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n + D))
    Bud = sp.csr_matrix(np.concatenate([slot_cost, np.zeros(D)])[None, :])
    cons = [LinearConstraint(A, -np.inf, 0.0), LinearConstraint(Bud, -np.inf, B)]
    obj = np.concatenate([np.zeros(n), -np.ones(D)])
    res = milp(obj, constraints=cons, integrality=np.ones(n + D), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    val = -res.fun if res.x is not None else float("nan")
    return val, res.status, getattr(res, "mip_gap", None)


if __name__ == "__main__":
    from wcdp import key_dp, brute
    rng = np.random.default_rng(1); bad = bad_int = 0; n = 0
    for t in range(2000):
        m = int(rng.integers(1, 9)); c = rng.uniform(0.01, 5.0, m).round(3).tolist()   # real costs
        R = []
        for _ in range(int(rng.integers(0, 10))):
            p = int(rng.integers(1, m + 1)); q = int(rng.integers(p, m + 1)); R.append((p, q))
        B = float(rng.uniform(0, sum(c)))
        g = key_dp_value(c, R)
        bad += int(answer(g, B) != brute_real(c, R, B)); n += 1
        ci = rng.integers(1, 4, m).tolist()                                           # integer: equals budget DP
        Bi = int(sum(ci)); fb = key_dp(ci, R, Bi); gv = key_dp_value(ci, R)
        bad_int += int(any(answer(gv, b) != fb[b] for b in range(Bi + 1)))
    # multi-key combine vs brute force over two keys
    badc = 0
    for t in range(300):
        cs = [rng.uniform(0.1, 3, int(rng.integers(1, 5))).tolist() for _ in range(2)]
        Rs = []
        for c in cs:
            m = len(c); Rs.append([(p, int(rng.integers(p, m + 1))) for p in rng.integers(1, m + 1, int(rng.integers(1, 5)))])
        B = float(rng.uniform(0, sum(map(sum, cs))))
        tot = combine_value([key_dp_value(c, R) for c, R in zip(cs, Rs)])
        bf = 0
        for r1 in range(len(cs[0]) + 1):
            for P1 in itertools.combinations(range(1, len(cs[0]) + 1), r1):
                c1 = sum(cs[0][x - 1] for x in P1)
                if c1 > B: continue
                v1 = sum(all(x in P1 for x in range(p, q + 1)) for p, q in Rs[0])
                bf = max(bf, v1 + brute_real(cs[1], Rs[1], B - c1))
        badc += int(answer(tot, B) != bf)
    # MILP vs brute force on small arbitrary-sharing instances
    badm = 0
    for t in range(200):
        nS = int(rng.integers(2, 8)); cost = rng.uniform(0.5, 3, nS)
        decs = [rng.choice(nS, int(rng.integers(1, min(4, nS) + 1)), replace=False) for _ in range(int(rng.integers(1, 8)))]
        B = float(rng.uniform(0, cost.sum()))
        v, st, _ = milp_gr(cost, decs, B)
        bf = 0
        for r in range(nS + 1):
            for P in itertools.combinations(range(nS), r):
                if cost[list(P)].sum() > B + 1e-9: continue
                bf = max(bf, sum(set(d.tolist()) <= set(P) for d in decs))
        badm += int(round(v) != bf)
    out = {"value_dp_real_cost_instances": n, "value_dp_mismatches": bad, "value_vs_budget_dp_mismatches": bad_int,
           "combine_mismatches": badc, "milp_mismatches": badm}
    print(out)
    import json; json.dump(out, open(r"C:\Users\mailt\Downloads\ICDE-2027\results\dp_verification.json", "w"))
