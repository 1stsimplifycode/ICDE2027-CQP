"""Experiment E: guaranteed-resolution planning WITH industry roll-ups (arbitrary-sharing class) on the real panel.
Exact reference: 0-1 ILP (HiGHS) with group-day 'clean' variables z_{g,t} <= x_{k,t}.
Compared plans (all evaluated on the same full objective: stock + industry decisions guaranteed):
  milp          exact (time limit; status and gap recorded)
  dp_stock      exact window-DP plan for stock decisions only (ignores roll-ups), then evaluated on full objective
  greedy        cheapest-remaining-completion over all decisions
  seaq, widest  static day orders
Also times the budget-indexed vs value-indexed DP with real-valued slot costs.
Usage: python run_rollup.py -> results/rollup.json
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, r"C:\Users\mailt\Downloads\ICDE-2027\src")
from workload import load_real, Calib, evidence, CAL_DAYS, WINDOWS, MULTS
from wcdp import key_dp
from wcdp_sp import key_dp_value, combine_value, answer

H = 40; FRACS = [0.05, 0.20]; TL = 60
V, industry, keys, dates = load_real(); cal = Calib(V, industry)
K, S, T = V.shape
gnames = sorted(cal.groups); gid = np.zeros(K, int)
for j, g in enumerate(gnames): gid[cal.groups[g]] = j
G = len(gnames)
days = np.sort(np.random.default_rng(123).choice(np.arange(CAL_DAYS + H, T), 3, replace=False))
out = []
for e, t in enumerate(days):
    rng = np.random.default_rng(7000 + e)
    truth = V[:, :, t - H + 1:t + 1].reshape(-1); lag = np.tile(np.arange(H)[::-1], K * S)
    cap = np.repeat(cal.cap[:, :, None], H, 2).reshape(-1); smed = np.repeat(cal.src_med[:, :, None], H, 2).reshape(-1)
    lo, hi, est, kind = evidence(truth, cap, smed, lag, "late", rng)
    lo = lo.reshape(K, S, H); hi = hi.reshape(K, S, H)
    dcost = ((hi - lo) > 0).sum(1)                        # (K,H)
    Ld, Ud = lo.sum(1), hi.sum(1)
    GL = np.zeros((G, H)); GU = np.zeros((G, H)); np.add.at(GL, gid, Ld); np.add.at(GU, gid, Ud)
    gmed = np.array([cal.med[cal.groups[g]].sum() for g in gnames])
    sdec, gdec = [], []
    for W in WINDOWS:
        for c in MULTS:
            for b in range(W - 1, H):
                a = b - W + 1
                L = Ld[:, a:b + 1].sum(1); U = Ud[:, a:b + 1].sum(1); th = c * W * cal.med
                for k in np.flatnonzero((L <= th) & (U > th)): sdec.append((k, a, b))
                L = GL[:, a:b + 1].sum(1); U = GU[:, a:b + 1].sum(1); th = c * W * gmed
                for g in np.flatnonzero((L <= th) & (U > th)): gdec.append((g, a, b))
    sdec, gdec = np.array(sdec), np.array(gdec)
    total = int(dcost.sum()); Bs = [int(f * total) for f in FRACS]

    def value(probed):
        un = ((dcost > 0) & ~probed).astype(int)
        pre = np.concatenate([np.zeros((K, 1), int), np.cumsum(un, 1)], 1)
        sv = int(((pre[sdec[:, 0], sdec[:, 2] + 1] - pre[sdec[:, 0], sdec[:, 1]]) == 0).sum())
        gu = np.zeros((G, H), int); np.add.at(gu, gid, un)
        gp = np.concatenate([np.zeros((G, 1), int), np.cumsum(gu, 1)], 1)
        gv = int(((gp[gdec[:, 0], gdec[:, 2] + 1] - gp[gdec[:, 0], gdec[:, 1]]) == 0).sum()) if len(gdec) else 0
        return sv, gv

    res = {"day": int(t), "n_stock": int(len(sdec)), "n_ind": int(len(gdec)), "total_cost": total, "B": Bs}
    # ---------------- MILP exact ----------------
    from scipy.optimize import milp, LinearConstraint, Bounds
    slots = np.argwhere(dcost > 0); sidx = -np.ones((K, H), int); sidx[slots[:, 0], slots[:, 1]] = np.arange(len(slots))
    nX = len(slots); gday = [(g, d) for g in range(G) for d in range(H)]; zidx = {x: nX + i for i, x in enumerate(gday)}
    nZ = len(gday); nY = len(sdec) + len(gdec); N = nX + nZ + nY
    rows, cols, vals = [], [], []; r = 0
    for (g, d), zi in zidx.items():                                   # z_{g,d} <= x_{k,d}
        for k in cal.groups[gnames[g]]:
            if sidx[k, d] >= 0:
                rows += [r, r]; cols += [zi, sidx[k, d]]; vals += [1, -1]; r += 1
    for j, (k, a, b) in enumerate(sdec):                              # y <= x
        for d in range(a, b + 1):
            if sidx[k, d] >= 0:
                rows += [r, r]; cols += [nX + nZ + j, sidx[k, d]]; vals += [1, -1]; r += 1
    for j, (g, a, b) in enumerate(gdec):                              # y <= z
        for d in range(a, b + 1):
            rows += [r, r]; cols += [nX + nZ + len(sdec) + j, zidx[(g, d)]]; vals += [1, -1]; r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, N))
    cvec = np.zeros(N); cvec[:nX] = dcost[slots[:, 0], slots[:, 1]]
    obj = np.zeros(N); obj[nX + nZ:] = -1
    res["milp"], res["milp_status"], res["milp_secs"], res["milp_gap"] = [], [], [], []
    res["dp_stock"], res["greedy"], res["seaq"], res["widest"] = [], [], [], []
    for B in Bs:
        t0 = time.perf_counter()
        sol = milp(obj, constraints=[LinearConstraint(A, -np.inf, 0), LinearConstraint(sp.csr_matrix(cvec[None, :]), -np.inf, B)],
                   integrality=np.ones(N), bounds=Bounds(0, 1), options={"time_limit": TL, "disp": False})
        res["milp_secs"].append(time.perf_counter() - t0); res["milp_status"].append(int(sol.status))
        res["milp_gap"].append(float(getattr(sol, "mip_gap", np.nan) or 0.0))
        if sol.x is not None:
            pr = np.zeros((K, H), bool); xs = sol.x[:nX] > 0.5; pr[slots[xs, 0], slots[xs, 1]] = True
            res["milp"].append(value(pr))
        else:
            res["milp"].append(None)
    # ---------------- stock-only exact DP plan (with reconstruction by greedy on per-key optimal value) ----
    # plan reconstruction: per key choose budget split via value-DP, then per key brute-force-free reconstruction:
    # we reconstruct by re-running the budget DP with back-pointers.
    def key_plan(c, ranges, b):
        m = len(c); cnt = np.zeros((m + 2, m + 2), int)
        for p, q in ranges: cnt[p, q] += 1
        Gm = np.zeros((m + 2, m + 2), int)
        for i in range(m + 1):
            acc = np.cumsum(cnt[i + 1:, :].sum(0))
            for j in range(i + 1, m + 2): Gm[i, j] = acc[j - 1]
        pc = np.concatenate([[0], np.cumsum(c)]); NEG = -10**9
        best = np.full((m + 2, b + 1), NEG); best[0, 0] = 0; bp = {}
        for j in range(1, m + 2):
            for i in range(j):
                cost = pc[j - 1] - pc[i]
                if cost > b: continue
                for bb in range(cost, b + 1):
                    v = best[i, bb - cost] + Gm[i, j]
                    if v > best[j, bb]: best[j, bb] = v; bp[(j, bb)] = (i, bb - cost)
        bb = int(np.argmax(best[m + 1])); j = m + 1; unprobed = []
        while j != 0:
            i, b2 = bp[(j, bb)]; unprobed.append(i); j, bb = i, b2
        return set(range(1, m + 1)) - set(unprobed)
    t0 = time.perf_counter()
    per = []
    for k in range(K):
        ud = np.flatnonzero(dcost[k] > 0); pos = {d: i + 1 for i, d in enumerate(ud)}; R = []
        for (_, a, b) in sdec[sdec[:, 0] == k]:
            ins = ud[(ud >= a) & (ud <= b)]; R.append((pos[ins[0]], pos[ins[-1]]))
        per.append((ud, dcost[k, ud].astype(int).tolist(), R))
    gs = [key_dp_value(c, R) for (_, c, R) in per]
    for B in Bs:
        # optimal split across keys from value DP: greedy back-tracking over combine
        tot = [np.array([0.0])]
        for g in gs:
            prev = tot[-1]; new = np.full(len(prev) + len(g) - 1, np.inf)
            for v, cv in enumerate(g):
                if np.isfinite(cv): new[v:v + len(prev)] = np.minimum(new[v:v + len(prev)], prev + cv)
            tot.append(new)
        vstar = int(np.flatnonzero(tot[-1] <= B).max()); probed = np.zeros((K, H), bool); rem = vstar
        for k in range(K - 1, -1, -1):
            g = gs[k]; prev = tot[k]
            for v in range(min(rem, len(g) - 1), -1, -1):
                if np.isfinite(g[v]) and rem - v < len(prev) and abs(prev[rem - v] + g[v] - tot[k + 1][rem]) < 1e-9:
                    ud, c, R = per[k]
                    if v > 0:
                        Pset = key_plan(c, R, int(g[v]))
                        for x in Pset: probed[k, ud[x - 1]] = True
                    rem -= v; break
        res["dp_stock"].append(value(probed))
    res["dp_secs"] = time.perf_counter() - t0
    # ---------------- greedy completion over all decisions ----------------
    for B in Bs:
        probed = np.zeros((K, H), bool); spent = 0
        while True:
            un = np.where(probed, 0, dcost)
            pre = np.concatenate([np.zeros((K, 1)), np.cumsum(un, 1)], 1)
            rs = pre[sdec[:, 0], sdec[:, 2] + 1] - pre[sdec[:, 0], sdec[:, 1]]
            gu = np.zeros((G, H)); np.add.at(gu, gid, un); gp = np.concatenate([np.zeros((G, 1)), np.cumsum(gu, 1)], 1)
            rg = gp[gdec[:, 0], gdec[:, 2] + 1] - gp[gdec[:, 0], gdec[:, 1]] if len(gdec) else np.array([])
            allr = np.concatenate([rs, rg]); allr = np.where(allr > 0, allr, np.inf)
            j = int(np.argmin(allr))
            if not np.isfinite(allr[j]) or spent + allr[j] > B: break
            spent += allr[j]
            if j < len(sdec):
                k, a, b = sdec[j]; probed[k, a:b + 1] |= dcost[k, a:b + 1] > 0
            else:
                g, a, b = gdec[j - len(sdec)]
                for k in cal.groups[gnames[g]]: probed[k, a:b + 1] |= dcost[k, a:b + 1] > 0
        res["greedy"].append(value(probed))
    # ---------------- static orders ----------------
    wday = (Ud - Ld); cst = np.maximum(dcost, 1)
    for name, sc in {"widest": wday / cst, "seaq": None}.items():
        if sc is None:
            sc = np.zeros((K, H)); prefW = np.concatenate([np.zeros((K, 1)), np.cumsum(wday, 1)], 1)
            for (k, a, b) in sdec: sc[k, a:b + 1] += wday[k, a:b + 1] / (prefW[k, b + 1] - prefW[k, a])
            gW = np.zeros((G, H)); np.add.at(gW, gid, wday); gpW = np.concatenate([np.zeros((G, 1)), np.cumsum(gW, 1)], 1)
            for (g, a, b) in gdec:
                ks = cal.groups[gnames[g]]; sc[ks, a:b + 1] += wday[ks, a:b + 1] / (gpW[g, b + 1] - gpW[g, a])
            sc = sc / cst
        sc = np.where(dcost > 0, sc, -np.inf); flat = np.argsort(-sc, axis=None, kind="stable"); cc = np.cumsum(dcost.reshape(-1)[flat])
        for B in Bs:
            pr = np.zeros(K * H, bool); pr[flat[cc <= B]] = True; res[name].append(value(pr.reshape(K, H)))
    # ---------------- budget-DP vs value-DP timing with real-valued costs ----------------
    rc = rng.uniform(0.5, 3.0, (K, H)) * (dcost > 0)
    t0 = time.perf_counter(); gsr = [key_dp_value(rc[k, ud].tolist(), R) for (ud, _, R) in per]; totr = combine_value(gsr)
    res["value_dp_real_secs"] = time.perf_counter() - t0
    res["value_dp_real_answers"] = [answer(totr, f * rc.sum()) for f in FRACS]
    out.append(res)
    print({k: v for k, v in res.items()}, flush=True)
json.dump(out, open(r"C:\Users\mailt\Downloads\ICDE-2027\results\rollup.json", "w"))
