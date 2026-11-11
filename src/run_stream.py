"""Experiment B: streaming certified sliding-window threshold monitoring over the real panel.

Each item (stock k, source s, day t) produces events: an initial report (on time, late, or as a
noisy interval), a later refinement to the exact value, and with small probability a
restatement (a correction that contradicts the previously reported exact value, e.g. a trade
bust / restatement). Ground truth = final values after all restatements.

Engines (all incremental over sliding windows W in {5,20}, two threshold levels, stock + industry):
  CERT  : interval bounds [L,U]; emits a decision once certified; contradiction => flagged retraction
  POINT : conventional; missing -> calibration median, noisy -> reading; emits at window close
  WAIT  : emits once every member item is exact (watermark-style completeness)
  RECOMP: same semantics as CERT but recomputes affected window sums from members (for cost only)
Usage: python run_stream.py [seed] [p_late] [p_noisy] [p_restate]
"""
import sys, time, json
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from workload import load_real, Calib, CAL_DAYS, WINDOWS, MULTS

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
P_LATE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
P_NOISY = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
P_RESTATE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.002
EPS = 0.10
rng = np.random.default_rng(seed)
V, industry, keys, dates = load_real()
cal = Calib(V, industry)
K, S, T = V.shape
t0d = CAL_DAYS                       # streaming starts after calibration
gid = np.zeros(K, int); gnames = sorted(cal.groups)
for j, g in enumerate(gnames): gid[cal.groups[g]] = j
G = len(gnames)

# ---------------- event generation ----------------
reported = V.copy()                  # value first reported as exact (may be later restated)
final = V.copy()
restate = rng.random(V.shape) < P_RESTATE
final[restate] = np.round(V[restate] * np.where(rng.random(restate.sum()) < 0.5, 0.6, 1.5))
ev = []                              # (time, k, s, t, lo, hi, kind) kind: 0 report,1 refine,2 restate
for k in range(K):
    for s in range(S):
        for t in range(t0d, T):
            v = reported[k, s, t]
            u = rng.random()
            if u < P_LATE:                                  # late: nothing until exact arrival
                ev.append((t + rng.geometric(0.5) + 0.5, k, s, t, v, v, 1))
            elif u < P_LATE + P_NOISY:                      # noisy interval now, exact later
                r = v * (1 + rng.uniform(-EPS, EPS))
                ev.append((t + 0.5, k, s, t, r / (1 + EPS), r / (1 - EPS), 0))
                ev.append((t + rng.geometric(0.4) + 0.5, k, s, t, v, v, 1))
            else:
                ev.append((t + 0.5, k, s, t, v, v, 0))
            if restate[k, s, t]:
                ev.append((t + rng.integers(16, 30) + 0.7, k, s, t, final[k, s, t], final[k, s, t], 2))
ev.sort(key=lambda e: e[0])
n_ev = len(ev)
print("K", K, "T", T, "events", n_ev, flush=True)

cap = cal.cap                                             # (K,S) declared bound for unreported items
nW = len(WINDOWS)
thr_s = np.array([[c * W * cal.med for c in MULTS] for W in WINDOWS])           # (nW,nC,K)
gmed = np.array([cal.med[cal.groups[g]].sum() for g in gnames])
thr_g = np.array([[c * W * gmed for c in MULTS] for W in WINDOWS])              # (nW,nC,G)


def window_sum(X, W):           # X (...,T) -> sums of windows ending at each t (0 before W-1)
    c = np.cumsum(X, axis=-1)
    out = c.copy(); out[..., W:] = c[..., W:] - c[..., :-W]
    return out


def truth_decisions():
    out = []
    for wi, W in enumerate(WINDOWS):
        ss = window_sum(final.sum(1), W)
        gs = np.zeros((G, T)); np.add.at(gs, gid, ss)
        for ci in range(len(MULTS)):
            out.append((ss > thr_s[wi, ci][:, None], gs > thr_g[wi, ci][:, None]))
    return out
TRUTH = truth_decisions()


class Engine:
    def __init__(self, mode):
        self.mode = mode
        if mode == "point":
            self.lo = np.repeat(cal.src_med[:, :, None], T, 2).astype(float); self.hi = self.lo
        else:
            self.lo = np.zeros((K, S, T)); self.hi = np.repeat(cap[:, :, None], T, 2).astype(float)
        self.exact = np.zeros((K, S, T), bool)
        self.L = {}; self.U = {}; self.GL = {}; self.GU = {}
        for W in WINDOWS:
            self.L[W] = window_sum(self.lo.sum(1), W)
            self.U[W] = self.L[W] if mode == "point" else window_sum(self.hi.sum(1), W)
            self.GL[W] = np.zeros((G, T)); np.add.at(self.GL[W], gid, self.L[W])
            self.GU[W] = self.GL[W] if mode == "point" else (lambda a: (np.add.at(a, gid, self.U[W]), a)[1])(np.zeros((G, T)))
        self.ncontra = 0

    def update(self, k, s, t, lo, hi):
        olo, ohi = self.lo[k, s, t], self.hi[k, s, t]
        contra = (hi < olo - 1e-9) or (lo > ohi + 1e-9)
        if contra: self.ncontra += 1
        dl = lo - olo; g = gid[k]
        if self.mode == "point":
            self.lo[k, s, t] = lo
            for W in WINDOWS:
                self.L[W][k, t:t + W] += dl; self.GL[W][g, t:t + W] += dl
            return contra
        dh = hi - ohi
        self.lo[k, s, t] = lo; self.hi[k, s, t] = hi
        if self.mode == "cert":
            for W in WINDOWS:
                self.L[W][k, t:t + W] += dl; self.U[W][k, t:t + W] += dh
                self.GL[W][g, t:t + W] += dl; self.GU[W][g, t:t + W] += dh
        else:                                    # recompute affected windows from members
            ks = cal.groups[gnames[g]]
            for W in WINDOWS:
                for tt in range(t, min(t + W, T)):
                    a = max(0, tt - W + 1)
                    self.L[W][k, tt] = self.lo[k, :, a:tt + 1].sum(); self.U[W][k, tt] = self.hi[k, :, a:tt + 1].sum()
                    self.GL[W][g, tt] = self.lo[ks, :, a:tt + 1].sum(); self.GU[W][g, tt] = self.hi[ks, :, a:tt + 1].sum()
        return contra


def throughput(mode, n=20000):
    e = Engine(mode); t1 = time.perf_counter()
    for x in ev[:n]:
        e.update(x[1], x[2], x[3], x[4], x[5])
    return n / (time.perf_counter() - t1)


def simulate(mode):
    """Day-by-day emission. Returns per-decision emission day, value at emission, reversals."""
    e = Engine("point" if mode == "point" else "cert")
    nd = [(wi, ci, lvl) for wi in range(nW) for ci in range(len(MULTS)) for lvl in (0, 1)]
    shp = {0: (K, T), 1: (G, T)}
    emit_day = {x: np.full(shp[x[2]], -1) for x in nd}
    emit_val = {x: np.zeros(shp[x[2]], bool) for x in nd}
    cur_val = {x: np.zeros(shp[x[2]], bool) for x in nd}
    reversals = {x: 0 for x in nd}; flagged = {x: 0 for x in nd}
    nexact_k = np.zeros((K, T), int)        # number of exact sources per key-day
    j = 0; t_upd = 0.0
    last_contra_day = np.full((K, T), -10**9)
    for day in range(t0d, T + 32):
        t1 = time.perf_counter()
        while j < n_ev and ev[j][0] < day + 1:
            _, k, s, t, lo, hi, kind = ev[j]
            c = e.update(k, s, t, lo, hi)
            if lo == hi and not e.exact[k, s, t]:
                e.exact[k, s, t] = True; nexact_k[k, t] += 1
            if c: last_contra_day[k, t] = day
            j += 1
        t_upd += time.perf_counter() - t1
        tcol = min(day, T - 1)
        for (wi, ci, lvl) in nd:
            W = WINDOWS[wi]; x = (wi, ci, lvl)
            if lvl == 0:
                L, U, th = e.L[W][:, :tcol + 1], e.U[W][:, :tcol + 1], thr_s[wi, ci][:, None]
            else:
                L, U, th = e.GL[W][:, :tcol + 1], e.GU[W][:, :tcol + 1], thr_g[wi, ci][:, None]
            valid = np.zeros(L.shape, bool); valid[:, max(t0d + W - 1, 0):] = True
            if mode == "point":
                cert = valid; val = L > th
            elif mode == "cert":
                cert = valid & ((L > th) | (U <= th)); val = L > th
            else:   # wait: all members exact
                comp = (nexact_k[:, :tcol + 1] == S).astype(int)
                full = window_sum(comp, W) == W
                if lvl == 1:
                    gfull = np.zeros(L.shape); np.add.at(gfull, gid, full.astype(float))
                    gsz = np.bincount(gid, minlength=G)[:, None]
                    full = gfull == gsz
                cert = valid & full; val = L > th
            ed, ev_, cv = emit_day[x][:, :tcol + 1], emit_val[x][:, :tcol + 1], cur_val[x][:, :tcol + 1]
            new = cert & (ed < 0)
            ed[new] = day; ev_[new] = val[new]; cv[new] = val[new]
            old = (ed >= 0) & (ed < day) & cert
            flip = old & (val != cv)
            reversals[x] += int(flip.sum()); cv[flip] = val[flip]
            if mode != "point":
                # a CERT/WAIT reversal is only possible after a contradiction; count flagged ones
                flagged[x] += int(flip.sum())
            # a certified decision that loses certification (after contradiction) is retracted
            if mode == "cert":
                lost = (ed >= 0) & ~cert & valid
                if lost.any():
                    flagged[x] += int(lost.sum()); reversals[x] += int(lost.sum())
                    ed[lost] = -1
    res = {"upd_secs": t_upd}
    tot = wrong_at_emit = emitted = delay_sum = rev = flg = 0
    on_time = 0
    for x in nd:
        wi, ci, lvl = x; W = WINDOWS[wi]
        tr = TRUTH[wi * len(MULTS) + ci][lvl][:, t0d + W - 1:]
        ed = emit_day[x][:, t0d + W - 1:]; evv = emit_val[x][:, t0d + W - 1:]
        tt = np.arange(t0d + W - 1, T)[None, :]
        m = ed >= 0
        tot += m.size; emitted += int(m.sum())
        wrong_at_emit += int((evv[m] != tr[m]).sum())
        delay_sum += float((ed - tt)[m].sum()); on_time += int(((ed - tt) <= 0)[m].sum())
        rev += reversals[x]; flg += flagged[x]
    res.update(total=tot, emitted=emitted, wrong_at_emission=wrong_at_emit, mean_delay_days=delay_sum / max(emitted, 1),
               on_time=on_time, reversals=rev, flagged=flg, contradictions=e.ncontra)
    return res


out = {"seed": seed, "P_LATE": P_LATE, "P_NOISY": P_NOISY, "P_RESTATE": P_RESTATE, "K": K, "T": T,
       "events": n_ev, "restated_items": int(restate[:, :, t0d:].sum()),
       "bound_violations": int((final[:, :, t0d:] > cap[:, :, None]).sum())}
for mode in ["point", "cert", "recomp"]:
    out[f"tput_{mode}"] = throughput(mode, 20000 if mode != "recomp" else 3000)
    print(mode, "events/s", round(out[f"tput_{mode}"]), flush=True)
for mode in ["cert", "point", "wait"]:
    t1 = time.perf_counter(); r = simulate(mode); r["wall"] = time.perf_counter() - t1
    out[mode] = r; print(mode, r, flush=True)
import workload as _w; tag = f"s{seed}_l{P_LATE}_n{P_NOISY}_r{P_RESTATE}" + ("" if _w.CAP_MULT == 3.0 else f"_cap{_w.CAP_MULT}")
json.dump(out, open(rf"C:\Users\mailt\Downloads\ICDE-2027\results\stream_{tag}.json", "w"), indent=1)
