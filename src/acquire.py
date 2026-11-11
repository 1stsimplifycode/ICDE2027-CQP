"""Budgeted evidence acquisition for many SUM-threshold decisions over shared evidence.

A probe of item i costs c_i and reveals v_i exactly. Policies choose probes adaptively in
small batches; after each batch the decision bounds are updated incrementally (cert.DecisionSet).

SEAQ (Shared-Evidence Acquisition for Queries), proposed (frozen form after dev log D1):
  score(i) = sum_{d undetermined, i in S_d} (w_i / W_d) / c_i
Original design (now ablation 'seaq_myopic'):
  score(i) = sum_{d undetermined, i in S_d} [ P1(d,i) + lam * w_i / W_d ] / c_i
  P1(d,i)  = (max(0, w_i - g_L) + max(0, w_i - g_U)) / w_i   (closed form, X_i ~ U[lo_i,hi_i])
  g_L = theta_d - L_d, g_U = U_d - theta_d, W_d = U_d - L_d.
P1 is the exact one-probe resolution probability under a uniform-in-interval model; the
w_i/W_d term is the fraction of d's remaining uncertainty removed (multi-probe progress).
"""
import numpy as np
from cert import DecisionSet, UNK

POLICIES = ["random", "widest", "degree", "local", "voi1", "seaq", "seaq_myopic", "seaq_max", "comp", "hybrid"]


def _scores(ds, policy, cost, lam, rng):
    st = ds.state()
    und_d = st == UNK
    w_all = ds.hi - ds.lo
    m = und_d[ds.nz_d] & (w_all[ds.nz_i] > 0)
    d, i = ds.nz_d[m], ds.nz_i[m]
    n = len(ds.lo)
    if len(i) == 0:
        return None
    if policy == "random":
        s = np.zeros(n); s[np.unique(i)] = rng.random(len(np.unique(i))) + 1
        return s
    if policy == "widest":
        s = np.zeros(n); s[i] = w_all[i] / cost[i]
        return s
    if policy == "degree":
        return np.bincount(i, minlength=n) / cost
    if policy == "local":
        WD = ds.U - ds.L
        gap = np.minimum(ds.theta - ds.L, ds.U - ds.theta) / np.maximum(WD, 1e-12)
        gap_nz = gap[d]
        dstar = d[np.argmin(gap_nz)]
        sel = d == dstar
        s = np.zeros(n); s[i[sel]] = w_all[i[sel]] / cost[i[sel]]
        return s
    w = w_all[i]
    gL = ds.theta[d] - ds.L[d]
    gU = ds.U[d] - ds.theta[d]
    WD = ds.U[d] - ds.L[d]
    p1 = (np.maximum(0, w - gL) + np.maximum(0, w - gU)) / w
    prog = w / WD
    if policy == "seaq":            # frozen main method (dev log D1): shared progress score
        return np.bincount(i, weights=prog, minlength=n) / cost
    if policy == "seaq_myopic":     # original design, lam=0.5 (ablation)
        return np.bincount(i, weights=p1 + lam * prog, minlength=n) / cost
    if policy == "voi1":            # myopic one-probe value of information (baseline)
        s = np.bincount(i, weights=p1, minlength=n)
        s[i] += 1e-9 * w / WD          # tie-break among candidates only (P1 is often 0)
        return s / cost
    if policy == "seaq_max":        # no sharing: best single decision (ablation)
        s = np.zeros(n); np.maximum.at(s, i, prog)
        return s / cost
    if policy in ("comp", "hybrid"):  # completion-oriented (round 3): credit 1/r_d, r_d = #uncertain items left in d
        unc = (w > 0).astype(float)
        r = np.bincount(d, weights=unc, minlength=len(ds.theta))
        comp = np.bincount(i, weights=1.0 / np.maximum(r[d], 1), minlength=n) / cost
        if policy == "comp":
            return comp
        sq = np.bincount(i, weights=prog, minlength=n) / cost
        return sq / max(sq.max(), 1e-12) + comp / max(comp.max(), 1e-12)
    raise ValueError(policy)


def run_policy(M, theta, lo, hi, truth, cost, policy, budget, batch=8, lam=0.5, seed=0):
    """Returns arrays (cum_cost, n_resolved, n_wrong_certified) after each batch."""
    rng = np.random.default_rng(seed)
    ds = DecisionSet(M, theta, lo, hi)
    tv = (M @ truth) > theta
    spent, curve = 0.0, []

    def record():
        st = ds.state()
        res = st != UNK
        wrong = int(((st == 1) & ~tv).sum() + ((st == 0) & tv).sum())
        curve.append((spent, int(res.sum()), wrong))
        return res.all()

    done = record()
    while spent < budget and not done:
        s = _scores(ds, policy, cost, lam, rng)
        if s is None:
            break
        cand = np.flatnonzero(s > 0)
        if len(cand) == 0:
            break
        top = cand[np.argsort(-s[cand], kind="stable")[:batch]]
        for j in top:
            if spent + cost[j] > budget:
                continue
            ds.refine(j, truth[j]); spent += cost[j]
        done = record()
        if all(spent + cost[j] > budget for j in top):
            break
    return np.array(curve)
