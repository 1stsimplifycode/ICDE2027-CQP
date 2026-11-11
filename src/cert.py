"""Certified threshold decisions over interval-annotated evidence (core library).

An evidence item i carries a sound interval [lo_i, hi_i] that is assumed to contain the
true value v_i (exact items have lo_i == hi_i; missing items carry a declared domain bound).
A decision d is a SUM-threshold predicate  sum_{i in S_d} v_i > theta_d  over an item set S_d.
Three-valued semantics:
    TRUE  if L_d = sum lo_i >  theta_d   (true for every completion consistent with evidence)
    FALSE if U_d = sum hi_i <= theta_d
    UNK   otherwise (evidence insufficient)
Items are shared by many decisions (overlapping windows, several thresholds, roll-ups);
membership is a sparse decision x item incidence matrix.
"""
import numpy as np
import scipy.sparse as sp

TRUE, FALSE, UNK = 1, 0, -1


class DecisionSet:
    def __init__(self, M, theta, lo, hi):
        # M: csr (n_dec x n_items) 0/1 incidence; theta: (n_dec,)
        self.M = M.tocsr().astype(np.float64)
        self.MT = self.M.T.tocsr()
        self.theta = np.asarray(theta, float)
        self.lo = np.asarray(lo, float).copy()
        self.hi = np.asarray(hi, float).copy()
        self.L = self.M @ self.lo
        self.U = self.M @ self.hi
        coo = self.M.tocoo()
        self.nz_d, self.nz_i = coo.row.astype(np.int64), coo.col.astype(np.int64)

    # ---- semantics -------------------------------------------------------
    def state(self):
        s = np.full(len(self.theta), UNK, dtype=np.int8)
        s[self.L > self.theta] = TRUE
        s[self.U <= self.theta] = FALSE
        return s

    # ---- incremental maintenance: one item's evidence changes -------------
    def set_interval(self, i, lo, hi):
        """O(deg(i)) update. Returns decisions whose L/U changed."""
        dlo, dhi = lo - self.lo[i], hi - self.hi[i]
        a, b = self.MT.indptr[i], self.MT.indptr[i + 1]
        ds = self.MT.indices[a:b]
        self.L[ds] += dlo
        self.U[ds] += dhi
        self.lo[i], self.hi[i] = lo, hi
        return ds

    def refine(self, i, v):
        return self.set_interval(i, v, v)


def check_contradiction(old_lo, old_hi, new_lo, new_hi, tol=1e-9):
    """New evidence for an item is contradictory if its interval is disjoint from the old one.
    Narrowing (new within old) can never flip a certified decision (monotonicity)."""
    return new_hi < old_lo - tol or new_lo > old_hi + tol
